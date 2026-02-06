# analysis/position_delta_optimizer.py
"""
Position Delta Threshold Optimizer voor QBN v3.2.

MI Grid Search voor optimalisatie van delta thresholds in Position Management.
Delta scores meten verandering in coincident/confirming composites sinds trade entry.

DOEL:
- Vind optimale thresholds voor discretisatie van delta scores
- Maximaliseer Mutual Information tussen delta states en barrier outcomes
- Pas uniqueness weighting toe (López de Prado) om serial correlation te elimineren

DELTA STATES:
- deteriorating: delta < -threshold (situatie verslechtert voor positie)
- stable: -threshold <= delta <= +threshold
- improving: delta > +threshold (situatie verbetert voor positie)

USAGE:
    from analysis.position_delta_optimizer import PositionDeltaThresholdOptimizer
    
    optimizer = PositionDeltaThresholdOptimizer(asset_id=1)
    results = optimizer.analyze()
    optimizer.save_results(results)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from sklearn.metrics import mutual_info_score

from database.db import get_cursor
from core.config_defaults import (
    DEFAULT_DELTA_THRESHOLD_COINCIDENT,
    DEFAULT_DELTA_THRESHOLD_CONFIRMING
)

logger = logging.getLogger(__name__)

# Check matplotlib availability (headless mode)
try:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib/seaborn not available - plotting disabled")


# Delta states
DELTA_STATES = ['deteriorating', 'stable', 'improving']

# Diversity constraints
MIN_ACTIVE_STATES = 2          # Minimaal 2 states met >5% representatie
MIN_STATE_THRESHOLD = 0.05     # Threshold voor "actieve" state (5%)
MAX_STABLE_PERCENT = 0.80      # Stable state maximaal 80%


@dataclass
class DeltaThresholdResult:
    """Resultaat van delta threshold optimalisatie."""
    delta_type: str  # 'cumulative' of 'instantaneous'
    score_type: str  # 'coincident' of 'confirming'
    optimal_threshold: float
    mi_score: float
    distribution: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Converteer naar dictionary voor JSON/DB opslag."""
        return {
            'delta_type': self.delta_type,
            'score_type': self.score_type,
            'optimal_threshold': self.optimal_threshold,
            'mi_score': self.mi_score,
            'distribution': self.distribution,
            'metadata': self.metadata
        }


class PositionDeltaThresholdOptimizer:
    """
    MI Grid Search voor Position Management delta thresholds.
    
    Laadt event-gelabelde data met delta scores en zoekt optimale
    thresholds die MI tussen delta states en outcomes maximaliseren.
    """
    
    # Grid search parameters
    DELTA_THRESHOLD_VALUES = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    
    def __init__(
        self,
        asset_id: int,
        lookback_days: Optional[int] = None,  # None = alle data
        output_dir: Optional[Path] = None,
        enforce_diversity: bool = True
    ):
        """
        Initialiseer optimizer.
        
        Args:
            asset_id: Asset ID waarvoor thresholds worden geoptimaliseerd
            lookback_days: Aantal dagen terug voor data (None = alle beschikbare event data)
            output_dir: Output directory voor reports/heatmaps
            enforce_diversity: Pas diversity constraints toe
        """
        self.asset_id = asset_id
        self.lookback_days = lookback_days  # None = all data
        self.output_dir = output_dir or Path("_validation/position_delta_analysis")
        self.enforce_diversity = enforce_diversity
        
        self._event_data: Optional[pd.DataFrame] = None
    
    def load_event_data(self) -> pd.DataFrame:
        """
        Laad event-gelabelde data met delta scores.
        
        Vereist dat EventWindowDetector al is gedraaid met delta score berekening.
        
        Returns:
            DataFrame met kolommen:
            - event_id, event_outcome, event_direction
            - delta_cum_coincident, delta_cum_confirming
            - uniqueness_weight
        """
        lookback_desc = f"{self.lookback_days} days" if self.lookback_days else "all data"
        logger.info(f"Loading event data for asset {self.asset_id} (lookback={lookback_desc})")
        
        # REASON: Laad event-gelabelde barrier outcomes.
        # Composite scores worden LOKAAL berekend via _calculate_delta_scores_from_events()
        query = """
            SELECT 
                bo.time_1,
                bo.event_id,
                bo.first_significant_barrier as barrier_outcome,
                ew.direction as event_direction,
                ew.outcome as event_outcome,
                ew.start_time as event_start_time
            FROM qbn.barrier_outcomes bo
            INNER JOIN qbn.event_windows ew ON bo.event_id = ew.event_id
            WHERE bo.asset_id = %(asset_id)s
              AND bo.event_id IS NOT NULL
            ORDER BY bo.time_1
        """
        
        try:
            with get_cursor() as cur:
                cur.execute(query, {'asset_id': self.asset_id})
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            
            df = pd.DataFrame(rows, columns=columns)
            logger.info(f"Loaded {len(df)} event-labeled rows")
            
            if df.empty:
                logger.warning("No event data found. Run EventWindowDetector first.")
                return df
            
            # REASON: We moeten de delta scores berekenen per event.
            # Voor elke rij in een event berekenen we de delta t.o.v. de entry score.
            logger.info("Calculating delta scores from event data...")
            df = self._calculate_delta_scores_from_events(df)
            
            if df.empty:
                logger.warning("Could not calculate delta scores.")
                return df
            
            # Bereken uniqueness weight
            df['uniqueness_weight'] = df.groupby('event_id')['event_id'].transform(
                lambda x: 1.0 / len(x) if len(x) > 0 else 0.0
            )
            
            # Map outcome naar target
            df['target_hit'] = df['event_outcome'].apply(self._is_target_hit)
            
            self._event_data = df
            
            logger.info(f"Event data prepared: {len(df)} rows, "
                       f"{df['event_id'].nunique()} unique events, "
                       f"target_hit_rate={df['target_hit'].mean():.1%}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading event data: {e}")
            return pd.DataFrame()
    
    def _calculate_delta_scores_from_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereken delta scores door composite scores LOKAAL te berekenen.
        
        ARCHITECTUUR: 
        - Simpele SELECTs om ruwe signal data te laden
        - Composite scores worden LOKAAL berekend (GPU node, NumPy)
        - GEEN complexe JOINs op de database
        """
        if df.empty:
            return df
        
        logger.info("Loading signal data and calculating composite scores locally...")
        
        try:
            # 1. Laad signal configuration (lichtgewicht query)
            signal_config = self._load_signal_config()
            
            if not signal_config['coincident'] and not signal_config['confirming']:
                logger.warning("No COINCIDENT/CONFIRMING signals configured")
                return pd.DataFrame()
            
            # 2. Laad ruwe signal data (simpele SELECTs, geen JOINs)
            coin_data = self._load_signal_data('kfl.mtf_signals_coin', signal_config['coincident'])
            conf_data = self._load_signal_data('kfl.mtf_signals_conf', signal_config['confirming'])
            
            if coin_data.empty and conf_data.empty:
                logger.warning("No signal data found")
                return pd.DataFrame()
            
            # 3. Bereken composite scores LOKAAL (NumPy op GPU node)
            scores_df = self._calculate_composite_scores_local(coin_data, conf_data, signal_config)
            
            if scores_df.empty:
                logger.warning("Could not calculate composite scores")
                return pd.DataFrame()
            
            logger.info(f"Calculated composite scores for {len(scores_df)} timestamps")
            
            # 4. Merge met event data
            df['time_1'] = pd.to_datetime(df['time_1'])
            scores_df['time_1'] = pd.to_datetime(scores_df['time_1'])
            
            df = df.merge(
                scores_df[['time_1', 'coincident_score', 'confirming_score']],
                on='time_1',
                how='left'
            )
            
            # Fill NaN met 0 voor ontbrekende scores
            df['coincident_score'] = df['coincident_score'].fillna(0.0)
            df['confirming_score'] = df['confirming_score'].fillna(0.0)
            
            # 5. Bereken delta scores per event (lokaal, NumPy)
            df = self._calculate_deltas_per_event(df)
            
            logger.info(f"Delta scores calculated for {df['event_id'].nunique()} events")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating delta scores: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _load_signal_config(self) -> Dict[str, List[Dict]]:
        """
        Laad signal configuratie (lichtgewicht query).
        
        Returns:
            Dict met 'coincident' en 'confirming' signal configs
        """
        # REASON: qbn.signal_classification heeft geen 'weight' of 'is_active' kolom
        query = """
            SELECT signal_name, semantic_class, 
                   COALESCE(polarity, 'neutral') as polarity
            FROM qbn.signal_classification
            WHERE semantic_class IN ('COINCIDENT', 'CONFIRMING')
        """
        
        polarity_map = {'bullish': 1, 'bearish': -1, 'neutral': 0}
        
        with get_cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        
        config = {'coincident': [], 'confirming': []}
        
        for signal_name, semantic_class, polarity in rows:
            sig_config = {
                'signal_name': signal_name.lower(),
                'polarity': polarity_map.get(polarity.lower(), 0)
            }
            if semantic_class == 'COINCIDENT':
                config['coincident'].append(sig_config)
            else:
                config['confirming'].append(sig_config)
        
        logger.info(f"Loaded {len(config['coincident'])} COINCIDENT, {len(config['confirming'])} CONFIRMING signals")
        return config
    
    def _load_signal_data(self, table: str, signal_configs: List[Dict]) -> pd.DataFrame:
        """
        Laad ruwe signal data (simpele SELECT, geen JOINs).
        
        ARCHITECTUUR: Data wordt naar GPU node geladen, berekening gebeurt lokaal.
        """
        if not signal_configs:
            return pd.DataFrame()
        
        # Bepaal signal kolommen (met _60 suffix voor 60-min interval)
        signal_cols = [f"{s['signal_name']}_60" for s in signal_configs]
        
        # Simpele SELECT - geen JOINs
        query = f"""
            SELECT time_1, {', '.join(signal_cols)}
            FROM {table}
            WHERE asset_id = %(asset_id)s
            ORDER BY time_1
        """
        
        try:
            with get_cursor() as cur:
                cur.execute(query, {'asset_id': self.asset_id})
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            
            df = pd.DataFrame(rows, columns=columns)
            logger.debug(f"Loaded {len(df)} rows from {table}")
            return df
            
        except Exception as e:
            logger.warning(f"Could not load from {table}: {e}")
            return pd.DataFrame()
    
    def _calculate_composite_scores_local(
        self, 
        coin_data: pd.DataFrame, 
        conf_data: pd.DataFrame,
        signal_config: Dict
    ) -> pd.DataFrame:
        """
        Bereken composite scores LOKAAL met NumPy (GPU node).
        
        PERFORMANCE: Vectorized berekening, geen database overhead.
        """
        result_frames = []
        
        # Coincident score
        if not coin_data.empty and signal_config['coincident']:
            coin_scores = self._aggregate_signals(
                coin_data, 
                signal_config['coincident'],
                score_name='coincident_score'
            )
            result_frames.append(coin_scores)
        
        # Confirming score  
        if not conf_data.empty and signal_config['confirming']:
            conf_scores = self._aggregate_signals(
                conf_data,
                signal_config['confirming'], 
                score_name='confirming_score'
            )
            result_frames.append(conf_scores)
        
        if not result_frames:
            return pd.DataFrame()
        
        # Merge frames
        result = result_frames[0]
        for frame in result_frames[1:]:
            result = result.merge(frame, on='time_1', how='outer')
        
        # Fill NaN
        result = result.fillna(0.0)
        
        return result
    
    def _aggregate_signals(
        self, 
        data: pd.DataFrame, 
        signal_configs: List[Dict],
        score_name: str
    ) -> pd.DataFrame:
        """
        Aggregeer signalen naar composite score (NumPy vectorized).
        
        Formule: sum(signal * polarity) / n_signals
        """
        # Initialiseer score array
        n_rows = len(data)
        scores = np.zeros(n_rows)
        valid_signals = 0
        
        for sig in signal_configs:
            col_name = f"{sig['signal_name']}_60"
            if col_name in data.columns:
                signal_values = data[col_name].fillna(0).values
                scores += signal_values * sig['polarity']
                valid_signals += 1
        
        # Normaliseer
        if valid_signals > 0:
            scores = scores / valid_signals
        
        return pd.DataFrame({
            'time_1': data['time_1'],
            score_name: scores
        })
    
    def _calculate_deltas_per_event(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereken delta scores per event (NumPy vectorized).
        """
        df['delta_cum_coincident'] = 0.0
        df['delta_cum_confirming'] = 0.0
        df['entry_coincident_score'] = 0.0
        df['entry_confirming_score'] = 0.0
        
        for event_id in df['event_id'].unique():
            mask = df['event_id'] == event_id
            event_rows = df[mask].sort_values('time_1')
            
            if len(event_rows) == 0:
                continue
            
            # Entry scores zijn de eerste rij van het event
            entry_coinc = event_rows.iloc[0]['coincident_score']
            entry_conf = event_rows.iloc[0]['confirming_score']
            direction = event_rows.iloc[0]['event_direction']
            
            # Bereken deltas
            df.loc[mask, 'entry_coincident_score'] = entry_coinc
            df.loc[mask, 'entry_confirming_score'] = entry_conf
            df.loc[mask, 'delta_cum_coincident'] = df.loc[mask, 'coincident_score'] - entry_coinc
            df.loc[mask, 'delta_cum_confirming'] = df.loc[mask, 'confirming_score'] - entry_conf
            
            # Direction-aware
            if direction == 'short':
                df.loc[mask, 'delta_cum_coincident'] *= -1
                df.loc[mask, 'delta_cum_confirming'] *= -1
        
        return df
    
    def _is_target_hit(self, outcome: str) -> int:
        """Map event outcome naar binary target (1 = gunstig, 0 = ongunstig)."""
        if outcome is None:
            return 0
        
        # Voor long: up = gunstig, voor short: down = gunstig
        # Maar we hebben al direction-aware deltas, dus we kijken naar strong outcomes
        favorable = ['up_strong', 'down_strong']  # Sterke beweging in verwachte richting
        return 1 if outcome in favorable else 0
    
    def analyze(self) -> Dict[str, DeltaThresholdResult]:
        """
        Voer MI Grid Search uit voor alle delta/score combinaties.
        
        Returns:
            Dict met resultaten per combinatie (key: 'cumulative_coincident', etc.)
        """
        if self._event_data is None or self._event_data.empty:
            self.load_event_data()
        
        if self._event_data is None or self._event_data.empty:
            logger.error("No event data available for analysis")
            return {}
        
        results = {}
        
        # Analyseer cumulative deltas (meest relevant voor position management)
        for score_type in ['coincident', 'confirming']:
            result = self._analyze_single(
                delta_type='cumulative',
                score_type=score_type
            )
            if result:
                key = f"cumulative_{score_type}"
                results[key] = result
        
        return results
    
    def _analyze_single(
        self,
        delta_type: str,
        score_type: str
    ) -> Optional[DeltaThresholdResult]:
        """
        Analyseer een enkele delta/score combinatie.
        
        Args:
            delta_type: 'cumulative' of 'instantaneous'
            score_type: 'coincident' of 'confirming'
            
        Returns:
            DeltaThresholdResult of None bij fout
        """
        logger.info(f"Analyzing {delta_type} {score_type} delta thresholds...")
        
        df = self._event_data
        delta_col = f"delta_{delta_type[:3]}_{score_type}"  # delta_cum_coincident
        
        if delta_col not in df.columns:
            logger.error(f"Column {delta_col} not found in data")
            return None
        
        # Filter NaN values
        valid_mask = df[delta_col].notna() & df['target_hit'].notna()
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) < 100:
            logger.warning(f"Insufficient data for {delta_type} {score_type}: {len(valid_df)} rows")
            return None
        
        # Log delta distributie
        delta_stats = valid_df[delta_col].describe()
        logger.info(f"Delta distribution: mean={delta_stats['mean']:.4f}, "
                   f"std={delta_stats['std']:.4f}, "
                   f"min={delta_stats['min']:.4f}, max={delta_stats['max']:.4f}")
        
        # Grid search
        best_threshold = DEFAULT_DELTA_THRESHOLD_COINCIDENT if score_type == 'coincident' else DEFAULT_DELTA_THRESHOLD_CONFIRMING
        best_mi = -1.0
        best_distribution = {}
        valid_combinations = 0
        
        results_array = []
        
        for threshold in self.DELTA_THRESHOLD_VALUES:
            # Discretiseer met uniqueness weighting
            states = valid_df[delta_col].apply(
                lambda x: self._discretize_delta(x, threshold)
            )
            
            # Bereken gewogen MI
            mi, distribution = self._weighted_mutual_info(
                states, 
                valid_df['target_hit'],
                valid_df['uniqueness_weight']
            )
            
            results_array.append({
                'threshold': threshold,
                'mi': mi,
                'distribution': distribution
            })
            
            # Check diversity constraints
            is_valid = self._check_diversity(distribution)
            
            if is_valid:
                valid_combinations += 1
                
                if mi > best_mi:
                    best_mi = mi
                    best_threshold = threshold
                    best_distribution = distribution
        
        logger.info(f"Grid search complete. Valid: {valid_combinations}/{len(self.DELTA_THRESHOLD_VALUES)}")
        logger.info(f"Best: threshold={best_threshold}, MI={best_mi:.4f}")
        logger.info(f"Distribution: {', '.join(f'{k}={v:.1%}' for k, v in sorted(best_distribution.items()))}")
        
        # Genereer heatmap
        if HAS_PLOTTING:
            self._generate_heatmap(
                results_array, delta_type, score_type, best_threshold
            )
        
        return DeltaThresholdResult(
            delta_type=delta_type,
            score_type=score_type,
            optimal_threshold=best_threshold,
            mi_score=best_mi,
            distribution=best_distribution,
            metadata={
                'n_samples': len(valid_df),
                'n_events': valid_df['event_id'].nunique(),
                'valid_combinations': valid_combinations,
                'delta_stats': delta_stats.to_dict(),
                'all_results': results_array
            }
        )
    
    def _discretize_delta(self, delta: float, threshold: float) -> str:
        """Discretiseer delta naar state."""
        if delta < -threshold:
            return 'deteriorating'
        elif delta > threshold:
            return 'improving'
        return 'stable'
    
    def _weighted_mutual_info(
        self,
        states: pd.Series,
        targets: pd.Series,
        weights: pd.Series
    ) -> Tuple[float, Dict[str, float]]:
        """
        Bereken gewogen Mutual Information.
        
        Gebruikt uniqueness weights om serial correlation te corrigeren.
        
        Returns:
            Tuple van (MI score, state distribution)
        """
        # Bereken gewogen state distribution
        distribution = {}
        total_weight = weights.sum()
        
        for state in DELTA_STATES:
            mask = states == state
            state_weight = weights[mask].sum()
            distribution[state] = state_weight / total_weight if total_weight > 0 else 0
        
        # Voor MI gebruiken we sklearn maar met gewogen samples
        # Simpele benadering: sample met replacement volgens weights
        if total_weight > 0:
            # Normaliseer weights
            norm_weights = weights / weights.sum()
            
            # Bootstrap sample voor MI berekening
            n_samples = min(len(states), 10000)
            indices = np.random.choice(
                len(states), 
                size=n_samples, 
                replace=True, 
                p=norm_weights.values
            )
            
            sampled_states = states.iloc[indices]
            sampled_targets = targets.iloc[indices]
            
            mi = mutual_info_score(sampled_states, sampled_targets)
        else:
            mi = 0.0
        
        return mi, distribution
    
    def _check_diversity(self, distribution: Dict[str, float]) -> bool:
        """Check diversity constraints."""
        if not self.enforce_diversity:
            return True
        
        # Check minimaal aantal actieve states
        active_states = sum(1 for v in distribution.values() if v > MIN_STATE_THRESHOLD)
        if active_states < MIN_ACTIVE_STATES:
            return False
        
        # Check stable niet te dominant
        stable_pct = distribution.get('stable', 0)
        if stable_pct > MAX_STABLE_PERCENT:
            return False
        
        return True
    
    def _generate_heatmap(
        self,
        results: List[Dict],
        delta_type: str,
        score_type: str,
        best_threshold: float
    ):
        """Genereer en sla MI heatmap op."""
        output_dir = self.output_dir / f"asset_{self.asset_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Maak bar chart van MI per threshold
        fig, ax = plt.subplots(figsize=(10, 6))
        
        thresholds = [r['threshold'] for r in results]
        mi_scores = [r['mi'] for r in results]
        
        colors = ['green' if t == best_threshold else 'steelblue' for t in thresholds]
        
        bars = ax.bar(range(len(thresholds)), mi_scores, color=colors)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
        ax.set_xlabel('Delta Threshold')
        ax.set_ylabel('Mutual Information')
        ax.set_title(f'Delta Threshold Optimization - {delta_type} {score_type}\n'
                    f'Asset {self.asset_id} | Best: {best_threshold} (MI={max(mi_scores):.4f})')
        
        # Voeg distributie labels toe
        for i, (bar, result) in enumerate(zip(bars, results)):
            dist = result['distribution']
            label = f"d:{dist.get('deteriorating', 0):.0%}\ns:{dist.get('stable', 0):.0%}\ni:{dist.get('improving', 0):.0%}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   label, ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        
        output_path = output_dir / f'delta_mi_{delta_type}_{score_type}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved heatmap to {output_path}")
    
    def save_results(self, results: Dict[str, DeltaThresholdResult], run_id: Optional[str] = None):
        """
        Sla resultaten op in database en genereer report.
        
        Args:
            results: Dict met DeltaThresholdResult per combinatie
            run_id: Optional run ID for traceability
        """
        if not results:
            logger.warning("No results to save")
            return
        
        # Save to database
        self._save_to_db(results, run_id=run_id)
        
        # Generate markdown report
        self._generate_report(results)
    
    def _save_to_db(self, results: Dict[str, DeltaThresholdResult], run_id: Optional[str] = None):
        """Sla thresholds op in qbn.position_delta_threshold_config."""
        import json
        
        logger.info(f"Saving {len(results)} delta thresholds to database...")
        
        with get_cursor(commit=True) as cur:
            for key, result in results.items():
                cur.execute("""
                    INSERT INTO qbn.position_delta_threshold_config 
                        (asset_id, delta_type, score_type, threshold, mi_score, distribution, source_method, run_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (asset_id, delta_type, score_type) 
                    DO UPDATE SET 
                        threshold = EXCLUDED.threshold,
                        mi_score = EXCLUDED.mi_score,
                        distribution = EXCLUDED.distribution,
                        source_method = EXCLUDED.source_method,
                        run_id = EXCLUDED.run_id,
                        updated_at = NOW()
                """, (
                    self.asset_id,
                    result.delta_type,
                    result.score_type,
                    result.optimal_threshold,
                    result.mi_score,
                    json.dumps(result.distribution),
                    'MI Grid Search',
                    run_id
                ))
        
        logger.info(f"✅ Delta thresholds saved for asset {self.asset_id}")
        
        # HANDSHAKE_OUT logging
        from core.step_validation import log_handshake_out
        log_handshake_out(
            step="run_position_delta_threshold_analysis",
            target="qbn.position_delta_threshold_config",
            run_id=run_id or "N/A",
            rows=len(results),
            operation="INSERT"
        )
    
    def _generate_report(self, results: Dict[str, DeltaThresholdResult]):
        """Genereer markdown report."""
        output_dir = self.output_dir / f"asset_{self.asset_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        lookback_desc = f"{self.lookback_days} days" if self.lookback_days else "all data"
        
        report_lines = [
            f"# Position Delta Threshold Analysis Report",
            f"",
            f"**Asset ID:** {self.asset_id}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Lookback:** {lookback_desc}",
            f"",
            f"---",
            f"",
            f"## Optimal Thresholds",
            f"",
            f"| Delta Type | Score Type | Threshold | MI Score | Distribution |",
            f"|------------|------------|-----------|----------|--------------|",
        ]
        
        for key, result in results.items():
            dist_str = f"d:{result.distribution.get('deteriorating', 0):.0%} s:{result.distribution.get('stable', 0):.0%} i:{result.distribution.get('improving', 0):.0%}"
            report_lines.append(
                f"| {result.delta_type} | {result.score_type} | {result.optimal_threshold:.3f} | {result.mi_score:.4f} | {dist_str} |"
            )
        
        report_lines.extend([
            f"",
            f"---",
            f"",
            f"## Methodology",
            f"",
            f"1. **Data Source:** Event-labeled barrier outcomes with delta scores",
            f"2. **Delta Calculation:** Cumulative change since entry (direction-aware)",
            f"3. **Weighting:** Uniqueness weighting (1/N per event) to eliminate serial correlation",
            f"4. **Optimization:** MI Grid Search over threshold values",
            f"5. **Diversity Constraints:** Min {MIN_ACTIVE_STATES} active states, max {MAX_STABLE_PERCENT:.0%} stable",
            f"",
            f"## Delta States",
            f"",
            f"- **deteriorating:** delta < -threshold (situatie verslechtert)",
            f"- **stable:** -threshold <= delta <= +threshold",
            f"- **improving:** delta > +threshold (situatie verbetert)",
            f"",
        ])
        
        report_path = output_dir / f"delta_threshold_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {report_path}")
