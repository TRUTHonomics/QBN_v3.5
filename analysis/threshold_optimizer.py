# analysis/threshold_optimizer.py
"""
Base class voor Threshold Optimalisatie.

Bevat gedeelde logica voor data loading en preprocessing die door alle
analyse-methoden wordt gebruikt.

REFACTORED: Vervangt hardcoded signalen door dynamische loader uit qbn.signal_classification.
Dit zorgt ervoor dat alle 125+ signalen worden gebruikt in de composite score berekening.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from database.db import get_cursor
from inference.signal_aggregator import SignalAggregator
from inference.node_types import SemanticClass

logger = logging.getLogger(__name__)

# REASON: Mapping van semantic_class naar database tabel
SEMANTIC_CLASS_TABLE_MAP = {
    'LEADING': 'kfl.mtf_signals_lead',
    'COINCIDENT': 'kfl.mtf_signals_coin',
    'CONFIRMING': 'kfl.mtf_signals_conf'
}

# REASON: Tabel alias voor SQL queries
TABLE_ALIAS_MAP = {
    'kfl.mtf_signals_lead': 'l',
    'kfl.mtf_signals_coin': 'c',
    'kfl.mtf_signals_conf': 'f'
}


@dataclass
class ThresholdAnalysisResult:
    """Resultaat van een threshold analyse."""
    method: str  # 'mi_grid', 'cart', 'logreg'
    horizon: str  # '1h', '4h', '1d'
    target: str = "leading"  # 'leading', 'coincident', 'confirming'
    optimal_neutral_band: Optional[float] = None
    optimal_strong_threshold: Optional[float] = None
    # v3.5: Asymmetrische thresholds (bullish/bearish apart)
    optimal_bullish_neutral_band: Optional[float] = None
    optimal_bearish_neutral_band: Optional[float] = None
    optimal_bullish_strong_threshold: Optional[float] = None
    optimal_bearish_strong_threshold: Optional[float] = None
    optimal_alignment_high: Optional[float] = None
    optimal_alignment_low: Optional[float] = None
    score: float = 0.0
    score_name: str = ""  # 'MI', 'Gini', 'AUC'
    signal_weights: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Converteer naar dictionary voor JSON serialisatie."""
        return {
            'method': self.method,
            'horizon': self.horizon,
            'target': self.target,
            'optimal_neutral_band': self.optimal_neutral_band,
            'optimal_strong_threshold': self.optimal_strong_threshold,
            'optimal_bullish_neutral_band': self.optimal_bullish_neutral_band,
            'optimal_bearish_neutral_band': self.optimal_bearish_neutral_band,
            'optimal_bullish_strong_threshold': self.optimal_bullish_strong_threshold,
            'optimal_bearish_strong_threshold': self.optimal_bearish_strong_threshold,
            'optimal_alignment_high': self.optimal_alignment_high,
            'optimal_alignment_low': self.optimal_alignment_low,
            'score': self.score,
            'score_name': self.score_name,
            'signal_weights': self.signal_weights,
            'feature_importance': self.feature_importance,
            'metadata': self.metadata
        }


class ThresholdOptimizer(ABC):
    """
    Base class voor threshold optimalisatie methoden.
    
    Verzorgt:
    - Data loading uit database (outcomes, signals)
    - Preprocessing en normalisatie
    - Multi-timeframe signal anchor filtering
    - Train/test splitting
    """
    
    # Standaard horizons
    HORIZONS = ['1h', '4h', '1d']
    
    # Horizon naar minuten mapping
    HORIZON_MINUTES = {
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    # REASON: Elke horizon heeft zijn eigen time anchor en signal suffix.
    # Dit voorkomt autocorrelatie en koppelt outcomes aan de juiste signaal-timeframes.
    HORIZON_CONFIG = {
        '1h': {
            'time_col': 'time_60',
            'signal_suffix': '_60',
            'interval_filter': "EXTRACT(MINUTE FROM l.time_60) = 0",  # Elk uur
        },
        '4h': {
            'time_col': 'time_240',
            'signal_suffix': '_240',
            'interval_filter': "EXTRACT(HOUR FROM l.time_240) IN (0, 4, 8, 12, 16, 20) AND EXTRACT(MINUTE FROM l.time_240) = 0",
        },
        '1d': {
            'time_col': 'time_d',
            'signal_suffix': '_d',
            'interval_filter': "EXTRACT(HOUR FROM l.time_d) = 0 AND EXTRACT(MINUTE FROM l.time_d) = 0",  # Midnight UTC
        }
    }
    
    # REASON: Minimale samples per horizon om analyse zinvol te laten zijn.
    # 1h: ~180 dagen data
    # 4h: ~180 dagen data
    # 1d: ~150 dagen data
    MIN_SAMPLES_PER_HORIZON = {
        '1h': 4000,
        '4h': 1000,
        '1d': 150
    }
    
    def __init__(
        self,
        asset_id: int,
        lookback_days: int = 180,
        train_ratio: float = 0.8,
        min_samples: Optional[int] = None
    ):
        """
        Initialiseer de optimizer.
        
        Args:
            asset_id: Asset ID om te analyseren
            lookback_days: Aantal dagen historie
            train_ratio: Fractie voor training (rest is test)
            min_samples: Optioneel geforceerd minimum (overschrijft horizon-defaults)
        """
        self.asset_id = asset_id
        self.lookback_days = lookback_days
        self.train_ratio = train_ratio
        self.forced_min_samples = min_samples
        
        # Data cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._signal_classification: Optional[Dict] = None
        self._signal_weights: Optional[Dict] = None
        
        logger.info(f"ThresholdOptimizer initialized for asset {asset_id}, lookback={lookback_days}d")
    
    def load_data(self, horizon: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Laad en combineer signals met outcomes voor een specifieke horizon.
        
        Args:
            horizon: De te analyseren horizon ('1h', '4h', '1d')
            df: Optioneel reeds geladen DataFrame om database query over te slaan.
            
        Returns:
            DataFrame met kolommen:
            - timestamp
            - leading_score (normalized [-1, +1])
            - coincident_score (normalized [-1, +1])
            - confirming_score (normalized [-1, +1])
            - outcome_{horizon} (int: -3 tot +3)
            - outcome_direction (int: -1, 0, +1)
            - is_win (bool)
        """
        if df is not None:
            logger.debug(f"Using provided DataFrame for {horizon}")
            return df

        cache_key = f"{self.asset_id}_{horizon}"
        if cache_key in self._data_cache:
            logger.debug(f"Using cached data for {cache_key}")
            return self._data_cache[cache_key]
        
        logger.info(f"Loading data for asset {self.asset_id}, horizon {horizon}...")
        
        # Load signal classification en weights
        self._load_signal_metadata()
        
        # Fetch merged data
        df = self._fetch_merged_data(horizon)
        
        # Bepaal vereiste samples
        required_samples = self.forced_min_samples or self.MIN_SAMPLES_PER_HORIZON.get(horizon, 100)
        
        if len(df) < required_samples:
            raise ValueError(
                f"Insufficient data for {horizon}: {len(df)} samples, minimum required: {required_samples}"
            )
        
        # Cache result
        self._data_cache[cache_key] = df
        logger.info(f"Loaded {len(df)} samples for {horizon}")
        
        return df
    
    def _load_signal_metadata(self):
        """
        Laad signal classification en weights uit database.
        
        REFACTORED: Laadt nu alle signalen uit qbn.signal_classification en groepeert
        ze per semantic_class voor dynamische query generatie.
        """
        if self._signal_classification is not None:
            return
        
        # Signal classification met volledige info
        with get_cursor() as cur:
            cur.execute("""
                SELECT signal_name, semantic_class, polarity
                FROM qbn.signal_classification
                ORDER BY semantic_class, signal_name
            """)
            rows = cur.fetchall()
        
        self._signal_classification = {}
        self._signals_by_class: Dict[str, List[Dict]] = {
            'LEADING': [],
            'COINCIDENT': [],
            'CONFIRMING': []
        }
        
        for row in rows:
            signal_name, semantic_class, polarity = row
            
            # REASON: Correcte mapping van tekstuele polariteit naar numerieke waarde
            if isinstance(polarity, str):
                pol_lower = polarity.lower()
                if pol_lower == 'bullish':
                    pol_val = 1
                elif pol_lower == 'bearish':
                    pol_val = -1
                else:
                    # Probeer alsnog als digit te parsen voor backwards compatibility
                    try:
                        pol_val = int(polarity)
                    except (ValueError, TypeError):
                        pol_val = 0
            else:
                pol_val = int(polarity) if polarity is not None else 0
            
            signal_info = {
                'signal_name': signal_name,
                'semantic_class': semantic_class,
                'polarity': pol_val
            }
            self._signal_classification[signal_name] = signal_info
            
            if semantic_class in self._signals_by_class:
                self._signals_by_class[semantic_class].append(signal_info)
        
        # Signal weights (per horizon and layer)
        with get_cursor() as cur:
            cur.execute("""
                SELECT signal_name, horizon, weight, layer
                FROM qbn.signal_weights
                WHERE asset_id = %s
            """, (self.asset_id,))
            rows = cur.fetchall()
        
        # Weights partition: {layer: {horizon: {signal_name: weight}}}
        self._signal_weights_by_layer: Dict[str, Dict[str, Dict[str, float]]] = {
            'HYPOTHESIS': {},
            'CONFIDENCE': {}
        }
        
        for signal_name, horizon, weight, layer in rows:
            layer = layer or 'HYPOTHESIS' # Fallback
            if layer not in self._signal_weights_by_layer:
                self._signal_weights_by_layer[layer] = {}
            if horizon not in self._signal_weights_by_layer[layer]:
                self._signal_weights_by_layer[layer][horizon] = {}
            
            self._signal_weights_by_layer[layer][horizon][signal_name.upper()] = float(weight)
        
        # Log statistieken
        for cls, signals in self._signals_by_class.items():
            logger.info(f"Loaded {len(signals)} {cls} signals from qbn.signal_classification")
        
        logger.info(f"Total: {len(self._signal_classification)} signal classifications, "
                   f"{len(rows)} weighted signal-horizon entries loaded.")
    
    def _get_signal_columns_for_class(self, semantic_class: str, suffix: str) -> List[Tuple[str, str, int]]:
        """
        Genereer lijst van (db_column, alias, polarity) tuples voor een semantic class.
        
        Args:
            semantic_class: 'LEADING', 'COINCIDENT', of 'CONFIRMING'
            suffix: '_60', '_240', of '_d'
            
        Returns:
            List van tuples: (database_column_name, alias_name, polarity)
        """
        if not hasattr(self, '_signals_by_class'):
            self._load_signal_metadata()
        
        result = []
        for sig_info in self._signals_by_class.get(semantic_class, []):
            # Basis signal name (zonder suffix) -> database column met suffix
            base_name = sig_info['signal_name'].lower()
            db_col = f"{base_name}{suffix}"
            # Alias: gebruik base name voor leesbaarheid
            alias = base_name
            polarity = sig_info['polarity']
            result.append((db_col, alias, polarity))
        
        return result
    
    def _build_signal_select_clause(self, semantic_class: str, suffix: str, alias: str) -> Tuple[str, List[str]]:
        """
        Bouw SQL SELECT clause voor alle signalen van een semantic class.
        
        Returns:
            Tuple van (SQL select string, list of alias names)
        """
        columns = self._get_signal_columns_for_class(semantic_class, suffix)
        if not columns:
            return "", []
        
        select_parts = []
        aliases = []
        for db_col, col_alias, _ in columns:
            select_parts.append(f"COALESCE({alias}.{db_col}, 0) as {col_alias}")
            aliases.append(col_alias)
        
        return ", ".join(select_parts), aliases
    
    def _fetch_merged_data(self, horizon: str) -> pd.DataFrame:
        """
        Fetch en merge signals met outcomes voor een specifieke horizon.
        
        REFACTORED: Parallelle data retrieval voor optimale snelheid.
        """
        config = self.HORIZON_CONFIG[horizon]
        time_col = config['time_col']
        suffix = config['signal_suffix']
        interval_filter = config['interval_filter']
        
        # Store current suffix for composite score calculation
        self._current_suffix = suffix
        self._current_horizon = horizon
        
        # Ensure metadata is loaded
        self._load_signal_metadata()
        
        # Bouw dynamische SELECT clauses per semantic class
        leading_select, leading_aliases = self._build_signal_select_clause('LEADING', suffix, 'l')
        coincident_select, coincident_aliases = self._build_signal_select_clause('COINCIDENT', suffix, 'c')
        confirming_select, confirming_aliases = self._build_signal_select_clause('CONFIRMING', suffix, 'f')
        
        logger.info(f"Parallel fetch setup (60m Tactical): {len(leading_aliases)} LEADING, "
                   f"{len(coincident_aliases)} COINCIDENT, {len(confirming_aliases)} CONFIRMING signals")
        
        # Helper voor individuele queries
        def fetch_table(query_type, select_clause, table_name, alias, cols):
            # REASON: interval_filter gebruikt l.<time_col>; pas dit aan per alias.
            interval_filter_alias = interval_filter.replace('l.', f'{alias}.') if alias else interval_filter
            query = f"""
                SELECT DISTINCT ON ({alias}.{time_col})
                    {alias}.{time_col} as timestamp,
                    {select_clause}
                FROM {table_name} {alias}
                WHERE {alias}.asset_id = %(asset_id)s
                  AND {alias}.{time_col} >= NOW() - INTERVAL '%(lookback)s days'
                  AND {interval_filter_alias}
                ORDER BY {alias}.{time_col}
            """
            
            # Voor outcomes is de query iets anders
            if query_type == 'outcomes':
                query = f"""
                    SELECT 
                        o.time_1 as timestamp,
                        o.first_significant_barrier,
                        o.first_significant_time_min
                    FROM qbn.barrier_outcomes o
                    WHERE o.asset_id = %(asset_id)s
                      AND o.time_1 >= NOW() - INTERVAL '%(lookback)s days'
                      AND o.first_significant_barrier IS NOT NULL
                    ORDER BY o.time_1
                """
            
            with get_cursor() as cur:
                # logger.debug(f"Fetching {query_type}...")
                cur.execute(query, {'asset_id': self.asset_id, 'lookback': self.lookback_days})
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                
            return query_type, pd.DataFrame(rows, columns=columns)

        # Voer queries parallel uit
        import concurrent.futures
        dfs = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(fetch_table, 'leading', leading_select, 'kfl.mtf_signals_lead', 'l', leading_aliases),
                executor.submit(fetch_table, 'coincident', coincident_select, 'kfl.mtf_signals_coin', 'c', coincident_aliases),
                executor.submit(fetch_table, 'confirming', confirming_select, 'kfl.mtf_signals_conf', 'f', confirming_aliases),
                executor.submit(fetch_table, 'outcomes', None, None, None, None)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                q_type, df = future.result()
                if not df.empty:
                    # Zet timestamp als index voor snelle merge
                    df.set_index('timestamp', inplace=True)
                    dfs[q_type] = df
                    logger.info(f"✅ Fetched {q_type}: {len(df)} rows")
                else:
                    logger.warning(f"⚠️ Empty result for {q_type}")

        # Merge DataFrames (Inner Join om dataconsistentie te garanderen)
        if 'leading' not in dfs or dfs['leading'].empty:
             raise ValueError(f"No leading signals found for asset {self.asset_id}")
             
        df = dfs['leading']
        
        # Merge de andere tabellen
        for key in ['coincident', 'confirming', 'outcomes']:
            if key in dfs:
                df = df.join(dfs[key], how='inner') # Inner join behouden zoals in originele query
        
        # Reset index om timestamp weer als kolom te hebben
        df.reset_index(inplace=True)
        
        if df.empty:
            raise ValueError(f"No overlapping data found for asset {self.asset_id}, horizon {horizon}")
            
        # REASON: Map barrier hit naar numerieke outcome voor deze horizon
        # 1h = 60min, 4h = 240min, 1d = 1440min
        h_min = self.HORIZON_MINUTES[horizon]
        df['outcome'] = 0
        
        # Gebruik vectorisatie voor performance
        is_up = df['first_significant_barrier'].str.startswith('up', na=False)
        is_down = df['first_significant_barrier'].str.startswith('down', na=False)
        is_within = df['first_significant_time_min'] <= h_min
        
        df.loc[is_up & is_within, 'outcome'] = 1
        df.loc[is_down & is_within, 'outcome'] = -1
        
        logger.info(f"Loaded {len(df)} samples for {horizon} (Mapped to {h_min}min window)")
        
        # Store column mappings for composite calculation
        self._leading_aliases = leading_aliases
        self._coincident_aliases = coincident_aliases
        self._confirming_aliases = confirming_aliases

        # Bereken normalized composite scores met ALLE signalen
        df = self._compute_composite_scores(df)
        
        # Voeg afgeleide outcome kolommen toe
        df['outcome_direction'] = df['outcome'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        df['is_win'] = df['outcome'] > 0
        
        return df
    
    def _compute_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereken normalized composite scores per semantic class.
        
        REFACTORED: Gebruikt nu ALLE signalen uit qbn.signal_classification
        met hun respectievelijke weights en polarities.
        
        Score = sum(signal_value * weight * polarity) / sum(|weight|)
        Result is normalized to [-1, +1] range.
        
        Polariteit:
        - +1: Bullish signaal (bijv. RSI_OVERSOLD -> koopsignaal)
        - -1: Bearish signaal (bijv. RSI_OVERBOUGHT -> verkoopsignaal)
        -  0: Neutral/context signaal (bijv. BB_SQUEEZE -> volatiliteit)
        """
        # Bereken composite score voor elke semantic class
        df['leading_score'] = self._compute_class_score(df, 'LEADING')
        df['coincident_score'] = self._compute_class_score(df, 'COINCIDENT')
        df['confirming_score'] = self._compute_class_score(df, 'CONFIRMING')
        
        return df
    
    def _compute_class_score(self, df: pd.DataFrame, semantic_class: str) -> pd.Series:
        """
        Bereken gewogen composite score voor een semantic class.
        
        Args:
            df: DataFrame met signaal kolommen
            semantic_class: 'LEADING', 'COINCIDENT', of 'CONFIRMING'
            
        Returns:
            Series met normalized scores [-1, +1]
        """
        if not hasattr(self, '_signals_by_class'):
            self._load_signal_metadata()
        
        signals = self._signals_by_class.get(semantic_class, [])
        if not signals:
            logger.warning(f"No signals found for class {semantic_class}")
            return pd.Series(0.0, index=df.index)
        
        # REASON: Bepaal de juiste layer en horizon voor weights
        # LEADING signals -> HYPOTHESIS layer (altijd 1h horizon als fundament)
        # COINCIDENT/CONFIRMING -> CONFIDENCE layer (horizon-specifiek)
        layer = 'HYPOTHESIS' if semantic_class == 'LEADING' else 'CONFIDENCE'
        
        # Voor HYPOTHESIS layer gebruiken we altijd de 1h weights (onze trigger definitie)
        target_horizon = '1h' if layer == 'HYPOTHESIS' else self._current_horizon
        
        layer_weights = self._signal_weights_by_layer.get(layer, {})
        horizon_weights = layer_weights.get(target_horizon, {})
        
        # Accumuleer gewogen scores
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        signals_used = 0
        
        for sig_info in signals:
            base_name = sig_info['signal_name'].lower()
            polarity = sig_info['polarity']
            
            # Check if column exists in DataFrame
            if base_name not in df.columns:
                continue
            
            # Get weight: prefer explicit layer/horizon weight, fallback to 1.0
            signal_name_upper = sig_info['signal_name'].upper()
            weight = horizon_weights.get(signal_name_upper, 1.0)
            
            # Bereken bijdrage: signal_value * weight * polarity
            contribution = df[base_name] * weight * polarity
            weighted_sum += contribution
            total_weight += abs(weight)
            signals_used += 1
        
        if total_weight == 0:
            logger.warning(f"No weights found for {semantic_class} ({layer}), using equal weights")
            total_weight = signals_used or 1
        
        # Normalize naar [-1, +1]
        normalized_score = weighted_sum / total_weight
        normalized_score = normalized_score.clip(-1, 1)
        
        logger.debug(f"{semantic_class} score ({layer}/{target_horizon}): {signals_used} signals, "
                    f"mean={normalized_score.mean():.4f}, std={normalized_score.std():.4f}")
        
        return normalized_score
    
    def _get_weight(self, signal_name: str, horizon: Optional[str] = None) -> float:
        """
        Haal weight op voor een signaal.
        
        Args:
            signal_name: Naam van het signaal (case-insensitive)
            horizon: Optioneel, specifieke horizon voor horizon-specifieke weights
            
        Returns:
            Weight waarde, default 1.0
        """
        if self._signal_weights is None:
            return 1.0
        
        # Normalize naam naar uppercase voor lookup
        signal_upper = signal_name.upper()
        
        # Probeer horizon-specifieke weight eerst
        if horizon and hasattr(self, '_signal_weights_by_horizon'):
            horizon_weights = self._signal_weights_by_horizon.get(horizon, {})
            if signal_upper in horizon_weights:
                return horizon_weights[signal_upper]
        
        # Fallback naar gemiddelde weight
        return self._signal_weights.get(signal_upper, 1.0)
    
    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporele train/test split (geen shuffle om lookahead bias te voorkomen).
        
        Returns:
            Tuple van (train_df, test_df)
        """
        split_idx = int(len(df) * self.train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.debug(f"Train/test split: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df
    
    def discretize_composite(
        self,
        score: float,
        neutral_band: float,
        strong_threshold: float
    ) -> str:
        """
        Discretiseer een composite score naar een categorical state.
        
        Args:
            score: Normalized score [-1, +1]
            neutral_band: Threshold voor neutral zone (bijv. 0.15)
            strong_threshold: Threshold voor strong signals (bijv. 0.50)
            
        Returns:
            State: 'strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish'
        """
        if score >= strong_threshold:
            return 'strong_bullish'
        elif score >= neutral_band:
            return 'bullish'
        elif score <= -strong_threshold:
            return 'strong_bearish'
        elif score <= -neutral_band:
            return 'bearish'
        else:
            return 'neutral'

    def discretize_composite_asymmetric(
        self,
        score: float,
        bullish_neutral_band: float,
        bullish_strong_threshold: float,
        bearish_neutral_band: float,
        bearish_strong_threshold: float
    ) -> str:
        """
        Discretiseer een composite score met asymmetrische thresholds.

        REASON: Sommige assets hebben scheve score-verdelingen; symmetrische thresholds
        kunnen dan leiden tot 'bearish' states die praktisch nooit geactiveerd worden.
        """
        if score >= bullish_strong_threshold:
            return 'strong_bullish'
        if score >= bullish_neutral_band:
            return 'bullish'
        if score <= -bearish_strong_threshold:
            return 'strong_bearish'
        if score <= -bearish_neutral_band:
            return 'bearish'
        return 'neutral'
    
    @abstractmethod
    def analyze(self, horizon: str) -> ThresholdAnalysisResult:
        """
        Voer de analyse uit voor een specifieke horizon.
        
        Args:
            horizon: '1h', '4h', of '1d'
            
        Returns:
            ThresholdAnalysisResult met optimale parameters
        """
        pass
    
    def analyze_all_horizons(self) -> Dict[str, ThresholdAnalysisResult]:
        """
        Voer analyse uit voor alle horizons.
        
        Returns:
            Dict mapping horizon naar ThresholdAnalysisResult
        """
        results = {}
        for horizon in self.HORIZONS:
            try:
                logger.info(f"Analyzing horizon: {horizon}")
                results[horizon] = self.analyze(horizon)
            except Exception as e:
                logger.error(f"Error analyzing {horizon}: {e}")
                raise
        return results
    
    def get_signal_columns(self) -> Dict[str, List[str]]:
        """
        Retourneer mapping van semantic class naar signal kolommen.
        
        REFACTORED: Haalt nu dynamisch alle signalen uit qbn.signal_classification
        in plaats van een hardcoded lijst.
        """
        if not hasattr(self, '_signals_by_class'):
            self._load_signal_metadata()
        
        return {
            'leading': [s['signal_name'].lower() for s in self._signals_by_class.get('LEADING', [])],
            'coincident': [s['signal_name'].lower() for s in self._signals_by_class.get('COINCIDENT', [])],
            'confirming': [s['signal_name'].lower() for s in self._signals_by_class.get('CONFIRMING', [])]
        }
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Retourneer statistieken over geladen signalen.
        
        Returns:
            Dict met counts per semantic class en weight coverage
        """
        if not hasattr(self, '_signals_by_class'):
            self._load_signal_metadata()
        
        # Bereken het aantal unieke signalen waarvoor we ergens een weight hebben
        weighted_signals = set()
        for layer_horizons in self._signal_weights_by_layer.values():
            for horizon_weights in layer_horizons.values():
                weighted_signals.update(horizon_weights.keys())

        stats = {
            'total_signals': len(self._signal_classification),
            'signals_per_class': {
                cls: len(signals) for cls, signals in self._signals_by_class.items()
            },
            'signals_with_weights': len(weighted_signals),
            'weight_coverage': len(weighted_signals) / max(len(self._signal_classification), 1) * 100
        }
        return stats

