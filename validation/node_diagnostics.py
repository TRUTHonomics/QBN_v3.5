"""
Node-Level Diagnostic Validator voor QBN v3.

Valideert elke BN node individueel tegen historische outcomes om te bepalen
welke node(s) de slechte walk-forward resultaten veroorzaken.

Diagnostics per node:
1. State Distribution - Detecteert stuck/dead states
2. Outcome Correlation (MI) - Meet predictive power
3. Directional Alignment - Voor directional nodes
4. Calibration Check - Voor prediction nodes
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import Counter
import numpy as np
import pandas as pd

from database.db import get_cursor
from inference.inference_loader import InferenceLoader
from inference.trade_aligned_inference import SignalEvidence
from inference.node_types import SemanticClass

logger = logging.getLogger(__name__)

# Thresholds voor diagnostic status
DIAGNOSTIC_THRESHOLDS = {
    'stuck_state_pct': 0.90,      # >90% in één state = stuck
    'dead_state_threshold': 0.01, # <1% = dead state
    'mi_good': 0.10,              # MI > 0.10 = good predictive power
    'mi_warn': 0.03,              # MI 0.03-0.10 = marginal
    'directional_good': 0.60,    # >60% alignment = good
    'directional_warn': 0.55,    # 55-60% = marginal
}


@dataclass
class NodeDiagnosticResult:
    """Resultaat van diagnostic voor een enkele node."""
    node_name: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    
    # State distribution
    state_distribution: Dict[str, float] = field(default_factory=dict)
    stuck_state: Optional[str] = None
    dead_states: List[str] = field(default_factory=list)
    
    # Outcome correlation (Mutual Information)
    mi_1h: Optional[float] = None
    mi_4h: Optional[float] = None
    mi_1d: Optional[float] = None
    
    # Directional alignment (voor hypothesis/prediction nodes)
    directional_alignment: Optional[Dict[str, float]] = None
    
    # Issues gevonden
    issues: List[str] = field(default_factory=list)
    
    # Sample count
    sample_count: int = 0


class NodeDiagnosticValidator:
    """
    Valideert elke BN node individueel tegen historische outcomes.
    
    Usage:
        validator = NodeDiagnosticValidator(asset_id=1)
        results = validator.run_full_diagnostic(days=30)
        # results is Dict[str, NodeDiagnosticResult]
    """
    
    def __init__(self, asset_id: int, run_id: Optional[str] = None):
        self.asset_id = asset_id
        self.run_id = run_id
        self.loader = InferenceLoader()
        # REASON: Validation kan een specifieke training-run willen evalueren (CPTs per run_id).
        self.engine = self.loader.load_inference_engine(asset_id, horizon='1h', run_id=run_id)
        self.classification = self.engine.signal_classification
        
        # Cache for fetched data
        self._data_cache: Optional[pd.DataFrame] = None
        self._inference_cache: Optional[List[Dict]] = None
    
    def _get_earliest_data_timestamp(self) -> Optional[datetime]:
        """Haal het vroegste timestamp op voor dit asset uit de database."""
        query = "SELECT MIN(time) FROM kfl.indicators WHERE asset_id = %s"
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
        return None

    def run_full_diagnostic(self, days: int = 30) -> Dict[str, NodeDiagnosticResult]:
        """
        Voer volledige diagnostic uit op alle nodes.
        
        Args:
            days: Aantal dagen historische data om te analyseren
            
        Returns:
            Dict met NodeDiagnosticResult per node
        """
        # REASON: Stel vast wanneer de data voor dit asset echt begint
        # EXPL: Voorkomt queries op periodes zonder data.
        # REASON: Gebruik timezone-aware datetimes om te matchen met DB timestamps
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        earliest_db_time = self._get_earliest_data_timestamp()
        
        if earliest_db_time:
            # Zorg dat earliest_db_time ook aware is (indien niet zo uit DB)
            if earliest_db_time.tzinfo is None:
                earliest_db_time = earliest_db_time.replace(tzinfo=timezone.utc)
                
            if start_date < earliest_db_time:
                logger.info(f"📍 Start datum aangepast naar vroegste beschikbare data: {earliest_db_time}")
                start_date = earliest_db_time
                # Bereken 'days' opnieuw voor de logging
                days = (end_date - start_date).days
        else:
            logger.error(f"❌ Geen data gevonden voor asset {self.asset_id} in kfl.indicators")
            return {}

        logger.info(f"🔬 Start Node-Level Diagnostic voor Asset {self.asset_id} ({days} dagen)")
        
        # Fetch en cache data
        self._data_cache = self._fetch_test_data_by_range(start_date, end_date)
        if self._data_cache.empty:
            logger.error("Geen test data gevonden!")
            return {}
        
        logger.info(f"📊 {len(self._data_cache)} datapunten geladen")
        
        # ... rest of method ...
    
    def _run_inference_on_data(self) -> List[Dict]:
        """Voer inference uit op alle data en return resultaten."""
        results = []
        
        for idx, row in self._data_cache.iterrows():
            evidence = self._convert_to_evidence(row)
            inf_result = self.engine.infer(evidence)
            
            # Sla alle intermediate states op
            results.append({
                'time': row['time_1'],
                'regime': inf_result.regime,
                'leading_composite': inf_result.leading_composite,
                'coincident_composite': inf_result.coincident_composite,
                'confirming_composite': inf_result.confirming_composite,
                'trade_hypothesis': inf_result.trade_hypothesis,
                'entry_confidence': inf_result.entry_confidence,
                'position_confidence': inf_result.position_confidence,
                'prediction_1h': inf_result.predictions.get('1h').most_probable_state if inf_result.predictions.get('1h') else None,
                'prediction_4h': inf_result.predictions.get('4h').most_probable_state if inf_result.predictions.get('4h') else None,
                'prediction_1d': inf_result.predictions.get('1d').most_probable_state if inf_result.predictions.get('1d') else None,
                'dist_1h': inf_result.distributions.get('1h'),
                'dist_4h': inf_result.distributions.get('4h'),
                'dist_1d': inf_result.distributions.get('1d'),
                # Actual outcomes
                'outcome_1h': row.get('outcome_1h'),
                'outcome_4h': row.get('outcome_4h'),
                'outcome_1d': row.get('outcome_1d'),
            })
        
        return results
    
    def _convert_to_evidence(self, row: pd.Series) -> SignalEvidence:
        """Convert database row naar SignalEvidence."""
        evidence = SignalEvidence(
            asset_id=self.asset_id,
            timestamp=row['time_1']
        )

        # HTF regime signals (adx_signal_d en adx_signal_240 uit mtf_signals_conf)
        if 'adx_signal_d' in row and pd.notna(row['adx_signal_d']):
            evidence.adx_d = int(row['adx_signal_d'])
        if 'adx_signal_240' in row and pd.notna(row['adx_signal_240']):
            evidence.adx_240 = int(row['adx_signal_240'])

        for full_sig_name, info in self.classification.items():
            sem_class = info['semantic_class']
            
            if full_sig_name in row and pd.notna(row[full_sig_name]):
                val = int(row[full_sig_name])
                
                if sem_class == SemanticClass.LEADING.value:
                    evidence.leading_signals[full_sig_name] = val
                elif sem_class == SemanticClass.COINCIDENT.value:
                    evidence.coincident_signals[full_sig_name] = val
                elif sem_class == SemanticClass.CONFIRMING.value:
                    evidence.confirming_signals[full_sig_name] = val
        
        return evidence
    
    # =========================================================================
    # DIAGNOSTIC METHODS - To be implemented in subsequent todos
    # =========================================================================
    
    def _diagnose_regime(self) -> NodeDiagnosticResult:
        """
        Diagnostic voor HTF_Regime node.

        HTF_Regime is een CONTEXT/CLASSIFICATIE node, geen predictie node.
        Het doel is marktregime classificeren, niet direct outcomes voorspellen.

        Checkt:
        - State distribution (verwacht: redelijke spreiding over 5 states)
        - Stuck state detectie (>90% in één state = probleem)
        - State diversity (minstens 3 actieve states)
        - MI wordt getoond als info, maar is GEEN fail criterium
        """
        node_name = 'HTF_Regime'
        issues = []

        # Haal regime states uit inference cache
        regime_states = [r['regime'] for r in self._inference_cache]

        # State distribution analyse
        distribution, stuck_state, dead_states = self._calculate_state_distribution(regime_states)

        if stuck_state:
            issues.append(f"STUCK: {stuck_state} = {distribution[stuck_state]:.0%}")

        # Check state diversity (minstens 3 states met >5% representatie)
        active_states = len([s for s, pct in distribution.items() if pct > 0.05])
        if active_states < 3:
            issues.append(f"LOW DIVERSITY: slechts {active_states} actieve states (verwacht ≥3)")

        # MI met outcomes (informatief, geen fail criterium voor regime node)
        outcomes_1h = [r['outcome_1h'] for r in self._inference_cache]
        outcomes_4h = [r['outcome_4h'] for r in self._inference_cache]
        outcomes_1d = [r['outcome_1d'] for r in self._inference_cache]

        mi_1h = self._calculate_mutual_information(regime_states, outcomes_1h)
        mi_4h = self._calculate_mutual_information(regime_states, outcomes_4h)
        mi_1d = self._calculate_mutual_information(regime_states, outcomes_1d)

        # Bepaal overall status - alleen gebaseerd op distributie, niet MI
        # HTF_Regime is een context node, MI is niet relevant voor PASS/FAIL
        if stuck_state:
            status = 'FAIL'
        elif active_states < 3:
            status = 'WARN'
        else:
            status = 'PASS'

        return NodeDiagnosticResult(
            node_name=node_name,
            status=status,
            state_distribution=distribution,
            stuck_state=stuck_state,
            dead_states=dead_states,
            mi_1h=mi_1h,
            mi_4h=mi_4h,
            mi_1d=mi_1d,
            issues=issues,
            sample_count=len(regime_states)
        )
    
    def _diagnose_composite(self, composite_type: str) -> NodeDiagnosticResult:
        """
        Diagnostic voor Composite nodes (Leading/Coincident/Confirming).
        
        Checkt:
        - State distribution (verwacht: spreiding, niet >90% neutral)
        - MI met outcomes
        - Directional alignment voor bullish/bearish states
        """
        node_name = f'{composite_type.title()}_Composite'
        issues = []
        
        # Key in inference cache
        cache_key = f'{composite_type}_composite'
        
        # Haal composite states uit cache
        composite_states = [r[cache_key] for r in self._inference_cache]
        
        # State distribution
        distribution, stuck_state, dead_states = self._calculate_state_distribution(composite_states)
        
        # REASON: Voor composites is "neutral" dominant vaak een probleem
        # omdat het betekent dat signalen niet genoeg activeren
        neutral_pct = distribution.get('neutral', 0) + distribution.get('Neutral', 0)
        if neutral_pct > 0.80:
            issues.append(f"HIGH NEUTRAL: {neutral_pct:.0%} (signalen activeren niet)")
        
        if stuck_state:
            issues.append(f"STUCK: {stuck_state} = {distribution[stuck_state]:.0%}")
        
        # MI met outcomes
        outcomes_1h = [r['outcome_1h'] for r in self._inference_cache]
        outcomes_4h = [r['outcome_4h'] for r in self._inference_cache]
        outcomes_1d = [r['outcome_1d'] for r in self._inference_cache]
        
        mi_1h = self._calculate_mutual_information(composite_states, outcomes_1h)
        mi_4h = self._calculate_mutual_information(composite_states, outcomes_4h)
        mi_1d = self._calculate_mutual_information(composite_states, outcomes_1d)
        
        best_mi = max(mi_1h, mi_4h, mi_1d)
        if best_mi < DIAGNOSTIC_THRESHOLDS['mi_warn']:
            issues.append(f"LOW MI: max={best_mi:.3f}")
        
        # Directional alignment check
        directional = self._calculate_directional_alignment(composite_states, outcomes_1h)
        
        if directional:
            bullish_acc = directional.get('bullish_accuracy', 0)
            bearish_acc = directional.get('bearish_accuracy', 0)
            
            if bullish_acc < DIAGNOSTIC_THRESHOLDS['directional_warn']:
                issues.append(f"POOR BULLISH ALIGNMENT: {bullish_acc:.0%}")
            if bearish_acc < DIAGNOSTIC_THRESHOLDS['directional_warn']:
                issues.append(f"POOR BEARISH ALIGNMENT: {bearish_acc:.0%}")
        
        # Status
        if stuck_state or neutral_pct > 0.90 or best_mi < DIAGNOSTIC_THRESHOLDS['mi_warn']:
            status = 'FAIL'
        elif neutral_pct > 0.80 or best_mi < DIAGNOSTIC_THRESHOLDS['mi_good']:
            status = 'WARN'
        else:
            status = 'PASS'
        
        return NodeDiagnosticResult(
            node_name=node_name,
            status=status,
            state_distribution=distribution,
            stuck_state=stuck_state,
            dead_states=dead_states,
            mi_1h=mi_1h,
            mi_4h=mi_4h,
            mi_1d=mi_1d,
            directional_alignment=directional,
            issues=issues,
            sample_count=len(composite_states)
        )
    
    def _diagnose_trade_hypothesis(self) -> NodeDiagnosticResult:
        """
        Diagnostic voor Trade_Hypothesis node.
        
        Dit is een KRITIEKE node voor entry decisions.
        
        Checkt:
        - State distribution (verwacht: NIET >90% no_setup)
        - Directional alignment: strong_long → bullish outcomes, strong_short → bearish
        - MI met outcomes
        """
        node_name = 'Trade_Hypothesis'
        issues = []
        
        hypothesis_states = [r['trade_hypothesis'] for r in self._inference_cache]
        
        # State distribution
        distribution, stuck_state, dead_states = self._calculate_state_distribution(hypothesis_states)
        
        # CRITICAL: Check voor te veel no_setup
        no_setup_pct = distribution.get('no_setup', 0)
        if no_setup_pct > 0.90:
            issues.append(f"CRITICAL STUCK: no_setup = {no_setup_pct:.0%} (geen trade signals!)")
        elif no_setup_pct > 0.80:
            issues.append(f"HIGH no_setup: {no_setup_pct:.0%}")
        
        # Check voor actieve signal states
        long_states = sum(distribution.get(s, 0) for s in ['weak_long', 'strong_long'])
        short_states = sum(distribution.get(s, 0) for s in ['weak_short', 'strong_short'])
        
        if long_states < 0.02:
            issues.append(f"VERY FEW LONG SIGNALS: {long_states:.1%}")
        if short_states < 0.02:
            issues.append(f"VERY FEW SHORT SIGNALS: {short_states:.1%}")
        
        # MI met outcomes
        outcomes_1h = [r['outcome_1h'] for r in self._inference_cache]
        outcomes_4h = [r['outcome_4h'] for r in self._inference_cache]
        outcomes_1d = [r['outcome_1d'] for r in self._inference_cache]
        
        mi_1h = self._calculate_mutual_information(hypothesis_states, outcomes_1h)
        mi_4h = self._calculate_mutual_information(hypothesis_states, outcomes_4h)
        mi_1d = self._calculate_mutual_information(hypothesis_states, outcomes_1d)
        
        best_mi = max(mi_1h, mi_4h, mi_1d)
        if best_mi < DIAGNOSTIC_THRESHOLDS['mi_warn']:
            issues.append(f"LOW MI: max={best_mi:.3f} (hypothesis geeft geen info)")
        
        # Directional alignment per hypothesis state
        directional = {}
        
        for state in ['strong_long', 'weak_long', 'strong_short', 'weak_short']:
            # Filter for this state
            state_indices = [i for i, h in enumerate(hypothesis_states) if h == state]
            if len(state_indices) < 5:
                continue
            
            state_outcomes = [outcomes_1h[i] for i in state_indices if outcomes_1h[i] is not None and not pd.isna(outcomes_1h[i])]
            if not state_outcomes:
                continue
            
            # Expected direction
            expected_dir = 1 if 'long' in state else -1
            
            # Actual direction matches
            correct = sum(1 for o in state_outcomes if (o > 0 and expected_dir == 1) or (o < 0 and expected_dir == -1))
            accuracy = correct / len(state_outcomes) if state_outcomes else 0
            
            directional[state] = accuracy
            
            # Check alignment
            if accuracy < DIAGNOSTIC_THRESHOLDS['directional_warn']:
                issues.append(f"{state} → wrong direction {accuracy:.0%} (verwacht >{DIAGNOSTIC_THRESHOLDS['directional_good']:.0%})")
        
        # Status
        if no_setup_pct > 0.90 or best_mi < 0.01:
            status = 'FAIL'
        elif no_setup_pct > 0.80 or best_mi < DIAGNOSTIC_THRESHOLDS['mi_warn']:
            status = 'WARN'
        else:
            status = 'PASS'
        
        return NodeDiagnosticResult(
            node_name=node_name,
            status=status,
            state_distribution=distribution,
            stuck_state=stuck_state if no_setup_pct > 0.90 else None,
            dead_states=dead_states,
            mi_1h=mi_1h,
            mi_4h=mi_4h,
            mi_1d=mi_1d,
            directional_alignment=directional,
            issues=issues,
            sample_count=len(hypothesis_states)
        )
    
    def _diagnose_entry_confidence(self) -> NodeDiagnosticResult:
        """
        Diagnostic voor Entry_Confidence node.
        
        Checkt:
        - State distribution (low/medium/high)
        - Correlatie tussen confidence en outcome magnitude
        - MI met outcomes
        """
        node_name = 'Entry_Confidence'
        issues = []
        
        confidence_states = [r['entry_confidence'] for r in self._inference_cache]
        
        # State distribution
        distribution, stuck_state, dead_states = self._calculate_state_distribution(confidence_states)
        
        if stuck_state:
            issues.append(f"STUCK: {stuck_state} = {distribution[stuck_state]:.0%}")
        
        # Check voor variatie
        unique_states = len([s for s, pct in distribution.items() if pct > 0.05])
        if unique_states < 2:
            issues.append(f"LOW VARIATIE: slechts {unique_states} actieve states")
        
        # MI met outcomes
        outcomes_1h = [r['outcome_1h'] for r in self._inference_cache]
        mi_1h = self._calculate_mutual_information(confidence_states, outcomes_1h)
        mi_4h = self._calculate_mutual_information(confidence_states, [r['outcome_4h'] for r in self._inference_cache])
        mi_1d = self._calculate_mutual_information(confidence_states, [r['outcome_1d'] for r in self._inference_cache])
        
        best_mi = max(mi_1h, mi_4h, mi_1d)
        if best_mi < DIAGNOSTIC_THRESHOLDS['mi_warn']:
            issues.append(f"LOW MI: {best_mi:.3f}")
        
        # Magnitude correlation: does high confidence = bigger moves?
        magnitude_corr = self._check_confidence_magnitude_correlation(confidence_states, outcomes_1h)
        if magnitude_corr is not None and magnitude_corr < 0.1:
            issues.append(f"NO MAGNITUDE CORRELATION: r={magnitude_corr:.2f}")
        
        # Status
        if stuck_state:
            status = 'FAIL'
        elif unique_states < 2 or best_mi < DIAGNOSTIC_THRESHOLDS['mi_warn']:
            status = 'WARN'
        else:
            status = 'PASS'
        
        return NodeDiagnosticResult(
            node_name=node_name,
            status=status,
            state_distribution=distribution,
            stuck_state=stuck_state,
            dead_states=dead_states,
            mi_1h=mi_1h,
            mi_4h=mi_4h,
            mi_1d=mi_1d,
            issues=issues,
            sample_count=len(confidence_states)
        )
    
    def _diagnose_position_confidence(self) -> NodeDiagnosticResult:
        """
        Diagnostic voor Position_Confidence node.
        
        Checkt:
        - State distribution (low/medium/high)
        - MI met outcomes
        """
        node_name = 'Position_Confidence'
        issues = []
        
        confidence_states = [r['position_confidence'] for r in self._inference_cache]
        
        # State distribution
        distribution, stuck_state, dead_states = self._calculate_state_distribution(confidence_states)
        
        if stuck_state:
            issues.append(f"STUCK: {stuck_state} = {distribution[stuck_state]:.0%}")
        
        # MI met outcomes
        mi_1h = self._calculate_mutual_information(confidence_states, [r['outcome_1h'] for r in self._inference_cache])
        mi_4h = self._calculate_mutual_information(confidence_states, [r['outcome_4h'] for r in self._inference_cache])
        mi_1d = self._calculate_mutual_information(confidence_states, [r['outcome_1d'] for r in self._inference_cache])
        
        best_mi = max(mi_1h, mi_4h, mi_1d)
        if best_mi < DIAGNOSTIC_THRESHOLDS['mi_warn']:
            issues.append(f"LOW MI: {best_mi:.3f}")
        
        # Status
        if stuck_state:
            status = 'FAIL'
        elif best_mi < DIAGNOSTIC_THRESHOLDS['mi_warn']:
            status = 'WARN'
        else:
            status = 'PASS'
        
        return NodeDiagnosticResult(
            node_name=node_name,
            status=status,
            state_distribution=distribution,
            stuck_state=stuck_state,
            dead_states=dead_states,
            mi_1h=mi_1h,
            mi_4h=mi_4h,
            mi_1d=mi_1d,
            issues=issues,
            sample_count=len(confidence_states)
        )
    
    def _check_confidence_magnitude_correlation(self, confidence_states: List[str], outcomes: List) -> Optional[float]:
        """
        Check of hogere confidence correleert met grotere moves.
        
        Returns Spearman correlation coefficient.
        """
        # Map confidence to numeric
        conf_map = {'low': 1, 'medium': 2, 'high': 3}
        
        valid_pairs = []
        for conf, out in zip(confidence_states, outcomes):
            if conf and out is not None and not pd.isna(out):
                conf_num = conf_map.get(conf.lower(), 2)
                valid_pairs.append((conf_num, abs(float(out))))
        
        if len(valid_pairs) < 20:
            return None
        
        # Simple correlation
        conf_vals = [p[0] for p in valid_pairs]
        mag_vals = [p[1] for p in valid_pairs]
        
        # Spearman rank correlation (with fallback)
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(conf_vals, mag_vals)
            return corr
        except ImportError:
            # Fallback: simple Pearson correlation via numpy
            conf_arr = np.array(conf_vals)
            mag_arr = np.array(mag_vals)
            if np.std(conf_arr) == 0 or np.std(mag_arr) == 0:
                return 0.0
            return float(np.corrcoef(conf_arr, mag_arr)[0, 1])
        except Exception:
            return None
    
    def _diagnose_prediction(self, horizon: str) -> NodeDiagnosticResult:
        """
        Diagnostic voor Prediction nodes (1h/4h/1d).
        
        Checkt:
        - State distribution (7 states)
        - Accuracy (exact state match)
        - Directional accuracy
        - Calibration (predicted prob vs observed freq)
        - Brier score decomposition
        """
        node_name = f'Prediction_{horizon}'
        issues = []
        
        prediction_key = f'prediction_{horizon}'
        dist_key = f'dist_{horizon}'
        outcome_key = f'outcome_{horizon}'
        
        predictions = [r[prediction_key] for r in self._inference_cache]
        distributions = [r[dist_key] for r in self._inference_cache]
        outcomes = [r[outcome_key] for r in self._inference_cache]
        
        # State distribution
        distribution, stuck_state, dead_states = self._calculate_state_distribution(predictions)
        
        if stuck_state:
            issues.append(f"STUCK: {stuck_state} = {distribution[stuck_state]:.0%}")
        
        # Filter valid pairs for accuracy
        valid_indices = [i for i, o in enumerate(outcomes) if o is not None and not pd.isna(o)]
        
        if len(valid_indices) < 10:
            issues.append(f"INSUFFICIENT DATA: only {len(valid_indices)} valid outcomes")
            return NodeDiagnosticResult(
                node_name=node_name,
                status='WARN',
                state_distribution=distribution,
                issues=issues,
                sample_count=len(predictions)
            )
        
        valid_preds = [predictions[i] for i in valid_indices]
        valid_outcomes = [int(outcomes[i]) for i in valid_indices]
        valid_dists = [distributions[i] for i in valid_indices]
        
        # Map outcomes to state names for comparison
        outcome_state_map = {
            -3: 'Strong_Bearish', -2: 'Bearish', -1: 'Slight_Bearish',
            0: 'Neutral',
            1: 'Slight_Bullish', 2: 'Bullish', 3: 'Strong_Bullish'
        }
        valid_outcome_states = [outcome_state_map.get(o, 'Neutral') for o in valid_outcomes]
        
        # Exact accuracy
        exact_correct = sum(1 for p, o in zip(valid_preds, valid_outcome_states) if p == o)
        exact_accuracy = exact_correct / len(valid_preds)
        
        if exact_accuracy < 0.10:
            issues.append(f"VERY LOW ACCURACY: {exact_accuracy:.1%}")
        elif exact_accuracy < 0.20:
            issues.append(f"LOW ACCURACY: {exact_accuracy:.1%}")
        
        # Directional accuracy
        directional = self._calculate_directional_alignment(valid_preds, valid_outcomes)
        dir_accuracy = directional.get('overall', 0)
        
        if dir_accuracy < 0.50:
            issues.append(f"WORSE THAN RANDOM DIR: {dir_accuracy:.0%}")
        elif dir_accuracy < DIAGNOSTIC_THRESHOLDS['directional_warn']:
            issues.append(f"POOR DIR ACCURACY: {dir_accuracy:.0%}")
        
        # Calibration check
        calibration = self._check_calibration(valid_dists, valid_outcome_states)
        if calibration['reliability'] > 0.10:
            issues.append(f"POOR CALIBRATION: reliability={calibration['reliability']:.3f}")
        
        # Brier score
        brier = self._calculate_brier_score(valid_dists, valid_outcome_states)
        if brier > 0.30:
            issues.append(f"HIGH BRIER: {brier:.3f}")
        
        # MI (using predicted state, not distribution)
        mi = self._calculate_mutual_information(valid_preds, valid_outcomes)
        
        # Status
        if exact_accuracy < 0.05 or dir_accuracy < 0.40:
            status = 'FAIL'
        elif exact_accuracy < 0.15 or dir_accuracy < DIAGNOSTIC_THRESHOLDS['directional_warn']:
            status = 'WARN'
        else:
            status = 'PASS'
        
        result = NodeDiagnosticResult(
            node_name=node_name,
            status=status,
            state_distribution=distribution,
            stuck_state=stuck_state,
            dead_states=dead_states,
            mi_1h=mi if horizon == '1h' else None,
            mi_4h=mi if horizon == '4h' else None,
            mi_1d=mi if horizon == '1d' else None,
            directional_alignment=directional,
            issues=issues,
            sample_count=len(valid_preds)
        )
        
        # Extra metrics in details
        result.issues.insert(0, f"Accuracy: {exact_accuracy:.1%}, Dir: {dir_accuracy:.0%}, Brier: {brier:.3f}")
        
        return result
    
    def _check_calibration(self, distributions: List[Dict], actual_states: List[str]) -> Dict[str, float]:
        """
        Check calibration: are predicted probabilities well-calibrated?
        
        Groups predictions by confidence bucket and compares to observed frequency.
        """
        # Bin predictions by confidence
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        bin_counts = {b: {'correct': 0, 'total': 0, 'prob_sum': 0} for b in bins}
        
        for dist, actual in zip(distributions, actual_states):
            if not dist:
                continue
            
            predicted_state = max(dist, key=dist.get)
            prob = dist[predicted_state]
            
            # Find bin
            for (lo, hi) in bins:
                if lo <= prob < hi or (hi == 1.0 and prob == 1.0):
                    bin_counts[(lo, hi)]['total'] += 1
                    bin_counts[(lo, hi)]['prob_sum'] += prob
                    if predicted_state == actual:
                        bin_counts[(lo, hi)]['correct'] += 1
                    break
        
        # Calculate reliability (calibration error)
        reliability = 0.0
        total = sum(b['total'] for b in bin_counts.values())
        
        for bin_range, counts in bin_counts.items():
            if counts['total'] > 0:
                avg_prob = counts['prob_sum'] / counts['total']
                observed_freq = counts['correct'] / counts['total']
                weight = counts['total'] / total
                reliability += weight * (avg_prob - observed_freq) ** 2
        
        return {
            'reliability': reliability,
            'bins': {f"{lo:.1f}-{hi:.1f}": counts for (lo, hi), counts in bin_counts.items()}
        }
    
    def _calculate_brier_score(self, distributions: List[Dict], actual_states: List[str]) -> float:
        """
        Calculate Brier score for probabilistic predictions.
        
        Brier = mean((prob - indicator)^2) over all states and samples
        Lower is better, 0 = perfect, 0.25 = random for binary.
        """
        if not distributions or not actual_states:
            return 1.0
        
        all_states = ['Strong_Bearish', 'Bearish', 'Slight_Bearish', 'Neutral',
                      'Slight_Bullish', 'Bullish', 'Strong_Bullish']
        
        total_error = 0.0
        count = 0
        
        for dist, actual in zip(distributions, actual_states):
            if not dist:
                continue
            
            for state in all_states:
                prob = dist.get(state, 0.0)
                indicator = 1.0 if state == actual else 0.0
                total_error += (prob - indicator) ** 2
            
            count += 1
        
        if count == 0:
            return 1.0
        
        return total_error / (count * len(all_states))
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _calculate_state_distribution(self, states: List[str]) -> Tuple[Dict[str, float], Optional[str], List[str]]:
        """
        Bereken state distributie en detecteer stuck/dead states.
        
        Returns:
            Tuple van (distribution dict, stuck_state of None, list van dead_states)
        """
        if not states:
            return {}, None, []
        
        counter = Counter(states)
        total = len(states)
        distribution = {state: count / total for state, count in counter.items()}
        
        # Check voor stuck state
        stuck_state = None
        for state, pct in distribution.items():
            if pct >= DIAGNOSTIC_THRESHOLDS['stuck_state_pct']:
                stuck_state = state
                break
        
        # Check voor dead states (states die NOOIT voorkomen of < threshold)
        # REASON: We kennen de verwachte states niet altijd, dus we checken alleen op zeer lage frequentie
        dead_states = [state for state, pct in distribution.items() 
                       if pct < DIAGNOSTIC_THRESHOLDS['dead_state_threshold']]
        
        return distribution, stuck_state, dead_states
    
    def _calculate_mutual_information(self, node_states: List[str], outcome_states: List) -> float:
        """
        Bereken Mutual Information tussen node output en outcome.
        
        MI(X;Y) = H(Y) - H(Y|X)
        
        Higher MI = node geeft meer informatie over outcome.
        """
        # Filter None values
        valid_pairs = [(n, o) for n, o in zip(node_states, outcome_states) 
                       if n is not None and o is not None and not pd.isna(o)]
        
        if len(valid_pairs) < 10:
            return 0.0
        
        node_vals = [p[0] for p in valid_pairs]
        outcome_vals = [int(p[1]) for p in valid_pairs]
        
        # Convert to numpy arrays
        n = len(valid_pairs)
        
        # P(Y) - outcome marginal
        outcome_counter = Counter(outcome_vals)
        p_y = np.array([outcome_counter[y] / n for y in outcome_counter])
        h_y = -np.sum(p_y * np.log2(p_y + 1e-10))
        
        # H(Y|X) - conditional entropy
        node_counter = Counter(node_vals)
        h_y_given_x = 0.0
        
        for x, x_count in node_counter.items():
            p_x = x_count / n
            # P(Y|X=x)
            outcomes_given_x = [o for nv, o in valid_pairs if nv == x]
            if outcomes_given_x:
                outcome_given_x_counter = Counter(outcomes_given_x)
                p_y_given_x = np.array([outcome_given_x_counter[y] / len(outcomes_given_x) 
                                        for y in outcome_given_x_counter])
                h_y_given_x_val = -np.sum(p_y_given_x * np.log2(p_y_given_x + 1e-10))
                h_y_given_x += p_x * h_y_given_x_val
        
        mi = h_y - h_y_given_x
        return max(0.0, mi)  # MI should be non-negative
    
    def _calculate_directional_alignment(self, predictions: List[str], outcomes: List) -> Dict[str, float]:
        """
        Bereken directional alignment: hoe vaak is de predicted direction correct?
        
        Returns:
            Dict met alignment percentages per direction category
        """
        def get_direction(state: str) -> int:
            if state is None:
                return 0
            state_lower = state.lower()
            if 'bullish' in state_lower or 'long' in state_lower:
                return 1
            if 'bearish' in state_lower or 'short' in state_lower:
                return -1
            return 0
        
        def outcome_direction(outcome) -> int:
            if outcome is None or pd.isna(outcome):
                return 0
            return 1 if outcome > 0 else (-1 if outcome < 0 else 0)
        
        # Filter valid pairs
        valid = [(p, o) for p, o in zip(predictions, outcomes) 
                 if p is not None and o is not None and not pd.isna(o)]
        
        if not valid:
            return {}
        
        results = {}
        
        # Overall directional accuracy
        correct = sum(1 for p, o in valid if get_direction(p) == outcome_direction(o))
        results['overall'] = correct / len(valid) if valid else 0.0
        
        # Per-direction analysis
        for dir_name, dir_val in [('bullish', 1), ('bearish', -1)]:
            dir_preds = [(p, o) for p, o in valid if get_direction(p) == dir_val]
            if dir_preds:
                correct_dir = sum(1 for p, o in dir_preds if outcome_direction(o) == dir_val)
                results[f'{dir_name}_accuracy'] = correct_dir / len(dir_preds)
                results[f'{dir_name}_count'] = len(dir_preds)
        
        return results

    def _fetch_test_data_by_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Haal historische data op voor diagnostic."""
        # REASON: JOIN op qbn.barrier_outcomes in plaats van legacy signal_outcomes
        query = """
        SELECT 
            i.time as time_1, i.asset_id,
            l.*, c.*, f.*,
            bo.first_significant_barrier,
            bo.first_significant_time_min
        FROM kfl.indicators i
        JOIN kfl.mtf_signals_lead l ON l.asset_id = i.asset_id AND l.time_1 = i.time
        JOIN kfl.mtf_signals_coin c ON c.asset_id = i.asset_id AND c.time_1 = i.time
        JOIN kfl.mtf_signals_conf f ON f.asset_id = i.asset_id AND f.time_1 = i.time
        LEFT JOIN qbn.barrier_outcomes bo ON bo.asset_id = i.asset_id AND bo.time_1 = i.time
        WHERE i.asset_id = %s 
          AND i.interval_min = '60'
          AND i.time BETWEEN %s AND %s
        ORDER BY i.time ASC
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id, start_date, end_date))
            desc = cur.description
            if not desc:
                return pd.DataFrame()
            columns = [d[0] for d in desc]
            rows = cur.fetchall()
        
        df = pd.DataFrame(rows, columns=columns)
        
        # REASON: On-the-fly mapping van barriers naar numerieke outcomes voor alle 3 horizons
        if not df.empty:
            for horizon, window_min in [('1h', 60), ('4h', 240), ('1d', 1440)]:
                df[f'outcome_{horizon}'] = 0
                is_up = df['first_significant_barrier'].str.startswith('up', na=False)
                is_down = df['first_significant_barrier'].str.startswith('down', na=False)
                is_within = df['first_significant_time_min'] <= window_min
                
                df.loc[is_up & is_within, f'outcome_{horizon}'] = 1
                df.loc[is_down & is_within, f'outcome_{horizon}'] = -1
                
        return df
