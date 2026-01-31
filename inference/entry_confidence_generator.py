"""
Entry_Confidence Generator voor QBN v3

[DEPRECATED in v3.1]

Deze module is deprecated en wordt niet meer gebruikt voor prediction nodes.
In v3.1 is Entry_Confidence verwijderd als parent van Prediction_1h/4h/1d.

De functionaliteit blijft beschikbaar voor backwards compatibility en legacy
analyse, maar nieuwe code moet Position_Confidence gebruiken voor risk
management decisions.

MIGRATION:
    v3.0: P(Prediction | Regime, Hypothesis, Entry_Confidence)
    v3.1: P(Prediction | Regime, Hypothesis)

Genereert CPTs voor de Entry_Confidence node op basis van:
- Coincident_Composite state
- Confirming_Composite state
- Historische outcomes

REASON: Voorstel 2 - Timing confidence uit Coincident + Confirming
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import product

import logging
import warnings

from .node_types import CompositeState
from .alignment_engine import AlignmentEngine, AlignmentConfig
from .timing_precision_analyzer import TimingPrecisionAnalyzer, TimingPrecisionMetrics

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceMetrics:
    """Metrics voor Entry_Confidence distributie analyse."""
    low_pct: float
    medium_pct: float
    high_pct: float
    alignment_score_mean: float
    stability: float


class EntryConfidenceGenerator:
    """
    Generator voor Entry_Confidence CPTs.

    [DEPRECATED in v3.1] - Entry_Confidence is niet meer parent van Prediction nodes.
    Deze class blijft beschikbaar voor backwards compatibility.

    Combineert Coincident + Confirming → Entry_Confidence.
    """

    CONFIDENCE_STATES = ["low", "medium", "high"]

    def __init__(self, laplace_alpha: float = 1.0, suppress_warning: bool = False):
        if not suppress_warning:
            warnings.warn(
                "EntryConfidenceGenerator is deprecated in v3.1. "
                "Entry_Confidence is niet meer parent van Prediction nodes. "
                "Gebruik suppress_warning=True om deze warning te onderdrukken.",
                DeprecationWarning,
                stacklevel=2
            )
            logger.warning(
                "[DEPRECATED] EntryConfidenceGenerator instantiated - "
                "Entry_Confidence is verwijderd als Prediction parent in v3.1"
            )

        self.laplace_alpha = laplace_alpha
        self._composite_states = [s.value for s in CompositeState]
        self.alignment_engine = AlignmentEngine()
        self.analyzer = TimingPrecisionAnalyzer()
    
    def generate_cpt(
        self,
        training_data: List[Dict]
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Genereer Entry_Confidence CPT.
        
        Args:
            training_data: List van {coincident_composite, confirming_composite, outcome} dicts
            
        Returns:
            CPT: Dict[(coincident, confirming)] -> Dict[confidence] -> probability
        """
        counts = defaultdict(lambda: defaultdict(int))
        
        for record in training_data:
            coincident = record.get('coincident_composite', 'neutral')
            confirming = record.get('confirming_composite', 'neutral')
            
            # Derive confidence
            confidence = self.derive_confidence(coincident, confirming)
            
            key = (coincident, confirming)
            counts[key][confidence] += 1
        
        # Build CPT met Laplace smoothing
        cpt = {}
        composite_states = [s.value for s in CompositeState]
        
        for coinc, conf in product(composite_states, composite_states):
            key = (coinc, conf)
            state_counts = counts[key]
            total = sum(state_counts.values()) + self.laplace_alpha * len(self.CONFIDENCE_STATES)
            
            cpt[key] = {}
            for conf_state in self.CONFIDENCE_STATES:
                count = state_counts.get(conf_state, 0)
                cpt[key][conf_state] = (count + self.laplace_alpha) / total
        
        return cpt
    
    def derive_confidence(self, coincident: str, confirming: str) -> str:
        """
        Derive Entry_Confidence state van Coincident + Confirming alignment.
        Gedelegeerd naar AlignmentEngine.
        """
        return self.alignment_engine.get_alignment(coincident, confirming).confidence
    
    def compute_alignment_score(self, coincident: str, confirming: str) -> float:
        """
        Bereken numerieke alignment score via AlignmentEngine.
        """
        return self.alignment_engine.get_alignment(coincident, confirming).score

    def analyze_distribution(
        self,
        inference_results: List[Dict]
    ) -> ConfidenceMetrics:
        """
        Analyseer de Entry_Confidence verdeling over inference resultaten.
        """
        if not inference_results:
            return ConfidenceMetrics(0, 0, 0, 0, 0)
            
        confidence_counts = Counter(r.get('entry_confidence') for r in inference_results)
        total = len(inference_results)
        
        # Alignment scores en sterktes via engine
        alignments = [
            self.alignment_engine.get_alignment(
                r.get('coincident_composite', 'neutral'), 
                r.get('confirming_composite', 'neutral')
            )
            for r in inference_results
        ]
        
        scores = [a.score for a in alignments]
        
        return ConfidenceMetrics(
            low_pct=confidence_counts.get('low', 0) / total,
            medium_pct=confidence_counts.get('medium', 0) / total,
            high_pct=confidence_counts.get('high', 0) / total,
            alignment_score_mean=float(np.mean(scores)),
            stability=float(1.0 - np.std(scores))
        )

    def generate_outcome_refined_cpt(
        self,
        training_data: pd.DataFrame,
        horizon: str = '1h'
    ) -> Dict:
        """
        Genereer een CPT met outcome-based refinement (Phase 2.3).
        
        Two-phase approach:
        1. Base CPT van alignment logica (deterministisch).
        2. Refinement van geobserveerde win rates.
        """
        # Phase 1: Base CPT (deterministisch)
        base_cpt = self.generate_cpt([]) # Gebruik lege data voor pure deterministische base
        
        # Phase 2: Refinement
        refined_probs = self._refine_with_outcomes(
            base_cpt,
            training_data,
            horizon
        )
        
        # Bepaal metrics voor metadata
        # We moeten de training data tijdelijk voorzien van de derived confidence
        temp_data = training_data.copy()
        temp_data['entry_confidence'] = temp_data.apply(
            lambda r: self.derive_confidence(r['coincident_composite'], r['confirming_composite']),
            axis=1
        )
        metrics = self.analyzer.analyze_from_data(temp_data, horizon)
        
        return {
            'node_name': 'Entry_Confidence',
            'parents': ['Coincident_Composite', 'Confirming_Composite'],
            'states': self.CONFIDENCE_STATES,
            'conditional_probabilities': refined_probs,
            'metadata': {
                'horizon': horizon,
                'observations': len(training_data),
                'refinement_applied': True,
                'laplace_alpha': self.laplace_alpha,
                'timing_metrics': asdict(metrics)
            }
        }

    def _refine_with_outcomes(
        self,
        base_cpt: Dict,
        training_data: pd.DataFrame,
        horizon: str = '1h'
    ) -> Dict:
        """
        Refine CPT door te leren van outcome correlaties.
        Shift probability tussen confidence states op basis van win rate.
        """
        # REASON: Altijd barrier_state gebruiken
        outcome_col = 'barrier_state'
        refined_cpt = {}
        
        # Bereken win rates per parent combinatie
        # We gebruiken een groupby voor efficiëntie
        grouped = training_data.groupby(['coincident_composite', 'confirming_composite'])
        
        for parent_combo, distribution in base_cpt.items():
            coinc, conf = parent_combo
            derived = self.derive_confidence(coinc, conf)
            
            # Check of we data hebben voor deze combinatie
            if parent_combo in grouped.groups:
                group_data = grouped.get_group(parent_combo)
                
                if outcome_col in group_data.columns:
                    outcomes = group_data[outcome_col].dropna()
                    
                    if len(outcomes) >= 30: # Minimum sample size voor refinement
                        win_rate = outcomes.str.startswith('up_').sum() / len(outcomes)
                        # Pseudo-avg outcome voor barriers
                        def map_ret(s):
                            if 'strong' in s: return 2.0 if 'up' in s else -2.0
                            if 'weak' in s: return 1.0 if 'up' in s else -1.0
                            return 0.0
                        avg_outcome = outcomes.apply(map_ret).mean()
                        
                        # Pas distributie aan op basis van performance
                        refined_cpt[parent_combo] = self._adjust_distribution(
                            distribution, derived, win_rate, avg_outcome
                        )
                        continue
            
            # Geen data of te weinig data of geen kolom: behoud base distributie
            refined_cpt[parent_combo] = distribution
            
        return refined_cpt

    def _adjust_distribution(
        self,
        distribution: Dict[str, float],
        derived: str,
        win_rate: float,
        avg_outcome: float
    ) -> Dict[str, float]:
        """
        Pas distributie aan op basis van observed metrics.
        
        Regels (conform plan):
        - Als derived='high' maar win_rate < 0.52: shift naar 'medium'
        - Als derived='medium' maar win_rate > 0.58: shift naar 'high'
        - Als derived='low' maar win_rate > 0.52: shift naar 'medium'
        """
        adjusted = distribution.copy()
        SHIFT_FACTOR = 0.15 # Hoe veel we verschuiven
        
        # Drempelwaarden
        EXPECTED_WIN_HIGH = 0.55
        HIGH_WIN_THRESHOLD = 0.58
        
        if derived == 'high' and win_rate < EXPECTED_WIN_HIGH:
            # Downgrade: verschuif van high naar medium
            shift = min(SHIFT_FACTOR, adjusted['high'] * 0.5)
            adjusted['high'] -= shift
            adjusted['medium'] += shift
            
        elif derived == 'medium' and win_rate > HIGH_WIN_THRESHOLD:
            # Upgrade: verschuif van medium naar high
            shift = min(SHIFT_FACTOR, adjusted['medium'] * 0.5)
            adjusted['medium'] -= shift
            adjusted['high'] += shift
            
        elif derived == 'low' and win_rate > EXPECTED_WIN_HIGH:
            # Verrassend goed: verschuif van low naar medium
            shift = min(SHIFT_FACTOR, adjusted['low'] * 0.5)
            adjusted['low'] -= shift
            adjusted['medium'] += shift
            
        # Normaliseer
        total = sum(adjusted.values())
        return {k: v/total for k, v in adjusted.items()}

