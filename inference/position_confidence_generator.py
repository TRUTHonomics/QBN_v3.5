"""
Position_Confidence Generator voor QBN v3.2

v3.2 REDESIGN (Delta-based):
- Parents: Delta_Coincident, Delta_Confirming, Time_Since_Entry
- Delta scores: verandering in composites sinds entry (direction-aware)
- Uniqueness weighting (López de Prado) voor serial correlation correctie
- Gewogen counting in CPT generatie

v3.1 LEGACY:
- Parents: Coincident_Composite, Confirming_Composite, Time_Since_Entry
- Training op event-gefilterde data (absolute composite states)
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

from .node_types import CompositeState
from core.config_defaults import (
    DEFAULT_DELTA_THRESHOLD_COINCIDENT,
    DEFAULT_DELTA_THRESHOLD_CONFIRMING
)

logger = logging.getLogger(__name__)

@dataclass
class PositionConfidenceMetrics:
    """Metrics voor Position_Confidence distributie."""
    observations: int
    weighted_observations: float  # v3.2: gewogen observaties
    low_pct: float
    medium_pct: float
    high_pct: float
    coverage: float
    avg_entropy: float
    
    # v3.2: Delta-specific metrics
    delta_coinc_threshold: float = 0.0
    delta_conf_threshold: float = 0.0


# v3.2: Delta states voor discretisatie
DELTA_STATES = ['deteriorating', 'stable', 'improving']


class PositionConfidenceGenerator:
    """
    v3.2 Data-driven CPT generator voor Position_Confidence.
    
    NIEUW in v3.2:
    - Delta-based parents (verandering sinds entry)
    - Uniqueness weighting (1/N per event)
    - Gewogen counting voor CPT
    
    Parents v3.2: Delta_Coincident, Delta_Confirming, Time_Since_Entry
    Parents v3.1 (legacy): Coincident_Composite, Confirming_Composite, Time_Since_Entry
    """
    
    CONFIDENCE_STATES = ["low", "medium", "high"]
    TIME_BUCKETS = ["0-1h", "1-4h", "4-12h", "12-24h"]
    
    # v3.2: Delta states
    DELTA_STATES = DELTA_STATES
    
    def __init__(
        self, 
        laplace_alpha: float = 1.0,
        use_delta_mode: bool = True,  # v3.2: default delta mode
        delta_threshold_coinc: float = DEFAULT_DELTA_THRESHOLD_COINCIDENT,
        delta_threshold_conf: float = DEFAULT_DELTA_THRESHOLD_CONFIRMING
    ):
        self.laplace_alpha = laplace_alpha
        self.use_delta_mode = use_delta_mode
        self.delta_threshold_coinc = delta_threshold_coinc
        self.delta_threshold_conf = delta_threshold_conf
        
        self._cpt: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self._composite_states = [s.value for s in CompositeState]
        self._metrics: Optional[PositionConfidenceMetrics] = None
        
        # v3.0 fallback mapping (deprecated)
        self._static_mapping = {
            'strong_bullish': 'high',
            'bullish': 'high', 
            'neutral': 'medium',
            'bearish': 'low',
            'strong_bearish': 'low'
        }
    
    def generate_cpt(
        self, 
        training_data: pd.DataFrame,
        direction_column: str = 'event_direction'
    ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        """
        Genereer CPT van event-gefilterde training data.
        
        v3.2 DELTA MODE:
            training_data moet bevatten:
                - delta_cum_coincident: float (direction-aware delta)
                - delta_cum_confirming: float (direction-aware delta)
                - time_since_entry_min: int/float
                - event_outcome: str (up_strong, down_weak, etc.)
                - event_direction: str (long/short)
                - uniqueness_weight: float (1/N per event)
        
        v3.1 LEGACY MODE:
            training_data moet bevatten:
                - coincident_composite: str
                - confirming_composite: str
                - time_since_entry_min: int/float
                - event_outcome: str
                - event_direction: str
        
        Returns:
            CPT: Dict[(coinc_state, conf_state, time_bucket)] -> Dict[confidence] -> prob
        """
        if self.use_delta_mode:
            return self._generate_cpt_delta(training_data, direction_column)
        else:
            return self._generate_cpt_legacy(training_data, direction_column)
    
    def _generate_cpt_delta(
        self,
        training_data: pd.DataFrame,
        direction_column: str = 'event_direction'
    ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        """
        v3.2: Delta-based CPT generatie met uniqueness weighting.
        """
        logger.info(f"🔧 Generating Position_Confidence CPT v3.2 (delta mode, alpha={self.laplace_alpha})...")
        logger.info(f"   Delta thresholds: coinc={self.delta_threshold_coinc}, conf={self.delta_threshold_conf}")
        
        # Check vereiste kolommen
        required_cols = ['delta_cum_coincident', 'delta_cum_confirming', 'event_outcome']
        missing = [c for c in required_cols if c not in training_data.columns]
        if missing:
            logger.warning(f"Missing columns for delta mode: {missing}. Falling back to legacy mode.")
            return self._generate_cpt_legacy(training_data, direction_column)
        
        # Gewogen counts
        counts = defaultdict(lambda: defaultdict(float))  # float voor gewogen counting
        total_weighted_obs = 0.0
        total_raw_obs = 0
        
        for _, row in training_data.iterrows():
            # Haal delta waarden op
            delta_coinc = row.get('delta_cum_coincident', 0.0) or 0.0
            delta_conf = row.get('delta_cum_confirming', 0.0) or 0.0
            
            # Discretiseer delta naar state
            coinc_state = self._discretize_delta(delta_coinc, self.delta_threshold_coinc)
            conf_state = self._discretize_delta(delta_conf, self.delta_threshold_conf)
            
            # Time bucket
            time_bucket = row.get('time_bucket')
            if time_bucket is None:
                time_min = row.get('time_since_entry_min', 0) or 0
                time_bucket = self._discretize_time(time_min)
            
            # Derive label van outcome
            label = self._derive_confidence_label(
                row.get('event_outcome', 'neutral'),
                row.get(direction_column, 'long')
            )
            
            # Uniqueness weight (López de Prado)
            weight = row.get('uniqueness_weight', 1.0) or 1.0
            
            key = (coinc_state, conf_state, time_bucket)
            counts[key][label] += weight
            total_weighted_obs += weight
            total_raw_obs += 1
        
        # Build CPT met Laplace smoothing (gewogen)
        cpt = {}
        parent_states = self.DELTA_STATES  # deteriorating, stable, improving
        
        for coinc_state in parent_states:
            for conf_state in parent_states:
                for time_bucket in self.TIME_BUCKETS:
                    key = (coinc_state, conf_state, time_bucket)
                    state_counts = counts[key]
                    
                    total = sum(state_counts.values()) + \
                            self.laplace_alpha * len(self.CONFIDENCE_STATES)
                    
                    cpt[key] = {}
                    for state in self.CONFIDENCE_STATES:
                        count = state_counts.get(state, 0)
                        cpt[key][state] = (count + self.laplace_alpha) / total
        
        self._cpt = cpt
        
        # Calculate metrics
        self._metrics = self._calculate_metrics_delta(counts, total_raw_obs, total_weighted_obs)
        
        logger.info(f"✅ Position_Confidence CPT v3.2 generated: {len(cpt)} keys, "
                   f"{total_raw_obs} raw obs, {total_weighted_obs:.1f} weighted obs")
        return cpt
    
    def _generate_cpt_legacy(
        self,
        training_data: pd.DataFrame,
        direction_column: str = 'event_direction'
    ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        """
        v3.1: Legacy CPT generatie met absolute composite states.
        """
        logger.info(f"🔧 Generating Position_Confidence CPT v3.1 (legacy mode, alpha={self.laplace_alpha})...")
        
        counts = defaultdict(lambda: defaultdict(int))
        total_obs = 0
        
        for _, row in training_data.iterrows():
            coinc = row.get('coincident_composite', 'neutral')
            conf = row.get('confirming_composite', 'neutral')
            
            # Use discretize_time if time_bucket is not provided
            time_bucket = row.get('time_bucket')
            if time_bucket is None:
                time_min = row.get('time_since_entry_min', 0)
                time_bucket = self._discretize_time(time_min)
            
            # Derive label van outcome
            label = self._derive_confidence_label(
                row.get('event_outcome', 'neutral'),
                row.get(direction_column, 'long')
            )
            
            key = (coinc, conf, time_bucket)
            counts[key][label] += 1
            total_obs += 1
            
        # Build CPT met Laplace smoothing
        cpt = {}
        
        for coinc in self._composite_states:
            for conf in self._composite_states:
                for time_bucket in self.TIME_BUCKETS:
                    key = (coinc, conf, time_bucket)
                    state_counts = counts[key]
                    
                    total = sum(state_counts.values()) + \
                            self.laplace_alpha * len(self.CONFIDENCE_STATES)
                    
                    cpt[key] = {}
                    for state in self.CONFIDENCE_STATES:
                        count = state_counts.get(state, 0)
                        cpt[key][state] = (count + self.laplace_alpha) / total
                        
        self._cpt = cpt
        
        # Calculate metrics
        self._metrics = self._calculate_metrics(counts, total_obs)
        
        logger.info(f"✅ Position_Confidence CPT v3.1 generated: {len(cpt)} keys, {total_obs} observations")
        return cpt
    
    def _discretize_delta(self, delta: float, threshold: float) -> str:
        """
        v3.2: Discretiseer delta waarde naar state.
        
        Args:
            delta: Delta waarde (direction-aware, positief = gunstig)
            threshold: Threshold voor discretisatie
            
        Returns:
            State: 'deteriorating', 'stable', of 'improving'
        """
        if delta < -threshold:
            return 'deteriorating'
        elif delta > threshold:
            return 'improving'
        return 'stable'
    
    def get_confidence(
        self, 
        coincident: str, 
        confirming: str, 
        time_since_entry_min: int = 0
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Bereken confidence voor gegeven inputs.
        
        Returns:
            Tuple van (beste_state, confidence_score, distributie)
        """
        time_bucket = self._discretize_time(time_since_entry_min)
        key = (coincident, confirming, time_bucket)
        
        distribution = self._cpt.get(key, self._uniform_distribution())
        best_state = max(distribution, key=distribution.get)
        confidence_score = distribution[best_state]
        
        return best_state, confidence_score, distribution

    def _derive_confidence_label(
        self, 
        event_outcome: str,
        direction: str
    ) -> str:
        """
        Derive confidence label op basis van event outcome.
        """
        # Outcome definitions
        outcome_is_up = str(event_outcome).startswith('up_')
        outcome_is_strong = 'strong' in str(event_outcome)
        outcome_is_neutral = event_outcome in ['neutral', 'timeout', 'none']
        
        direction_is_long = direction == 'long'
        
        # Alignment check
        aligned = (direction_is_long and outcome_is_up) or \
                  (not direction_is_long and not outcome_is_up and not outcome_is_neutral)
        
        if outcome_is_neutral:
            return 'low'
        elif aligned and outcome_is_strong:
            return 'high'
        elif aligned:
            return 'medium'
        else:
            return 'low'

    def _discretize_time(self, minutes: float) -> str:
        """Convert minuten naar time bucket."""
        if minutes <= 60:
            return '0-1h'
        elif minutes <= 240:
            return '1-4h'
        elif minutes <= 720:
            return '4-12h'
        else:
            return '12-24h'

    def _uniform_distribution(self) -> Dict[str, float]:
        """Fallback uniform distribution."""
        p = 1.0 / len(self.CONFIDENCE_STATES)
        return {s: p for s in self.CONFIDENCE_STATES}

    def _calculate_metrics(self, counts: Dict, total_obs: int) -> PositionConfidenceMetrics:
        """Bereken quality metrics voor de CPT (legacy mode)."""
        from collections import Counter
        # Coverage
        expected_keys = (len(self._composite_states) ** 2) * len(self.TIME_BUCKETS)
        actual_keys = sum(1 for k in counts if sum(counts[k].values()) > 0)
        
        # Stats per state
        all_labels = []
        for state_counts in counts.values():
            for label, count in state_counts.items():
                all_labels.extend([label] * int(count))
        
        label_counts = Counter(all_labels) if all_labels else {}
        
        # Entropy
        entropies = []
        for dist in self._cpt.values():
            probs = list(dist.values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            entropies.append(entropy)
            
        return PositionConfidenceMetrics(
            observations=total_obs,
            weighted_observations=float(total_obs),
            low_pct=label_counts.get('low', 0) / max(1, total_obs),
            medium_pct=label_counts.get('medium', 0) / max(1, total_obs),
            high_pct=label_counts.get('high', 0) / max(1, total_obs),
            coverage=actual_keys / expected_keys,
            avg_entropy=np.mean(entropies) if entropies else 0,
            delta_coinc_threshold=0.0,
            delta_conf_threshold=0.0
        )
    
    def _calculate_metrics_delta(
        self, 
        counts: Dict, 
        total_raw_obs: int,
        total_weighted_obs: float
    ) -> PositionConfidenceMetrics:
        """Bereken quality metrics voor de CPT (delta mode)."""
        # Coverage voor delta states
        expected_keys = (len(self.DELTA_STATES) ** 2) * len(self.TIME_BUCKETS)
        actual_keys = sum(1 for k in counts if sum(counts[k].values()) > 0)
        
        # Gewogen stats per state
        label_weights = defaultdict(float)
        for state_counts in counts.values():
            for label, weight in state_counts.items():
                label_weights[label] += weight
        
        # Entropy
        entropies = []
        for dist in self._cpt.values():
            probs = list(dist.values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            entropies.append(entropy)
        
        return PositionConfidenceMetrics(
            observations=total_raw_obs,
            weighted_observations=total_weighted_obs,
            low_pct=label_weights.get('low', 0) / max(1, total_weighted_obs),
            medium_pct=label_weights.get('medium', 0) / max(1, total_weighted_obs),
            high_pct=label_weights.get('high', 0) / max(1, total_weighted_obs),
            coverage=actual_keys / expected_keys,
            avg_entropy=np.mean(entropies) if entropies else 0,
            delta_coinc_threshold=self.delta_threshold_coinc,
            delta_conf_threshold=self.delta_threshold_conf
        )

    # ========================================================================
    # DB INTERFACE
    # ========================================================================

    def save_cpt(self, asset_id: int, run_id: str):
        """Sla Position_Confidence CPT op in database."""
        from inference.cpt_cache_manager import CPTCacheManager
        cache_manager = CPTCacheManager()
        
        # Bepaal parents en versie op basis van mode
        if self.use_delta_mode:
            parents = ['Delta_Coincident', 'Delta_Confirming', 'Time_Since_Entry']
            version = '3.2'
            training_method = 'delta_weighted'
        else:
            parents = ['Coincident_Composite', 'Confirming_Composite', 'Time_Since_Entry']
            version = '3.1'
            training_method = 'event_driven'
        
        cpt_data = {
            'node_name': 'Position_Confidence',
            'parents': parents,
            'states': self.CONFIDENCE_STATES,
            'conditional_probabilities': {
                f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self._cpt.items()
            },
            'metadata': {
                'version': version,
                'training_method': training_method,
                'laplace_alpha': self.laplace_alpha,
                'delta_mode': self.use_delta_mode,
                'delta_threshold_coinc': self.delta_threshold_coinc if self.use_delta_mode else None,
                'delta_threshold_conf': self.delta_threshold_conf if self.use_delta_mode else None,
                'metrics': asdict(self._metrics) if self._metrics else {}
            },
            'observations': self._metrics.observations if self._metrics else 0
        }
        
        cache_manager.save_cpt(
            asset_id=asset_id,
            node_name='Position_Confidence',
            cpt_data=cpt_data,
            scope_type='single',
            scope_key=f"asset_{asset_id}",
            source_assets=[asset_id],
            run_id=run_id
        )
        logger.info(f"💾 Saved Position_Confidence CPT v{version} for asset {asset_id}")

    # ========================================================================
    # DEPRECATED BACKWARDS COMPATIBILITY
    # ========================================================================
    
    def derive_confidence(self, confirming: str) -> str:
        """
        DEPRECATED: Statische mapping voor v3.0 backwards compat.
        Gebruik get_confidence() voor v3.1.
        """
        return self._static_mapping.get(confirming, 'medium')
    
    def get_risk_parameters(self, confidence: str) -> Dict:
        """
        DEPRECATED: Risk parameters voor TSEM.
        """
        params = {
            'high': {'stop_loss_atr': 2.0, 'position_size_pct': 100, 'action': 'hold'},
            'medium': {'stop_loss_atr': 1.5, 'position_size_pct': 75, 'action': 'monitor'},
            'low': {'stop_loss_atr': 1.0, 'position_size_pct': 50, 'action': 'consider_partial_exit'}
        }
        return params.get(confidence, params['medium'])
