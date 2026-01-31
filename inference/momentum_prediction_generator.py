"""
Momentum_Prediction Generator voor QBN v3.3 Triple Composite Architecture

Genereert CPT voor Momentum_Prediction node:
- Parent: Delta_Leading + Time_Since_Entry
- States: bearish, neutral, bullish
- Training: Gebaseerd op return_since_entry labels

USAGE:
    from inference.momentum_prediction_generator import MomentumPredictionGenerator
    
    generator = MomentumPredictionGenerator()
    cpt = generator.generate_cpt(event_window_data)
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

from .position_label_generator import derive_momentum_labels_vectorized

logger = logging.getLogger(__name__)


# Delta states voor Leading (zelfde als Coincident/Confirming)
DELTA_STATES = ['deteriorating', 'stable', 'improving']
TIME_BUCKETS = ["0-1h", "1-4h", "4-12h", "12-24h"]
MOMENTUM_STATES = ["bearish", "neutral", "bullish"]

# Default delta threshold voor leading
DEFAULT_DELTA_THRESHOLD_LEADING = 0.03


@dataclass
class MomentumPredictionMetrics:
    """Metrics voor Momentum_Prediction distributie."""
    observations: int
    weighted_observations: float
    bearish_pct: float
    neutral_pct: float
    bullish_pct: float
    coverage: float
    avg_entropy: float


class MomentumPredictionGenerator:
    """
    v3.3 CPT generator voor Momentum_Prediction node.
    
    Parents: Delta_Leading, Time_Since_Entry
    States: bearish, neutral, bullish
    Training: Op event window data met return_since_entry
    """
    
    def __init__(
        self,
        laplace_alpha: float = 1.0,
        delta_threshold: float = DEFAULT_DELTA_THRESHOLD_LEADING
    ):
        self.laplace_alpha = laplace_alpha
        self.delta_threshold = delta_threshold
        self._cpt: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._metrics: Optional[MomentumPredictionMetrics] = None
    
    def generate_cpt(
        self,
        training_data: pd.DataFrame,
        use_ida_weights: bool = True
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Genereer CPT van event window training data.
        
        Args:
            training_data: DataFrame met columns:
                - delta_cum_leading: float
                - time_since_entry_min: float
                - return_since_entry: float (voor label)
                - uniqueness_weight: float (optioneel, IDA weights)
            use_ida_weights: Of IDA weights moeten worden gebruikt
        
        Returns:
            Dict[Tuple[delta_state, time_bucket], Dict[momentum_state, probability]]
        """
        logger.info(f"🔧 Generating Momentum_Prediction CPT v3.3 (alpha={self.laplace_alpha})...")
        
        # Check required columns
        required_cols = ['time_since_entry_min']
        for col in required_cols:
            if col not in training_data.columns:
                logger.error(f"Missing required column: {col}")
                return {}
        
        # Filter op event data (moet binnen een event window zijn)
        event_data = training_data[training_data['event_id'].notna()].copy()
        if len(event_data) == 0:
            logger.warning("No event data found for Momentum_Prediction training")
            return {}
        
        # Genereer labels als niet aanwezig
        if 'momentum_label' not in event_data.columns:
            if 'return_since_entry' in event_data.columns:
                event_data['momentum_label'] = derive_momentum_labels_vectorized(
                    event_data['return_since_entry']
                )
            else:
                logger.error("No return_since_entry column for momentum label generation")
                return {}
        
        # Discretiseer delta_leading
        if 'delta_cum_leading' in event_data.columns:
            event_data['delta_leading_state'] = event_data['delta_cum_leading'].apply(
                lambda x: self._discretize_delta(x)
            )
        else:
            # Fallback: gebruik leading_score direct als geen delta beschikbaar
            logger.warning("delta_cum_leading not found, using 'stable' as default")
            event_data['delta_leading_state'] = 'stable'
        
        # Discretiseer time
        event_data['time_bucket'] = event_data['time_since_entry_min'].apply(
            self._discretize_time
        )
        
        # Get weights
        if use_ida_weights and 'uniqueness_weight' in event_data.columns:
            weights = event_data['uniqueness_weight'].fillna(1.0)
            mode_str = "IDA weighted"
        else:
            weights = pd.Series(1.0, index=event_data.index)
            mode_str = "unweighted"
        
        logger.info(f"   Mode: {mode_str}, {len(event_data)} observations")
        
        # Build CPT with weighted counting
        counts = defaultdict(lambda: defaultdict(float))
        
        for idx, row in event_data.iterrows():
            delta_state = row['delta_leading_state']
            time_bucket = row['time_bucket']
            momentum_label = row['momentum_label']
            weight = weights.loc[idx]
            
            parent_key = (delta_state, time_bucket)
            counts[parent_key][momentum_label] += weight
        
        # Convert to probabilities with Laplace smoothing
        cpt = {}
        total_raw = 0
        total_weighted = 0.0
        
        for delta_state in DELTA_STATES:
            for time_bucket in TIME_BUCKETS:
                parent_key = (delta_state, time_bucket)
                state_counts = counts[parent_key]
                
                total = sum(state_counts.values())
                total_raw += len([c for c in state_counts.values() if c > 0])
                total_weighted += total
                
                # Laplace smoothing
                probs = {}
                denominator = total + self.laplace_alpha * len(MOMENTUM_STATES)
                
                for state in MOMENTUM_STATES:
                    count = state_counts.get(state, 0)
                    probs[state] = (count + self.laplace_alpha) / denominator
                
                cpt[parent_key] = probs
        
        self._cpt = cpt
        
        # Calculate metrics
        label_counts = event_data['momentum_label'].value_counts()
        total_obs = len(event_data)
        
        self._metrics = MomentumPredictionMetrics(
            observations=total_obs,
            weighted_observations=total_weighted,
            bearish_pct=label_counts.get('bearish', 0) / total_obs if total_obs > 0 else 0,
            neutral_pct=label_counts.get('neutral', 0) / total_obs if total_obs > 0 else 0,
            bullish_pct=label_counts.get('bullish', 0) / total_obs if total_obs > 0 else 0,
            coverage=len([k for k, v in cpt.items() if sum(v.values()) > len(MOMENTUM_STATES) * self.laplace_alpha]) / len(cpt),
            avg_entropy=self._calculate_avg_entropy(cpt)
        )
        
        logger.info(f"✅ Momentum_Prediction CPT generated: {len(cpt)} keys, "
                   f"{total_obs} obs, {total_weighted:.1f} weighted")
        logger.info(f"   Distribution: bearish={self._metrics.bearish_pct:.1%}, "
                   f"neutral={self._metrics.neutral_pct:.1%}, bullish={self._metrics.bullish_pct:.1%}")
        
        return cpt
    
    def _discretize_delta(self, delta: float) -> str:
        """Discretiseer delta score naar state."""
        if pd.isna(delta):
            return 'stable'
        if delta < -self.delta_threshold:
            return 'deteriorating'
        elif delta > self.delta_threshold:
            return 'improving'
        else:
            return 'stable'
    
    def _discretize_time(self, minutes: float) -> str:
        """Discretiseer time_since_entry naar bucket."""
        if pd.isna(minutes) or minutes < 60:
            return "0-1h"
        elif minutes < 240:
            return "1-4h"
        elif minutes < 720:
            return "4-12h"
        else:
            return "12-24h"
    
    def _calculate_avg_entropy(self, cpt: Dict) -> float:
        """Bereken gemiddelde entropy over CPT entries."""
        entropies = []
        for probs in cpt.values():
            prob_values = list(probs.values())
            # Shannon entropy
            entropy = -sum(p * np.log2(p + 1e-10) for p in prob_values if p > 0)
            entropies.append(entropy)
        return np.mean(entropies) if entropies else 0.0
    
    def get_cpt_data(self, asset_id: int) -> Dict[str, Any]:
        """Format CPT voor database opslag."""
        if not self._cpt:
            return {}
        
        return {
            'node_name': 'Momentum_Prediction',
            'parents': ['Delta_Leading', 'Time_Since_Entry'],
            'states': MOMENTUM_STATES,
            'conditional_probabilities': {
                f"{k[0]}|{k[1]}": v for k, v in self._cpt.items()
            },
            'metrics': asdict(self._metrics) if self._metrics else {}
        }
    
    def predict(
        self,
        delta_leading_state: str,
        time_bucket: str
    ) -> Dict[str, float]:
        """
        Inference: voorspel momentum gegeven parent states.
        
        Args:
            delta_leading_state: 'deteriorating', 'stable', of 'improving'
            time_bucket: '0-1h', '1-4h', '4-12h', of '12-24h'
        
        Returns:
            Dict met probabilities voor elke momentum state
        """
        parent_key = (delta_leading_state, time_bucket)
        
        if parent_key in self._cpt:
            return self._cpt[parent_key]
        
        # Fallback: uniform distribution
        logger.warning(f"No CPT entry for {parent_key}, returning uniform")
        return {state: 1/len(MOMENTUM_STATES) for state in MOMENTUM_STATES}
