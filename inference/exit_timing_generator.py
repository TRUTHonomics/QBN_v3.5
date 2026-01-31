"""
Exit_Timing Generator voor QBN v3.3 Triple Composite Architecture

Genereert CPT voor Exit_Timing node:
- Parent: Delta_Confirming + Time_Since_Entry + Current_PnL_ATR
- States: exit_now, hold, extend
- Training: Gebaseerd op retrospectieve exit analyse

USAGE:
    from inference.exit_timing_generator import ExitTimingGenerator
    
    generator = ExitTimingGenerator()
    cpt = generator.generate_cpt(event_window_data)
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

from .position_label_generator import derive_exit_timing_labels_vectorized, _derive_exit_timing_from_outcome
from core.config_defaults import DEFAULT_DELTA_THRESHOLD_CONFIRMING

logger = logging.getLogger(__name__)


# Delta states
DELTA_STATES = ['deteriorating', 'stable', 'improving']
TIME_BUCKETS = ["0-1h", "1-4h", "4-12h", "12-24h"]
PNL_STATES = ["losing", "breakeven", "winning"]
EXIT_TIMING_STATES = ["exit_now", "hold", "extend"]


@dataclass
class ExitTimingMetrics:
    """Metrics voor Exit_Timing distributie."""
    observations: int
    weighted_observations: float
    exit_now_pct: float
    hold_pct: float
    extend_pct: float
    coverage: float
    avg_entropy: float


class ExitTimingGenerator:
    """
    v3.3 CPT generator voor Exit_Timing node.
    
    Parents: Delta_Confirming, Time_Since_Entry, Current_PnL_ATR
    States: exit_now, hold, extend
    Training: Op event window data met retrospectieve exit analyse
    """
    
    def __init__(
        self,
        laplace_alpha: float = 1.0,
        delta_threshold: float = DEFAULT_DELTA_THRESHOLD_CONFIRMING
    ):
        self.laplace_alpha = laplace_alpha
        self.delta_threshold = delta_threshold
        self._cpt: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self._metrics: Optional[ExitTimingMetrics] = None
    
    def generate_cpt(
        self,
        training_data: pd.DataFrame,
        use_ida_weights: bool = True
    ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        """
        Genereer CPT van event window training data.
        
        Args:
            training_data: DataFrame met columns:
                - delta_cum_confirming: float
                - time_since_entry_min: float
                - current_pnl_atr: float (optioneel)
                - event_outcome: str (voor fallback labeling)
                - uniqueness_weight: float (optioneel, IDA weights)
            use_ida_weights: Of IDA weights moeten worden gebruikt
        
        Returns:
            Dict[Tuple[delta_state, time_bucket, pnl_state], Dict[exit_state, probability]]
        """
        logger.info(f"🔧 Generating Exit_Timing CPT v3.3 (alpha={self.laplace_alpha})...")
        
        # Check required columns
        required_cols = ['time_since_entry_min']
        for col in required_cols:
            if col not in training_data.columns:
                logger.error(f"Missing required column: {col}")
                return {}
        
        # Filter op event data
        event_data = training_data[training_data['event_id'].notna()].copy()
        if len(event_data) == 0:
            logger.warning("No event data found for Exit_Timing training")
            return {}
        
        # Genereer labels
        if 'exit_timing_label' not in event_data.columns:
            if 'pnl_current' in event_data.columns and 'max_profit_event' in event_data.columns:
                event_data['exit_timing_label'] = derive_exit_timing_labels_vectorized(
                    event_data['pnl_current'],
                    event_data['max_profit_event']
                )
            else:
                # Fallback: gebruik event_outcome + time
                event_data['exit_timing_label'] = _derive_exit_timing_from_outcome(event_data)
                logger.info("   Using outcome-based fallback for exit_timing labels")
        
        # Discretiseer delta_confirming
        if 'delta_cum_confirming' in event_data.columns:
            event_data['delta_confirming_state'] = event_data['delta_cum_confirming'].apply(
                lambda x: self._discretize_delta(x)
            )
        else:
            logger.warning("delta_cum_confirming not found, using 'stable' as default")
            event_data['delta_confirming_state'] = 'stable'
        
        # Discretiseer time
        event_data['time_bucket'] = event_data['time_since_entry_min'].apply(
            self._discretize_time
        )
        
        # Discretiseer PnL
        if 'current_pnl_atr' in event_data.columns:
            event_data['pnl_state'] = event_data['current_pnl_atr'].apply(
                self._discretize_pnl
            )
        else:
            # Fallback: probeer return_since_entry te gebruiken
            if 'return_since_entry' in event_data.columns:
                event_data['pnl_state'] = event_data['return_since_entry'].apply(
                    lambda x: 'winning' if x > 0.005 else ('losing' if x < -0.005 else 'breakeven')
                )
            else:
                event_data['pnl_state'] = 'breakeven'
        
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
            delta_state = row['delta_confirming_state']
            time_bucket = row['time_bucket']
            pnl_state = row['pnl_state']
            exit_label = row['exit_timing_label']
            weight = weights.loc[idx]
            
            parent_key = (delta_state, time_bucket, pnl_state)
            counts[parent_key][exit_label] += weight
        
        # Convert to probabilities with Laplace smoothing
        cpt = {}
        total_raw = 0
        total_weighted = 0.0
        
        for delta_state in DELTA_STATES:
            for time_bucket in TIME_BUCKETS:
                for pnl_state in PNL_STATES:
                    parent_key = (delta_state, time_bucket, pnl_state)
                    state_counts = counts[parent_key]
                    
                    total = sum(state_counts.values())
                    total_raw += len([c for c in state_counts.values() if c > 0])
                    total_weighted += total
                    
                    # Laplace smoothing
                    probs = {}
                    denominator = total + self.laplace_alpha * len(EXIT_TIMING_STATES)
                    
                    for state in EXIT_TIMING_STATES:
                        count = state_counts.get(state, 0)
                        probs[state] = (count + self.laplace_alpha) / denominator
                    
                    cpt[parent_key] = probs
        
        self._cpt = cpt
        
        # Calculate metrics
        label_counts = event_data['exit_timing_label'].value_counts()
        total_obs = len(event_data)
        
        self._metrics = ExitTimingMetrics(
            observations=total_obs,
            weighted_observations=total_weighted,
            exit_now_pct=label_counts.get('exit_now', 0) / total_obs if total_obs > 0 else 0,
            hold_pct=label_counts.get('hold', 0) / total_obs if total_obs > 0 else 0,
            extend_pct=label_counts.get('extend', 0) / total_obs if total_obs > 0 else 0,
            coverage=len([k for k, v in cpt.items() if sum(v.values()) > len(EXIT_TIMING_STATES) * self.laplace_alpha]) / len(cpt) if cpt else 0,
            avg_entropy=self._calculate_avg_entropy(cpt)
        )
        
        logger.info(f"✅ Exit_Timing CPT generated: {len(cpt)} keys, "
                   f"{total_obs} obs, {total_weighted:.1f} weighted")
        logger.info(f"   Distribution: exit_now={self._metrics.exit_now_pct:.1%}, "
                   f"hold={self._metrics.hold_pct:.1%}, extend={self._metrics.extend_pct:.1%}")
        
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
    
    def _discretize_pnl(self, pnl_atr: float) -> str:
        """Discretiseer PnL in ATR eenheden naar state."""
        if pd.isna(pnl_atr):
            return 'breakeven'
        if pnl_atr < -0.25:  # -0.25 ATR = losing
            return 'losing'
        elif pnl_atr > 0.25:  # +0.25 ATR = winning
            return 'winning'
        else:
            return 'breakeven'
    
    def _calculate_avg_entropy(self, cpt: Dict) -> float:
        """Bereken gemiddelde entropy over CPT entries."""
        entropies = []
        for probs in cpt.values():
            prob_values = list(probs.values())
            entropy = -sum(p * np.log2(p + 1e-10) for p in prob_values if p > 0)
            entropies.append(entropy)
        return np.mean(entropies) if entropies else 0.0
    
    def get_cpt_data(self, asset_id: int) -> Dict[str, Any]:
        """Format CPT voor database opslag."""
        if not self._cpt:
            return {}
        
        return {
            'node_name': 'Exit_Timing',
            'parents': ['Delta_Confirming', 'Time_Since_Entry', 'Current_PnL_ATR'],
            'states': EXIT_TIMING_STATES,
            'conditional_probabilities': {
                f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self._cpt.items()
            },
            'metrics': asdict(self._metrics) if self._metrics else {}
        }
    
    def predict(
        self,
        delta_confirming_state: str,
        time_bucket: str,
        pnl_state: str
    ) -> Dict[str, float]:
        """
        Inference: voorspel exit timing gegeven parent states.
        """
        parent_key = (delta_confirming_state, time_bucket, pnl_state)
        
        if parent_key in self._cpt:
            return self._cpt[parent_key]
        
        logger.warning(f"No CPT entry for {parent_key}, returning uniform")
        return {state: 1/len(EXIT_TIMING_STATES) for state in EXIT_TIMING_STATES}
