"""
Timing Precision Analyzer voor QBN v3.

Berekent metrics om de effectiviteit van de Entry_Confidence node te evalueren,
waaronder win rates, Expected Calibration Error (ECE) en monotoniciteit.

Rationale: Fase 2.3 - Kalibratie van timing-zekerheid op historische outcomes.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimingPrecisionMetrics:
    """Metrics voor Entry_Confidence timing precisie."""
    win_rate_high: float
    win_rate_medium: float
    win_rate_low: float
    avg_return_high: float
    avg_return_medium: float
    avg_return_low: float
    is_monotonic: bool
    expected_calibration_error: float
    observations: int

class TimingPrecisionAnalyzer:
    """
    Analyzer voor Entry_Confidence timing metrics.
    """

    def analyze_from_data(self, data: pd.DataFrame, horizon: str = '1h') -> TimingPrecisionMetrics:
        """
        Bereken alle timing metrics op basis van een DataFrame met predictions en outcomes.
        
        Verwacht columns:
        - entry_confidence: 'low', 'medium', 'high'
        - outcome_{horizon}: -3 tot +3 (legacy) OF barrier_state: 'up_strong', etc.
        - return_{horizon}_pct: raw returns
        """
        outcome_col = 'barrier_state' if 'barrier_state' in data.columns else f'outcome_{horizon}'
        
        # REASON: Check if outcome column actually exists to prevent KeyError
        if outcome_col not in data.columns:
            logger.warning(f"Outcome column {outcome_col} not found in data. Returning default metrics.")
            return TimingPrecisionMetrics(
                win_rate_high=0.5, win_rate_medium=0.5, win_rate_low=0.5,
                avg_return_high=0.0, avg_return_medium=0.0, avg_return_low=0.0,
                is_monotonic=True, expected_calibration_error=0.0, observations=len(data)
            )

        return_col = f'return_{horizon}_pct' if f'return_{horizon}_pct' in data.columns else outcome_col
        
        # Groepeer data
        by_conf = data.groupby('entry_confidence')
        
        def get_stats(conf_level: str) -> Tuple[float, float]:
            if conf_level not in data['entry_confidence'].values:
                return 0.5, 0.0
            group = data[data['entry_confidence'] == conf_level]
            
            # Win rate logic:
            if outcome_col == 'barrier_state':
                # Win = start met 'up_'
                wins = group[outcome_col].str.startswith('up_').sum()
            else:
                # Win rate: outcome > 0 (legacy)
                wins = (group[outcome_col] > 0).sum()
                
            total = len(group)
            win_rate = wins / total if total > 0 else 0.5
            
            # Avg return logic:
            if return_col == 'barrier_state':
                # Map barriers naar pseudo-returns voor gemiddelde (strong=2, weak=1, none=0)
                def map_ret(s):
                    if not isinstance(s, str): return 0.0
                    if 'strong' in s: return 2.0 if 'up' in s else -2.0
                    if 'weak' in s: return 1.0 if 'up' in s else -1.0
                    return 0.0
                avg_ret = group[outcome_col].apply(map_ret).mean()
            else:
                avg_ret = group[return_col].mean()
                
            return win_rate, avg_ret

        wr_high, ret_high = get_stats('high')
        wr_med, ret_med = get_stats('medium')
        wr_low, ret_low = get_stats('low')

        # Monotonicity check
        is_monotonic = (wr_high >= wr_med >= wr_low) or (ret_high >= ret_med >= ret_low)

        # ECE berekening
        ece = self._compute_ece(data, outcome_col)

        return TimingPrecisionMetrics(
            win_rate_high=wr_high,
            win_rate_medium=wr_med,
            win_rate_low=wr_low,
            avg_return_high=ret_high,
            avg_return_medium=ret_med,
            avg_return_low=ret_low,
            is_monotonic=is_monotonic,
            expected_calibration_error=ece,
            observations=len(data)
        )

    def _compute_ece(self, data: pd.DataFrame, outcome_col: str) -> float:
        """
        Expected Calibration Error.
        ECE = Σ (|bin_size| / N) * |predicted_prob - actual_prob|
        """
        # Map confidence naar target win probabilities (voor kalibratie)
        target_probs = {
            'low': 0.45,
            'medium': 0.50,
            'high': 0.55
        }
        
        total_n = len(data)
        ece = 0.0
        
        for level, target_p in target_probs.items():
            subset = data[data['entry_confidence'] == level]
            if subset.empty:
                continue
            
            if outcome_col == 'barrier_state':
                actual_p = subset[outcome_col].str.startswith('up_').sum() / len(subset)
            else:
                actual_p = (subset[outcome_col] > 0).sum() / len(subset)
                
            weight = len(subset) / total_n
            ece += weight * abs(actual_p - target_p)
            
        return float(ece)

