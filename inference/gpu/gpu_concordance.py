"""
GPU-Accelerated Concordance Matrix

Provides GPU-accelerated implementations of multi-timeframe signal concordance
analysis using CuPy for vectorized operations.

Key optimizations:
- Vectorized scenario classification (replaces row-by-row .apply())
- GPU-accelerated concordance score calculation
- Single-pass statistical reductions
- Batch processing for multiple assets

Expected speedup: 20-40x for batch operations
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from config.bayesian_config import SignalState
from config.gpu_config import GPUConfig
from inference.gpu.accelerator import AdaptiveGPUAccelerator
from inference.concordance_matrix import ConcordanceScenario, ConcordanceMatrix

logger = logging.getLogger(__name__)


class GPUConcordanceMatrix:
    """GPU-accelerated multi-timeframe concordance analysis.

    Uses CuPy for vectorized operations to achieve 20-40x speedup over
    CPU-based row-by-row processing when analyzing large batches of signals.

    Automatically falls back to CPU for:
    - Small datasets (< min_size_for_gpu)
    - Systems without GPU
    - GPU memory errors
    
    Timeframe Mapping (KFL):
    - HTF/Structural: Daily (D) - 50%
    - MTF/Tactical: 4H (240) - 25%
    - LTF/Entry: 1H (60) - 15%
    - UTF/Micro: 1m (1) - 10%
    """

    def __init__(
        self,
        structural_weight: float = 0.50,
        tactical_weight: float = 0.25,
        entry_weight: float = 0.15,
        utf_weight: float = 0.10,
        config: Optional[GPUConfig] = None
    ):
        """Initialize GPU Concordance Matrix.

        Args:
            structural_weight: Weight for structural (Daily) signals (50%)
            tactical_weight: Weight for tactical (4H) signals (25%)
            entry_weight: Weight for entry (1H) signals (15%)
            utf_weight: Weight for micro (1m) signals (10%)
            config: GPU configuration (uses default if None)
        """
        self.structural_weight = structural_weight
        self.tactical_weight = tactical_weight
        self.entry_weight = entry_weight
        self.utf_weight = utf_weight

        total_weight = structural_weight + tactical_weight + entry_weight + utf_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Initialize GPU acceleration
        self.config = config or GPUConfig()
        self.config.validate()
        self.accelerator = AdaptiveGPUAccelerator(self.config)

        # CPU fallback for non-vectorizable operations
        self.cpu_matrix = ConcordanceMatrix(
            structural_weight, tactical_weight, entry_weight, utf_weight
        )

        logger.info(
            f"GPU ConcordanceMatrix initialized with weights: "
            f"S={structural_weight}, T={tactical_weight}, E={entry_weight}, U={utf_weight}, "
            f"GPU={'enabled' if self.config.use_gpu else 'disabled'}"
        )

    @property
    def xp(self):
        """Get NumPy-like interface (CuPy if GPU, NumPy if CPU)."""
        return self.accelerator.data_manager.xp

    def classify_signals_dataframe(
        self,
        signals_df: pd.DataFrame,
        use_gpu: bool = True
    ) -> pd.DataFrame:
        """Classify concordance for DataFrame with multi-timeframe signals (GPU-accelerated).

        CRITICAL OPTIMIZATION: Replaces row-by-row .apply() (line 216) with
        vectorized GPU batch operations.

        BEFORE (CPU - SLOW):
            concordance_data = signals_df.apply(classify_row, axis=1)  # Serial

        AFTER (GPU - FAST):
            Vectorized batch classification on entire array

        Args:
            signals_df: DataFrame with htf/mtf/ltf/utf signal columns
            use_gpu: Whether to use GPU acceleration (auto-fallback if fails)

        Returns:
            DataFrame with added concordance_scenario and concordance_score columns
        """
        if signals_df.empty:
            return signals_df

        # Determine signal columns (now returns 4 columns)
        htf_col, mtf_col, ltf_col, utf_col = self._get_signal_columns(signals_df)

        if not htf_col:
            raise ValueError("Missing required signal columns")

        # Extract signal arrays
        htf_signals = signals_df[htf_col].values
        mtf_signals = signals_df[mtf_col].values
        ltf_signals = signals_df[ltf_col].values
        # REASON: UTF optioneel voor backwards compatibility
        utf_signals = signals_df[utf_col].values if utf_col else None

        data_size = len(signals_df)

        # Define GPU and CPU implementations
        def gpu_impl():
            return self._classify_signals_batch_gpu(
                htf_signals, mtf_signals, ltf_signals, utf_signals
            )

        def cpu_impl():
            # Use original CPU implementation
            return self.cpu_matrix.classify_signals_dataframe(signals_df)

        # Execute with adaptive fallback
        if use_gpu:
            result = self.accelerator.execute_with_fallback(
                gpu_impl, cpu_impl, data_size, "concordance_classification"
            )

            # If GPU implementation returned arrays, build DataFrame
            if isinstance(result, tuple):
                scenarios, scores = result
                result_df = signals_df.copy()
                result_df['concordance_scenario'] = scenarios
                result_df['concordance_score'] = scores
                return result_df
            else:
                # CPU fallback returned DataFrame
                return result
        else:
            return cpu_impl()

    def _classify_signals_batch_gpu(
        self,
        htf_signals: np.ndarray,
        mtf_signals: np.ndarray,
        ltf_signals: np.ndarray,
        utf_signals: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated batch signal classification.

        Args:
            htf_signals: Higher timeframe signals array (Daily)
            mtf_signals: Medium timeframe signals array (4H)
            ltf_signals: Lower timeframe signals array (1H)
            utf_signals: Micro timeframe signals array (1m), optional

        Returns:
            Tuple of (scenarios, scores) as numpy arrays
        """
        xp = self.xp

        # REASON: Support zowel 3 als 4 timeframes
        if utf_signals is not None:
            # Transfer to GPU (4 timeframes)
            signals_gpu = xp.asarray(
                np.column_stack([htf_signals, mtf_signals, ltf_signals, utf_signals])
            )  # Shape: (n_samples, 4)
        else:
            # Transfer to GPU (3 timeframes voor backwards compatibility)
            signals_gpu = xp.asarray(
                np.column_stack([htf_signals, mtf_signals, ltf_signals])
            )  # Shape: (n_samples, 3)

        # Vectorized scenario classification
        scenarios_gpu = self._vectorized_classify_scenario(signals_gpu)

        # Vectorized score calculation
        scores_gpu = self._vectorized_concordance_score(signals_gpu)

        # Transfer back to CPU
        scenarios = self.accelerator.data_manager.transfer_to_cpu(scenarios_gpu)
        scores = self.accelerator.data_manager.transfer_to_cpu(scores_gpu)

        # Convert scenario indices to strings
        scenario_names = [s.value for s in ConcordanceScenario]
        scenarios_str = np.array([scenario_names[int(idx)] for idx in scenarios])

        return scenarios_str, scores

    def _vectorized_classify_scenario(self, signals: Any) -> Any:
        """Vectorized scenario classification using GPU.

        CRITICAL OPTIMIZATION: Replaces multiple passes for counting (lines 101-103)
        with single-pass vectorized operations.

        BEFORE (CPU - SLOW):
            bullish_count = sum(1 for s in signals if is_bullish(s))
            bearish_count = sum(1 for s in signals if is_bearish(s))
            neutral_count = sum(1 for s in signals if is_neutral(s))

        AFTER (GPU - FAST):
            Single pass with vectorized counting

        Args:
            signals: 2D array of shape (n_samples, 3 or 4) with HTF, MTF, LTF, [UTF] signals

        Returns:
            1D array of scenario indices (0-7 mapping to ConcordanceScenario)
        """
        xp = self.xp
        
        # REASON: Support zowel 3 als 4 timeframes
        total_signals = signals.shape[1]

        # Count signal types per row (single pass)
        bullish_count = xp.sum(signals > 0, axis=1)  # Shape: (n_samples,)
        bearish_count = xp.sum(signals < 0, axis=1)
        neutral_count = xp.sum(signals == 0, axis=1)

        # Initialize scenarios (default to NEUTRAL = 3)
        scenarios = xp.ones(len(signals), dtype=xp.int8) * 3

        # Strong scenarios (alle of bijna alle gelijk, geen tegengesteld)
        scenarios = xp.where(bullish_count == total_signals, 0, scenarios)  # STRONG_BULLISH
        scenarios = xp.where(bearish_count == total_signals, 6, scenarios)  # STRONG_BEARISH
        scenarios = xp.where(
            (bullish_count >= total_signals - 1) & (bearish_count == 0),
            0,  # STRONG_BULLISH
            scenarios
        )
        scenarios = xp.where(
            (bearish_count >= total_signals - 1) & (bullish_count == 0),
            6,  # STRONG_BEARISH
            scenarios
        )

        # Moderate scenarios (meerderheid + neutrals)
        half_signals = total_signals // 2
        scenarios = xp.where(
            (bullish_count >= half_signals + 1) & (bearish_count == 0),
            1,  # MODERATE_BULLISH
            scenarios
        )
        scenarios = xp.where(
            (bearish_count >= half_signals + 1) & (bullish_count == 0),
            5,  # MODERATE_BEARISH
            scenarios
        )
        scenarios = xp.where(
            (bullish_count == half_signals) & (bearish_count == 0),
            1,  # MODERATE_BULLISH
            scenarios
        )
        scenarios = xp.where(
            (bearish_count == half_signals) & (bullish_count == 0),
            5,  # MODERATE_BEARISH
            scenarios
        )

        # Weak scenarios (meerderheid met tegengestelde)
        scenarios = xp.where(
            (bullish_count > bearish_count) & (bearish_count > 0),
            2,  # WEAK_BULLISH
            scenarios
        )
        scenarios = xp.where(
            (bearish_count > bullish_count) & (bullish_count > 0),
            4,  # WEAK_BEARISH
            scenarios
        )

        # Neutral scenarios (meerderheid neutral)
        scenarios = xp.where(neutral_count >= half_signals + 1, 3, scenarios)  # NEUTRAL

        # Conflicted (gelijke verdeling bullish/bearish)
        scenarios = xp.where(
            (bullish_count == bearish_count) & (bullish_count > 0),
            7,  # CONFLICTED
            scenarios
        )

        return scenarios

    def _vectorized_concordance_score(self, signals: Any) -> Any:
        """Vectorized concordance score calculation using GPU.

        CRITICAL OPTIMIZATION: Single vectorized operation instead of per-row calculation.

        Formula (4 TF): (htf * 50 + mtf * 25 + ltf * 15 + utf * 10 + 200) / 400
        Formula (3 TF): (htf * 60 + mtf * 30 + ltf * 10 + 200) / 400

        Args:
            signals: 2D array of shape (n_samples, 3 or 4) with HTF, MTF, LTF, [UTF] signals

        Returns:
            1D array of concordance scores (0.0 - 1.0)
        """
        xp = self.xp

        # REASON: Support zowel 3 als 4 timeframes
        if signals.shape[1] == 4:
            # Weights as GPU array (4 timeframes: 50/25/15/10)
            weights = xp.array([50, 25, 15, 10], dtype=xp.float64)
        else:
            # Backwards compatibility (3 timeframes: 60/30/10)
            weights = xp.array([60, 30, 10], dtype=xp.float64)

        # Vectorized weighted sum
        raw_scores = xp.dot(signals, weights)  # Shape: (n_samples,)

        # Normalize to [0, 1]: (score + 200) / 400
        # Range is -200 to +200 for both 3 and 4 timeframes (max signal = 2)
        normalized = (raw_scores + 200.0) / 400.0

        return normalized

    def get_concordance_distribution(
        self,
        signals_df: pd.DataFrame,
        use_gpu: bool = True
    ) -> Dict[str, Any]:
        """Analyze concordance distribution in dataset (GPU-accelerated).

        CRITICAL OPTIMIZATION: Replaces multiple DataFrame operations (lines 240-245)
        with single-pass GPU reductions.

        BEFORE (CPU - SLOW):
            signals_df['concordance_score'].mean()
            signals_df['concordance_score'].median()
            signals_df['concordance_score'].std()
            signals_df['concordance_score'].min()
            signals_df['concordance_score'].max()

        AFTER (GPU - FAST):
            Single GPU operation computes all statistics

        Args:
            signals_df: DataFrame with signal data
            use_gpu: Whether to use GPU acceleration

        Returns:
            Dictionary with distribution statistics
        """
        # Classify if not already done
        if 'concordance_scenario' not in signals_df.columns:
            signals_df = self.classify_signals_dataframe(signals_df, use_gpu=use_gpu)

        scenario_counts = signals_df['concordance_scenario'].value_counts()
        total_count = len(signals_df)

        distribution = {}
        for scenario in ConcordanceScenario:
            count = scenario_counts.get(scenario.value, 0)
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            distribution[scenario.value] = {
                'count': int(count),
                'percentage': float(percentage)
            }

        # GPU-accelerated statistics
        if 'concordance_score' in signals_df.columns:
            if use_gpu and len(signals_df) >= self.config.min_size_for_gpu:
                score_stats = self._calculate_statistics_gpu(
                    signals_df['concordance_score'].values
                )
            else:
                # CPU fallback for small datasets
                score_stats = {
                    'mean_score': float(signals_df['concordance_score'].mean()),
                    'median_score': float(signals_df['concordance_score'].median()),
                    'std_score': float(signals_df['concordance_score'].std()),
                    'min_score': float(signals_df['concordance_score'].min()),
                    'max_score': float(signals_df['concordance_score'].max())
                }
        else:
            score_stats = {}

        return {
            'total_signals': total_count,
            'scenario_distribution': distribution,
            'score_statistics': score_stats,
            'weights': {
                'structural': self.structural_weight,
                'tactical': self.tactical_weight,
                'entry': self.entry_weight,
                'utf': self.utf_weight
            }
        }

    def _calculate_statistics_gpu(self, scores: np.ndarray) -> Dict[str, float]:
        """Calculate statistics using GPU (single pass).

        Args:
            scores: Array of concordance scores

        Returns:
            Dictionary with mean, median, std, min, max
        """
        xp = self.xp

        # Transfer to GPU
        scores_gpu = xp.asarray(scores)

        # Single-pass statistics (all GPU operations)
        stats = {
            'mean_score': float(xp.mean(scores_gpu)),
            'median_score': float(xp.median(scores_gpu)),
            'std_score': float(xp.std(scores_gpu)),
            'min_score': float(xp.min(scores_gpu)),
            'max_score': float(xp.max(scores_gpu))
        }

        return stats

    def classify_scenario(
        self,
        htf_signal: SignalState,
        mtf_signal: SignalState,
        ltf_signal: SignalState,
        utf_signal: Optional[SignalState] = None
    ) -> ConcordanceScenario:
        """Classify single concordance scenario (delegates to CPU).

        For single-sample classification, CPU is more efficient.

        Args:
            htf_signal: Higher timeframe signal (Daily)
            mtf_signal: Medium timeframe signal (4H)
            ltf_signal: Lower timeframe signal (1H)
            utf_signal: Micro timeframe signal (1m), optional

        Returns:
            ConcordanceScenario enum
        """
        return self.cpu_matrix.classify_scenario(htf_signal, mtf_signal, ltf_signal, utf_signal)

    def calculate_concordance_score(
        self,
        htf_signal: SignalState,
        mtf_signal: SignalState,
        ltf_signal: SignalState,
        utf_signal: Optional[SignalState] = None
    ) -> float:
        """Calculate single concordance score (delegates to CPU).

        For single-sample calculation, CPU is more efficient.

        Args:
            htf_signal: Higher timeframe signal (Daily)
            mtf_signal: Medium timeframe signal (4H)
            ltf_signal: Lower timeframe signal (1H)
            utf_signal: Micro timeframe signal (1m), optional

        Returns:
            Concordance score (0.0 - 1.0)
        """
        return self.cpu_matrix.calculate_concordance_score(
            htf_signal, mtf_signal, ltf_signal, utf_signal
        )

    def create_evidence_node_data(
        self,
        htf_signal: SignalState,
        mtf_signal: SignalState,
        ltf_signal: SignalState,
        utf_signal: Optional[SignalState] = None
    ) -> Dict[str, Any]:
        """Create evidence node data for Bayesian network (delegates to CPU).

        Args:
            htf_signal: Higher timeframe signal (Daily)
            mtf_signal: Medium timeframe signal (4H)
            ltf_signal: Lower timeframe signal (1H)
            utf_signal: Micro timeframe signal (1m), optional

        Returns:
            Dictionary with evidence data
        """
        return self.cpu_matrix.create_evidence_node_data(
            htf_signal, mtf_signal, ltf_signal, utf_signal
        )

    def update_weights(self, structural: float, tactical: float, entry: float, utf: float = 0.0):
        """Update concordance weights.

        Args:
            structural: Structural weight (Daily)
            tactical: Tactical weight (4H)
            entry: Entry weight (1H)
            utf: Micro weight (1m)
        """
        total = structural + tactical + entry + utf
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.structural_weight = structural
        self.tactical_weight = tactical
        self.entry_weight = entry
        self.utf_weight = utf

        # Update CPU fallback weights
        self.cpu_matrix.update_weights(structural, tactical, entry, utf)

        logger.info(f"Updated weights: S={structural}, T={tactical}, E={entry}, U={utf}")

    def _get_signal_columns(self, signals_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Determine signal column names from DataFrame.

        Args:
            signals_df: DataFrame with signal data

        Returns:
            Tuple of (htf_col, mtf_col, ltf_col, utf_col) - utf_col can be None
        """
        if 'htf_signal_state' in signals_df.columns:
            utf_col = 'utf_signal_state' if 'utf_signal_state' in signals_df.columns else None
            return 'htf_signal_state', 'mtf_signal_state', 'ltf_signal_state', utf_col
        elif 'rsi_signal_d' in signals_df.columns:
            # Use RSI as representative signal
            utf_col = 'rsi_signal_1' if 'rsi_signal_1' in signals_df.columns else None
            return 'rsi_signal_d', 'rsi_signal_240', 'rsi_signal_60', utf_col
        else:
            return None, None, None, None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get GPU performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        return self.accelerator.get_performance_stats()


def create_gpu_concordance_matrix(
    structural_weight: float = 0.50,
    tactical_weight: float = 0.25,
    entry_weight: float = 0.15,
    utf_weight: float = 0.10,
    config: Optional[GPUConfig] = None
) -> GPUConcordanceMatrix:
    """Factory function for GPU ConcordanceMatrix with 4 timeframes.

    Args:
        structural_weight: Weight for structural signals (Daily, 50%)
        tactical_weight: Weight for tactical signals (4H, 25%)
        entry_weight: Weight for entry signals (1H, 15%)
        utf_weight: Weight for micro signals (1m, 10%)
        config: GPU configuration

    Returns:
        GPUConcordanceMatrix instance
    """
    return GPUConcordanceMatrix(
        structural_weight, tactical_weight, entry_weight, utf_weight, config
    )
