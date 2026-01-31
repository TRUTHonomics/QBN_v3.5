"""
GPU-Accelerated Inference Engine for QBN v3.4.

Implements vectorized batch inference for the v3.4 architecture:
- HTF_Regime batch detection (numerical)
- Composite aggregation batch (matrix ops)
- CPT lookup batch (vectorized indexing)
- Position-side inference: Momentum, Volatility, Exit, Position Prediction

Optimized for CuPy: Uses integer-based states internally to maximize GPU throughput.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    import numpy as cp # Fallback to numpy if cupy is missing

from inference.node_types import CompositeState, OutcomeState, RegimeState, SemanticClass, BarrierOutcomeState
from inference.gpu.accelerator import AdaptiveGPUAccelerator
from inference.state_reduction import REDUCED_REGIME_STATES, FULL_TO_REDUCED_REGIME_MAP
from config.gpu_config import GPUConfig
from core.config_defaults import (
    DEFAULT_COMPOSITE_NEUTRAL_BAND, 
    DEFAULT_COMPOSITE_STRONG_THRESHOLD,
    DEFAULT_DELTA_THRESHOLD_LEADING,
    DEFAULT_DELTA_THRESHOLD_COINCIDENT,
    DEFAULT_DELTA_THRESHOLD_CONFIRMING
)
from core.config_warnings import warn_fallback_active

logger = logging.getLogger(__name__)

class GPUInferenceEngine:
    """
    Vectorized Inference Engine using CuPy for GPU acceleration.
    Operates on integer-encoded states internally for maximum speed.
    
    v3.4: Includes position-side inference (Momentum, Volatility, Exit, Position Prediction).
    """

    # --- INTEGER STATE MAPPINGS ---
    # Regime: ["sync_strong_bearish", "sync_bearish", "macro_ranging", "sync_bullish", "sync_strong_bullish"]
    REGIME_MAP = {s: i for i, s in enumerate(REDUCED_REGIME_STATES)}
    REGIME_NAMES = np.array(REDUCED_REGIME_STATES)

    # Composite: Strong_Bearish(0) -> Bearish(1) -> Neutral(2) -> Bullish(3) -> Strong_Bullish(4)
    # Order matches the threshold logic (-st, -nb, nb, st)
    COMPOSITE_STATES = [
        CompositeState.STRONG_BEARISH.value,
        CompositeState.BEARISH.value,
        CompositeState.NEUTRAL.value,
        CompositeState.BULLISH.value,
        CompositeState.STRONG_BULLISH.value
    ]
    COMPOSITE_NAMES = np.array(COMPOSITE_STATES)

    # Hypothesis: Strong_Short(0) -> Weak_Short(1) -> No_Setup(2) -> Weak_Long(3) -> Strong_Long(4)
    # Matches Composite mapping 1:1
    HYPOTHESIS_STATES = ["strong_short", "weak_short", "no_setup", "weak_long", "strong_long"]
    HYPOTHESIS_MAP = {s: i for i, s in enumerate(HYPOTHESIS_STATES)}
    HYPOTHESIS_NAMES = np.array(HYPOTHESIS_STATES)
    
    # --- POSITION-SIDE STATE MAPPINGS (v3.4) ---
    # Delta states: deteriorating(0), stable(1), improving(2)
    DELTA_STATES = ["deteriorating", "stable", "improving"]
    DELTA_MAP = {s: i for i, s in enumerate(DELTA_STATES)}
    DELTA_NAMES = np.array(DELTA_STATES)
    
    # Time buckets: 0-1h(0), 1-4h(1), 4-12h(2), 12-24h(3)
    TIME_BUCKETS = ["0-1h", "1-4h", "4-12h", "12-24h"]
    TIME_MAP = {s: i for i, s in enumerate(TIME_BUCKETS)}
    TIME_NAMES = np.array(TIME_BUCKETS)
    
    # PnL states: losing(0), breakeven(1), winning(2)
    PNL_STATES = ["losing", "breakeven", "winning"]
    PNL_MAP = {s: i for i, s in enumerate(PNL_STATES)}
    PNL_NAMES = np.array(PNL_STATES)
    
    # Momentum states: bearish(0), neutral(1), bullish(2)
    MOMENTUM_STATES = ["bearish", "neutral", "bullish"]
    MOMENTUM_MAP = {s: i for i, s in enumerate(MOMENTUM_STATES)}
    MOMENTUM_NAMES = np.array(MOMENTUM_STATES)
    
    # Volatility states: low_vol(0), normal(1), high_vol(2)
    VOLATILITY_STATES = ["low_vol", "normal", "high_vol"]
    VOLATILITY_MAP = {s: i for i, s in enumerate(VOLATILITY_STATES)}
    VOLATILITY_NAMES = np.array(VOLATILITY_STATES)
    
    # Exit timing states: exit_now(0), hold(1), extend(2)
    EXIT_TIMING_STATES = ["exit_now", "hold", "extend"]
    EXIT_TIMING_MAP = {s: i for i, s in enumerate(EXIT_TIMING_STATES)}
    EXIT_TIMING_NAMES = np.array(EXIT_TIMING_STATES)
    
    # Position prediction states: target_hit(0), stoploss_hit(1), timeout(2)
    POSITION_STATES = ["target_hit", "stoploss_hit", "timeout"]
    POSITION_MAP = {s: i for i, s in enumerate(POSITION_STATES)}
    POSITION_NAMES = np.array(POSITION_STATES)
    
    # Position confidence (derived from Position_Prediction distribution)
    CONFIDENCE_STATES = ["low", "medium", "high"]
    CONFIDENCE_MAP = {s: i for i, s in enumerate(CONFIDENCE_STATES)}
    CONFIDENCE_NAMES = np.array(CONFIDENCE_STATES)

    def __init__(
        self, 
        cpts: Dict[str, Dict[str, Any]], 
        signal_classification: Dict[str, Dict],
        threshold_loader: Optional[Any] = None,
        config: Optional[GPUConfig] = None
    ):
        self.cpts = cpts
        self.signal_classification = signal_classification
        self._threshold_loader = threshold_loader
        self.config = config or GPUConfig()
        self.accelerator = AdaptiveGPUAccelerator(self.config)
        
        # Pre-process signal weights for faster matrix ops
        self._prepare_weights()
        
        # Pre-process CPTs into GPU arrays for fast indexing
        self._prepare_cpt_matrices()

    def _prepare_weights(self):
        """Build matrix of weights for vectorized composite calculation."""
        self.class_signals = {
            SemanticClass.LEADING: [],
            SemanticClass.COINCIDENT: [],
            SemanticClass.CONFIRMING: []
        }
        
        # Pre-compiled metadata for each class
        self.class_metadata = {}
        
        for sem_class in SemanticClass:
            signals = []
            polarities = []
            weights_1h = []
            
            for name, info in self.signal_classification.items():
                if info.get('semantic_class') == sem_class.value:
                    signals.append(name)
                    
                    # Map polarity string/int to float
                    p = info.get('polarity', 1)
                    if isinstance(p, str):
                        p = 1 if p.lower() == 'bullish' else -1 if p.lower() == 'bearish' else 0
                    polarities.append(float(p or 1))
                    
                    # Get 1h weight
                    w = info.get('weights', {}).get('1h', info.get('weight', 1.0))
                    weights_1h.append(float(w or 1.0))
            
            # Load thresholds for this class
            nb = DEFAULT_COMPOSITE_NEUTRAL_BAND
            st = DEFAULT_COMPOSITE_STRONG_THRESHOLD
            
            if self._threshold_loader:
                try:
                    params = self._threshold_loader.get_composite_params(sem_class.value)
                    nb = params.get('neutral_band', DEFAULT_COMPOSITE_NEUTRAL_BAND)
                    st = params.get('strong_threshold', DEFAULT_COMPOSITE_STRONG_THRESHOLD)
                except Exception as e:
                    warn_fallback_active(
                        component="GPUInferenceEngine",
                        config_name=f"thresholds_{sem_class.value}",
                        fallback_values={'nb': nb, 'st': st},
                        reason=f"Fout bij laden thresholds: {e}"
                    )
            else:
                warn_fallback_active(
                    component="GPUInferenceEngine",
                    config_name=f"thresholds_{sem_class.value}",
                    fallback_values={'nb': nb, 'st': st},
                    reason="Geen ThresholdLoader aanwezig"
                )

            self.class_signals[sem_class] = signals
            
            # Convert to GPU array if available
            vec = np.array(polarities) * np.array(weights_1h)
            if CUPY_AVAILABLE:
                vec = cp.asarray(vec)
                
            self.class_metadata[sem_class] = {
                'signals': signals,
                'vector': vec,
                'total_weight': sum(weights_1h),
                'thresholds': {'nb': nb, 'st': st}
            }

    def _prepare_cpt_matrices(self):
        """Convert dictionary-based CPTs into indexed GPU matrices."""
        self.cpt_matrices = {}
        self.cpt_outcome_names = {}
        
        # --- Entry-side CPTs (Prediction_1h/4h/1d) ---
        for horizon in ['1h', '4h', '1d']:
            node_name = f"Prediction_{horizon}"
            cpt = self.cpts.get(node_name)
            if not cpt: continue
            
            # Use states from CPT or default to BarrierOutcomeState
            outcome_states = cpt.get('states', BarrierOutcomeState.state_names())
            self.cpt_outcome_names[node_name] = np.array(outcome_states)
            
            n_r = len(self.REGIME_NAMES)
            n_h = len(self.HYPOTHESIS_NAMES)
            n_o = len(outcome_states)
            
            # Default to prior if available, else uniform
            prior = cpt.get('probabilities', {s: 1.0/n_o for s in outcome_states})
            prior_vec = np.array([prior.get(s, 1.0/n_o) for s in outcome_states])
            
            # Matrix shape: (n_regimes, n_hypotheses, n_outcomes)
            matrix = np.tile(prior_vec, (n_r, n_h, 1))
            
            # Track which cells were filled (for aggregation of multiple 11-state keys to 1 reduced key)
            fill_counts = np.zeros((n_r, n_h), dtype=int)
            sum_matrix = np.zeros((n_r, n_h, n_o))
            
            # Fill with conditional probabilities
            # REASON: CPT keys may use full 11-state regime names, map to reduced 5-state
            cond_probs = cpt.get('conditional_probabilities', {})
            for key, dist in cond_probs.items():
                parts = key.split('|')
                if len(parts) != 2: continue
                
                r_state, h_state = parts
                
                # Map full regime state to reduced state
                r_state_reduced = FULL_TO_REDUCED_REGIME_MAP.get(r_state, r_state)
                
                # Map strings to indices
                r_i = self.REGIME_MAP.get(r_state_reduced)
                h_i = self.HYPOTHESIS_MAP.get(h_state)
                
                if r_i is not None and h_i is not None:
                    # Aggregate distributions from multiple full states to one reduced state
                    probs = np.array([dist.get(s, 0.0) for s in outcome_states])
                    sum_matrix[r_i, h_i] += probs
                    fill_counts[r_i, h_i] += 1
            
            # Average where we have multiple mappings
            for r_i in range(n_r):
                for h_i in range(n_h):
                    if fill_counts[r_i, h_i] > 0:
                        matrix[r_i, h_i] = sum_matrix[r_i, h_i] / fill_counts[r_i, h_i]
            
            # Transfer to GPU
            if CUPY_AVAILABLE:
                self.cpt_matrices[node_name] = cp.asarray(matrix)
            else:
                self.cpt_matrices[node_name] = matrix
                
            # Log coverage
            filled = np.sum(fill_counts > 0)
            total = n_r * n_h
            logger.debug(f"CPT Matrix {node_name}: {filled}/{total} cells filled ({100*filled/total:.1f}%)")
        
        # --- Position-side CPTs (v3.4) ---
        self._prepare_position_cpt_matrices()
    
    def _prepare_position_cpt_matrices(self):
        """Prepare CPT matrices for position-side nodes (v3.4)."""
        
        # 1. Momentum_Prediction: Parents = (Delta_Leading, Time_Since_Entry)
        # Shape: (3, 4, 3) = (delta_states, time_buckets, momentum_states)
        self._prepare_2parent_cpt(
            node_name='Momentum_Prediction',
            parent1_states=self.DELTA_STATES,
            parent2_states=self.TIME_BUCKETS,
            outcome_states=self.MOMENTUM_STATES
        )
        
        # 2. Volatility_Regime: Parents = (Delta_Coincident, Time_Since_Entry)
        # Shape: (3, 4, 3) = (delta_states, time_buckets, volatility_states)
        self._prepare_2parent_cpt(
            node_name='Volatility_Regime',
            parent1_states=self.DELTA_STATES,
            parent2_states=self.TIME_BUCKETS,
            outcome_states=self.VOLATILITY_STATES
        )
        
        # 3. Exit_Timing: Parents = (Delta_Confirming, Time_Since_Entry, Current_PnL_ATR)
        # Shape: (3, 4, 3, 3) = (delta_states, time_buckets, pnl_states, exit_states)
        self._prepare_3parent_cpt(
            node_name='Exit_Timing',
            parent1_states=self.DELTA_STATES,
            parent2_states=self.TIME_BUCKETS,
            parent3_states=self.PNL_STATES,
            outcome_states=self.EXIT_TIMING_STATES
        )
        
        # 4. Position_Prediction: Parents = (Momentum, Volatility, Exit_Timing)
        # Shape: (3, 3, 3, 3) = (momentum_states, volatility_states, exit_states, position_states)
        self._prepare_3parent_cpt(
            node_name='Position_Prediction',
            parent1_states=self.MOMENTUM_STATES,
            parent2_states=self.VOLATILITY_STATES,
            parent3_states=self.EXIT_TIMING_STATES,
            outcome_states=self.POSITION_STATES
        )
    
    def _prepare_2parent_cpt(
        self, 
        node_name: str,
        parent1_states: List[str],
        parent2_states: List[str],
        outcome_states: List[str]
    ):
        """Prepare CPT matrix for a node with 2 parents."""
        cpt = self.cpts.get(node_name)
        
        n_p1 = len(parent1_states)
        n_p2 = len(parent2_states)
        n_o = len(outcome_states)
        
        # Default uniform distribution
        uniform = 1.0 / n_o
        matrix = np.full((n_p1, n_p2, n_o), uniform)
        
        if cpt:
            cond_probs = cpt.get('conditional_probabilities', {})
            for key, dist in cond_probs.items():
                parts = key.split('|')
                if len(parts) != 2: continue
                
                p1_state, p2_state = parts
                p1_i = parent1_states.index(p1_state) if p1_state in parent1_states else None
                p2_i = parent2_states.index(p2_state) if p2_state in parent2_states else None
                
                if p1_i is not None and p2_i is not None:
                    probs = np.array([dist.get(s, uniform) for s in outcome_states])
                    matrix[p1_i, p2_i] = probs
            
            logger.debug(f"CPT Matrix {node_name}: loaded from CPT")
        else:
            logger.warning(f"CPT Matrix {node_name}: no CPT found, using uniform distribution")
        
        self.cpt_outcome_names[node_name] = np.array(outcome_states)
        
        if CUPY_AVAILABLE:
            self.cpt_matrices[node_name] = cp.asarray(matrix)
        else:
            self.cpt_matrices[node_name] = matrix
    
    def _prepare_3parent_cpt(
        self, 
        node_name: str,
        parent1_states: List[str],
        parent2_states: List[str],
        parent3_states: List[str],
        outcome_states: List[str]
    ):
        """Prepare CPT matrix for a node with 3 parents."""
        cpt = self.cpts.get(node_name)
        
        n_p1 = len(parent1_states)
        n_p2 = len(parent2_states)
        n_p3 = len(parent3_states)
        n_o = len(outcome_states)
        
        # Default uniform distribution
        uniform = 1.0 / n_o
        matrix = np.full((n_p1, n_p2, n_p3, n_o), uniform)
        
        if cpt:
            cond_probs = cpt.get('conditional_probabilities', {})
            for key, dist in cond_probs.items():
                parts = key.split('|')
                if len(parts) != 3: continue
                
                p1_state, p2_state, p3_state = parts
                p1_i = parent1_states.index(p1_state) if p1_state in parent1_states else None
                p2_i = parent2_states.index(p2_state) if p2_state in parent2_states else None
                p3_i = parent3_states.index(p3_state) if p3_state in parent3_states else None
                
                if p1_i is not None and p2_i is not None and p3_i is not None:
                    probs = np.array([dist.get(s, uniform) for s in outcome_states])
                    matrix[p1_i, p2_i, p3_i] = probs
            
            logger.debug(f"CPT Matrix {node_name}: loaded from CPT")
        else:
            logger.warning(f"CPT Matrix {node_name}: no CPT found, using uniform distribution")
        
        self.cpt_outcome_names[node_name] = np.array(outcome_states)
        
        if CUPY_AVAILABLE:
            self.cpt_matrices[node_name] = cp.asarray(matrix)
        else:
            self.cpt_matrices[node_name] = matrix

    def infer_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run inference on a batch of data.
        
        Returns: Dict with CPU-compatible numpy arrays/lists for downstream usage.
        Includes 'raw_composite_scores' for position-side inference.
        """
        start_time = time.perf_counter()
        n_rows = len(df)
        
        # 1. Regime Detection (Returns Integer Indices on GPU)
        # 0=SSB, 1=SB, 2=Range, 3=SB, 4=SSB
        regime_indices = self._compute_regime_indices(df)
        
        # 2. Composites Aggregation (Returns Integer Indices + Raw Scores)
        # 0=StrongBearish ... 4=StrongBullish
        composite_indices, raw_composite_scores = self._compute_composite_indices(df)
        
        # 3. Trade Hypothesis derivation (Returns Integer Indices on GPU)
        # Since mapping is 1:1 with Leading Composite, we just reuse the index!
        # Leading: StrongBearish(0) -> Hypothesis: StrongShort(0)
        hypothesis_indices = composite_indices[SemanticClass.LEADING]
        
        # 4. Predictions (Returns Dict with results)
        predictions = {}
        for horizon in ['1h', '4h', '1d']:
            predictions[horizon] = self._compute_prediction_batch(horizon, regime_indices, hypothesis_indices)
            
        elapsed = (time.perf_counter() - start_time) * 1000
        
        # --- Convert GPU indices back to CPU strings for output ---
        
        # Helpers for conversion
        def to_cpu_strings(indices_gpu, names_array):
            if CUPY_AVAILABLE:
                idx = cp.asnumpy(indices_gpu)
            else:
                idx = indices_gpu
            return names_array[idx]

        regimes_str = to_cpu_strings(regime_indices, self.REGIME_NAMES)
        hypo_str = to_cpu_strings(hypothesis_indices, self.HYPOTHESIS_NAMES)
        
        composites_str = {}
        for sc, indices in composite_indices.items():
            composites_str[sc] = to_cpu_strings(indices, self.COMPOSITE_NAMES)

        logger.info(f"🚀 Vectorized GPU Batch inference: {n_rows} rows in {elapsed:.2f}ms")
        
        return {
            'regime': regimes_str,
            'leading_composite': composites_str[SemanticClass.LEADING],
            'coincident_composite': composites_str[SemanticClass.COINCIDENT],
            'confirming_composite': composites_str[SemanticClass.CONFIRMING],
            'trade_hypothesis': hypo_str,
            'predictions': predictions,
            'inference_time_ms': elapsed,
            # v3.4: Raw composite scores for position-side inference (delta calculation)
            'raw_composite_scores': {
                'leading': raw_composite_scores[SemanticClass.LEADING],
                'coincident': raw_composite_scores[SemanticClass.COINCIDENT],
                'confirming': raw_composite_scores[SemanticClass.CONFIRMING],
            }
        }

    def _compute_regime_indices(self, df: pd.DataFrame):
        """
        Compute reduced regime indices directly.
        Returns GPU array of int8 indices matching REDUCED_REGIME_STATES order:
        0: full_bearish
        1: bearish_transition
        2: macro_ranging (Default)
        3: bullish_transition
        4: full_bullish
        """
        # Load data to GPU
        adx_d = cp.asarray(df['adx_signal_d'].values if 'adx_signal_d' in df.columns else np.zeros(len(df)))
        adx_240 = cp.asarray(df['adx_signal_240'].values if 'adx_signal_240' in df.columns else np.zeros(len(df)))
        
        n = len(adx_d)
        # Start with default = 2 (macro_ranging)
        indices = cp.full(n, 2, dtype=cp.int8)
        
        # Boolean masks for directions
        bull_d = adx_d > 0
        bear_d = adx_d < 0
        range_d = adx_d == 0
        
        bull_240 = adx_240 > 0
        bear_240 = adx_240 < 0
        range_240 = adx_240 == 0
        
        # --- BEARISH REGIMES ---
        # sync_strong_bearish (Full Bearish): d < 0 AND 240 < 0 -> Index 0
        sync_strong_bear = bear_d & bear_240
        indices[sync_strong_bear] = 0
        
        # sync_bearish (Bearish Transition): d < 0 but NOT sync (240 >= 0 or 240 == 0) -> Index 1
        # Also includes bearish_emerging (d == 0 but 240 < 0)
        sync_bear = (bear_d & ~bear_240) | (range_d & bear_240)
        indices[sync_bear] = 1
        
        # --- BULLISH REGIMES ---
        # sync_strong_bullish (Full Bullish): d > 0 AND 240 > 0 -> Index 4
        sync_strong_bull = bull_d & bull_240
        indices[sync_strong_bull] = 4
        
        # sync_bullish (Bullish Transition): d > 0 but NOT sync (240 <= 0 or 240 == 0) -> Index 3
        # Also includes bullish_emerging (d == 0 but 240 > 0)
        sync_bull = (bull_d & ~bull_240) | (range_d & bull_240)
        indices[sync_bull] = 3
        
        return indices

    def _compute_composite_indices(self, df: pd.DataFrame) -> Tuple[Dict[SemanticClass, Any], Dict[SemanticClass, Any]]:
        """
        Compute composite state indices via matrix mult.
        
        Returns:
            Tuple of:
            - indices: dict of GPU arrays (int8) with state indices
            - raw_scores: dict of CPU numpy arrays with normalized scores (for delta calculation)
        
        State indices:
        0: Strong Bearish
        1: Bearish
        2: Neutral
        3: Bullish
        4: Strong Bullish
        """
        results_indices = {}
        results_scores = {}
        
        for sem_class, meta in self.class_metadata.items():
            signals = meta['signals']
            if not signals:
                results_indices[sem_class] = cp.full(len(df), 2, dtype=cp.int8) # Default Neutral (2)
                results_scores[sem_class] = np.zeros(len(df), dtype=np.float64)
                continue
                
            # Extract signal matrix (CPU -> GPU)
            valid_signals = [s for s in signals if s in df.columns]
            if not valid_signals:
                results_indices[sem_class] = cp.full(len(df), 2, dtype=cp.int8)
                results_scores[sem_class] = np.zeros(len(df), dtype=np.float64)
                continue
                
            # Values -2 to 2
            sig_data_cpu = df[valid_signals].fillna(0).values.astype(np.float64)
            
            # Weight vector (get as numpy for CPU computation)
            indices = [signals.index(s) for s in valid_signals]
            weight_vec = meta['vector']
            if hasattr(weight_vec, 'get'):  # CuPy array
                weight_vec = weight_vec.get()
            weight_vec_cpu = np.asarray(weight_vec)[indices].astype(np.float64)
            
            # REASON: Compute on CPU with numpy to avoid CUBLAS_STATUS_INVALID_VALUE
            # This is a small operation and more reliable than GPU for edge cases
            scores_cpu = sig_data_cpu @ weight_vec_cpu
            scores_normalized = scores_cpu.copy()
            
            # Normalize
            norm_factor = meta['total_weight'] * 2.0
            if norm_factor > 0:
                scores_normalized /= norm_factor
            
            # Store raw normalized scores (CPU) for delta calculation
            results_scores[sem_class] = scores_normalized
            
            # Transfer to GPU for thresholding
            scores = cp.asarray(scores_normalized)
                
            # Thresholding to Indices [0..4]
            nb = meta['thresholds']['nb']
            st = meta['thresholds']['st']
            
            # Default Neutral (2)
            state_indices = cp.full(len(df), 2, dtype=cp.int8)
            
            # Strong Bearish (0): < -st
            state_indices[scores < -st] = 0
            
            # Bearish (1): >= -st AND < -nb
            state_indices[(scores >= -st) & (scores < -nb)] = 1
            
            # Bullish (3): > nb AND <= st
            state_indices[(scores > nb) & (scores <= st)] = 3
            
            # Strong Bullish (4): > st
            state_indices[scores > st] = 4
            
            results_indices[sem_class] = state_indices
            
        return results_indices, results_scores

    def _compute_prediction_batch(self, horizon: str, r_indices: Any, h_indices: Any) -> Dict[str, Any]:
        """
        Compute predictions via direct GPU indexing.
        """
        node_name = f"Prediction_{horizon}"
        matrix = self.cpt_matrices.get(node_name)
        outcome_names = self.cpt_outcome_names.get(node_name)
        
        if matrix is None or outcome_names is None:
            # Fallback
            n_o = 5
            dist = {s: 1.0/n_o for s in BarrierOutcomeState.state_names()}
            return {
                'states': np.full(len(r_indices), "neutral", dtype=object),
                'distributions': [dist] * len(r_indices)
            }
            
        # GPU Indexing: matrix[regime_idx, hypo_idx] -> (N, n_outcomes)
        # r_indices and h_indices are already on GPU (if CuPy enabled)
        dists_gpu = matrix[r_indices, h_indices]
        
        # Best state index: argmax along last axis
        best_idx_gpu = cp.argmax(dists_gpu, axis=1)
        
        # --- Transfer results to CPU ---
        if CUPY_AVAILABLE:
            dists_cpu = cp.asnumpy(dists_gpu)
            best_idx_cpu = cp.asnumpy(best_idx_gpu)
        else:
            dists_cpu = dists_gpu
            best_idx_cpu = best_idx_gpu
            
        # Map indices to names
        best_states = outcome_names[best_idx_cpu]
        
        # Build list of dicts for distributions (expensive part, but necessary for compatibility)
        # Only done once at the end.
        n_outcomes = len(outcome_names)
        dist_dicts = []
        
        # Optimized list comprehension
        for row in dists_cpu:
            dist_dicts.append(dict(zip(outcome_names, row)))
            
        return {
            'states': best_states,
            'distributions': dist_dicts
        }
    
    # ========================================================================
    # POSITION-SIDE INFERENCE (v3.4)
    # ========================================================================
    
    def infer_position(
        self,
        current_scores: Dict[str, float],
        entry_scores: Dict[str, float],
        time_since_entry_min: float,
        current_pnl_atr: float
    ) -> Dict[str, Any]:
        """
        Compute position-side predictions for an active trade.
        
        Args:
            current_scores: Current raw composite scores {'leading': float, 'coincident': float, 'confirming': float}
            entry_scores: Entry raw composite scores {'leading': float, 'coincident': float, 'confirming': float}
            time_since_entry_min: Minutes since trade entry
            current_pnl_atr: Current PnL in ATR multiples
            
        Returns:
            Dict with:
            - momentum_prediction: str (bearish/neutral/bullish)
            - volatility_regime: str (low_vol/normal/high_vol)
            - exit_timing: str (exit_now/hold/extend)
            - position_prediction: str (target_hit/stoploss_hit/timeout)
            - position_confidence: str (low/medium/high)
            - distributions: Dict with probability distributions for each node
        """
        # 1. Calculate deltas (current - entry)
        delta_leading = current_scores['leading'] - entry_scores['leading']
        delta_coincident = current_scores['coincident'] - entry_scores['coincident']
        delta_confirming = current_scores['confirming'] - entry_scores['confirming']
        
        # 2. Discretize inputs
        delta_leading_idx = self._discretize_delta(delta_leading, DEFAULT_DELTA_THRESHOLD_LEADING)
        delta_coincident_idx = self._discretize_delta(delta_coincident, DEFAULT_DELTA_THRESHOLD_COINCIDENT)
        delta_confirming_idx = self._discretize_delta(delta_confirming, DEFAULT_DELTA_THRESHOLD_CONFIRMING)
        time_idx = self._discretize_time(time_since_entry_min)
        pnl_idx = self._discretize_pnl(current_pnl_atr)
        
        # 3. Momentum_Prediction: P(M | Delta_Leading, Time)
        mp_dist = self._lookup_2parent_cpt('Momentum_Prediction', delta_leading_idx, time_idx)
        mp_idx = int(np.argmax(mp_dist))
        momentum_prediction = self.MOMENTUM_NAMES[mp_idx]
        
        # 4. Volatility_Regime: P(V | Delta_Coincident, Time)
        vr_dist = self._lookup_2parent_cpt('Volatility_Regime', delta_coincident_idx, time_idx)
        vr_idx = int(np.argmax(vr_dist))
        volatility_regime = self.VOLATILITY_NAMES[vr_idx]
        
        # 5. Exit_Timing: P(E | Delta_Confirming, Time, PnL)
        et_dist = self._lookup_3parent_cpt('Exit_Timing', delta_confirming_idx, time_idx, pnl_idx)
        et_idx = int(np.argmax(et_dist))
        exit_timing = self.EXIT_TIMING_NAMES[et_idx]
        
        # 6. Position_Prediction: P(PP | MP, VR, ET)
        pp_dist = self._lookup_3parent_cpt('Position_Prediction', mp_idx, vr_idx, et_idx)
        pp_idx = int(np.argmax(pp_dist))
        position_prediction = self.POSITION_NAMES[pp_idx]
        
        # 7. Derive Position Confidence from PP distribution
        # High confidence: dominant outcome > 0.5
        # Medium: 0.4 < dominant < 0.5
        # Low: dominant < 0.4
        max_prob = float(np.max(pp_dist))
        if max_prob >= 0.5:
            position_confidence = "high"
        elif max_prob >= 0.4:
            position_confidence = "medium"
        else:
            position_confidence = "low"
        
        return {
            'momentum_prediction': momentum_prediction,
            'volatility_regime': volatility_regime,
            'exit_timing': exit_timing,
            'position_prediction': position_prediction,
            'position_confidence': position_confidence,
            'distributions': {
                'momentum': dict(zip(self.MOMENTUM_NAMES, mp_dist)),
                'volatility': dict(zip(self.VOLATILITY_NAMES, vr_dist)),
                'exit_timing': dict(zip(self.EXIT_TIMING_NAMES, et_dist)),
                'position': dict(zip(self.POSITION_NAMES, pp_dist)),
            },
            'delta_states': {
                'leading': self.DELTA_NAMES[delta_leading_idx],
                'coincident': self.DELTA_NAMES[delta_coincident_idx],
                'confirming': self.DELTA_NAMES[delta_confirming_idx],
            }
        }
    
    def _discretize_delta(self, delta: float, threshold: float) -> int:
        """Discretize delta score to state index."""
        if delta < -threshold:
            return 0  # deteriorating
        elif delta > threshold:
            return 2  # improving
        else:
            return 1  # stable
    
    def _discretize_time(self, minutes: float) -> int:
        """Discretize time_since_entry to bucket index."""
        if minutes < 60:
            return 0  # 0-1h
        elif minutes < 240:
            return 1  # 1-4h
        elif minutes < 720:
            return 2  # 4-12h
        else:
            return 3  # 12-24h
    
    def _discretize_pnl(self, pnl_atr: float) -> int:
        """Discretize PnL in ATR units to state index."""
        if pnl_atr < -0.25:
            return 0  # losing
        elif pnl_atr > 0.25:
            return 2  # winning
        else:
            return 1  # breakeven
    
    def _lookup_2parent_cpt(self, node_name: str, p1_idx: int, p2_idx: int) -> np.ndarray:
        """Lookup probability distribution from 2-parent CPT matrix."""
        matrix = self.cpt_matrices.get(node_name)
        if matrix is None:
            # Fallback: uniform distribution
            n_outcomes = len(self.cpt_outcome_names.get(node_name, ['a', 'b', 'c']))
            return np.full(n_outcomes, 1.0 / n_outcomes)
        
        if CUPY_AVAILABLE:
            dist = cp.asnumpy(matrix[p1_idx, p2_idx])
        else:
            dist = matrix[p1_idx, p2_idx]
        return dist
    
    def _lookup_3parent_cpt(self, node_name: str, p1_idx: int, p2_idx: int, p3_idx: int) -> np.ndarray:
        """Lookup probability distribution from 3-parent CPT matrix."""
        matrix = self.cpt_matrices.get(node_name)
        if matrix is None:
            # Fallback: uniform distribution
            n_outcomes = len(self.cpt_outcome_names.get(node_name, ['a', 'b', 'c']))
            return np.full(n_outcomes, 1.0 / n_outcomes)
        
        if CUPY_AVAILABLE:
            dist = cp.asnumpy(matrix[p1_idx, p2_idx, p3_idx])
        else:
            dist = matrix[p1_idx, p2_idx, p3_idx]
        return dist