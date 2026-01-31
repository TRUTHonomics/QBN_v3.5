"""
GPU Bootstrap Confidence Interval Calculator - Non-parametrische CI via GPU.

ARCHITECTUUR NOOT:
- 10.000 bootstrap iteraties per combinatie
- GPU is ~100x sneller dan CPU voor bootstrap resampling
- Robuuster dan parametrische logit-methode bij niet-normale data

Gebruik:
    bootstrap = GPUBootstrapCI(use_gpu=True)
    ci = bootstrap.calculate_or_ci(combination_mask, outcome_mask, n_bootstrap=10000)
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval calculation."""
    
    # Point estimate
    odds_ratio: float
    
    # Bootstrap CI
    ci_lower: float
    ci_upper: float
    
    # Additional stats
    median_or: float
    mean_or: float
    std_or: float
    
    # Metadata
    n_bootstrap: int
    confidence_level: float
    computation_time_ms: float
    
    @property
    def ci_width(self) -> float:
        """Width of confidence interval (ratio)."""
        return self.ci_upper / self.ci_lower if self.ci_lower > 0 else float('inf')
    
    @property
    def is_significant(self) -> bool:
        """Check if OR is significantly different from 1.0."""
        return self.ci_lower > 1.0 or self.ci_upper < 1.0


class GPUBootstrapCI:
    """
    GPU-versnelde Bootstrap Confidence Interval Calculator.
    
    Berekent non-parametrische CI voor Odds Ratios via resampling.
    Kritiek voor robuuste statistische validatie.
    """
    
    # Default parameters
    DEFAULT_N_BOOTSTRAP = 10000
    DEFAULT_CONFIDENCE = 0.95
    
    def __init__(self, use_gpu: bool = True, seed: Optional[int] = None):
        """
        Initialize GPU Bootstrap CI Calculator.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            seed: Random seed for reproducibility
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.seed = seed
        
        if self.use_gpu:
            logger.info("GPUBootstrapCI using GPU acceleration")
            if seed is not None:
                cp.random.seed(seed)
        else:
            logger.info("GPUBootstrapCI using CPU mode")
            if seed is not None:
                np.random.seed(seed)
    
    @property
    def xp(self):
        """Get array library (CuPy if GPU, NumPy if CPU)."""
        return cp if self.use_gpu else np
    
    def _calculate_or_from_counts(
        self,
        a: Any, b: Any, c: Any, d: Any,
        continuity_correction: float = 0.5
    ) -> Any:
        """
        Calculate Odds Ratio from contingency table counts.
        
        OR = (a * d) / (b * c)
        With continuity correction for zero cells.
        """
        xp = self.xp
        
        # Apply continuity correction
        a_adj = a + continuity_correction
        b_adj = b + continuity_correction
        c_adj = c + continuity_correction
        d_adj = d + continuity_correction
        
        # Calculate OR
        numerator = a_adj * d_adj
        denominator = b_adj * c_adj
        
        # Avoid division by zero
        return xp.where(denominator > 0, numerator / denominator, xp.nan)
    
    def calculate_or_ci(
        self,
        combination_mask: Any,
        outcome_mask: Any,
        weights: Any = None,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        confidence: float = DEFAULT_CONFIDENCE
    ) -> BootstrapResult:
        """
        Calculate bootstrap CI for Odds Ratio.
        
        Args:
            combination_mask: Boolean array (True = combination present)
            outcome_mask: Boolean array (True = target outcome reached)
            weights: Optional uniqueness weights (1/N)
            n_bootstrap: Number of bootstrap iterations
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            BootstrapResult with OR and CI
        """
        xp = self.xp
        start_time = time.perf_counter()
        
        # Ensure arrays are on correct device
        if self.use_gpu:
            if not isinstance(combination_mask, cp.ndarray):
                combination_mask = cp.asarray(combination_mask)
            if not isinstance(outcome_mask, cp.ndarray):
                outcome_mask = cp.asarray(outcome_mask)
            if weights is not None and not isinstance(weights, cp.ndarray):
                weights = cp.asarray(weights)
        else:
            if isinstance(combination_mask, cp.ndarray) if CUPY_AVAILABLE else False:
                combination_mask = cp.asnumpy(combination_mask)
            if isinstance(outcome_mask, cp.ndarray) if CUPY_AVAILABLE else False:
                outcome_mask = cp.asnumpy(outcome_mask)
            if weights is not None and isinstance(weights, cp.ndarray) if CUPY_AVAILABLE else False:
                weights = cp.asnumpy(weights)
        
        n = len(combination_mask)
        
        # Calculate observed OR first (weighted if weights provided)
        if weights is not None:
            a_obs = float(xp.sum(weights[combination_mask & outcome_mask]))
            b_obs = float(xp.sum(weights[combination_mask & ~outcome_mask]))
            c_obs = float(xp.sum(weights[~combination_mask & outcome_mask]))
            d_obs = float(xp.sum(weights[~combination_mask & ~outcome_mask]))
        else:
            a_obs = float(xp.sum(combination_mask & outcome_mask))
            b_obs = float(xp.sum(combination_mask & ~outcome_mask))
            c_obs = float(xp.sum(~combination_mask & outcome_mask))
            d_obs = float(xp.sum(~combination_mask & ~outcome_mask))
        
        observed_or = float(self._calculate_or_from_counts(
            xp.array(a_obs), xp.array(b_obs), 
            xp.array(c_obs), xp.array(d_obs)
        ))
        
        # Bootstrap resampling
        bootstrap_ors = self._run_bootstrap(
            combination_mask, outcome_mask, n_bootstrap, weights=weights
        )
        
        # Calculate CI percentiles
        alpha = 1.0 - confidence
        
        if self.use_gpu:
            bootstrap_ors_cpu = cp.asnumpy(bootstrap_ors)
        else:
            bootstrap_ors_cpu = bootstrap_ors
        
        # Remove NaN values
        valid_ors = bootstrap_ors_cpu[~np.isnan(bootstrap_ors_cpu)]
        
        if len(valid_ors) < n_bootstrap * 0.9:
            logger.warning(f"Many invalid bootstrap samples: {n_bootstrap - len(valid_ors)}")
        
        ci_lower = float(np.percentile(valid_ors, alpha / 2 * 100))
        ci_upper = float(np.percentile(valid_ors, (1 - alpha / 2) * 100))
        median_or = float(np.median(valid_ors))
        mean_or = float(np.mean(valid_ors))
        std_or = float(np.std(valid_ors))
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        return BootstrapResult(
            odds_ratio=observed_or,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            median_or=median_or,
            mean_or=mean_or,
            std_or=std_or,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence,
            computation_time_ms=computation_time
        )
    
    def _run_bootstrap(
        self,
        combination_mask: Any,
        outcome_mask: Any,
        n_bootstrap: int,
        weights: Any = None
    ) -> Any:
        """
        Run bootstrap resampling on GPU/CPU.
        
        REASON: Vectorized bootstrap voor maximale GPU throughput.
        """
        xp = self.xp
        n = len(combination_mask)
        
        # Pre-allocate result array
        bootstrap_ors = xp.zeros(n_bootstrap, dtype=xp.float32)
        
        # Generate all random indices at once
        indices = xp.random.randint(0, n, size=(n_bootstrap, n))
        
        # Vectorized bootstrap calculation
        for i in range(n_bootstrap):
            sample_idx = indices[i]
            sample_combo = combination_mask[sample_idx]
            sample_outcome = outcome_mask[sample_idx]
            
            if weights is not None:
                sample_weights = weights[sample_idx]
                a = xp.sum(sample_weights[sample_combo & sample_outcome])
                b = xp.sum(sample_weights[sample_combo & ~sample_outcome])
                c = xp.sum(sample_weights[~sample_combo & sample_outcome])
                d = xp.sum(sample_weights[~sample_combo & ~sample_outcome])
            else:
                a = xp.sum(sample_combo & sample_outcome)
                b = xp.sum(sample_combo & ~sample_outcome)
                c = xp.sum(~sample_combo & sample_outcome)
                d = xp.sum(~sample_combo & ~sample_outcome)
            
            # Calculate OR with continuity correction
            bootstrap_ors[i] = self._calculate_or_from_counts(a, b, c, d)
        
        return bootstrap_ors
    
    def calculate_batch_ci(
        self,
        data: Dict[str, Any],
        combination_ids: Any,
        target_mask: Any,
        combo_id_list: list,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        confidence: float = DEFAULT_CONFIDENCE
    ) -> Dict[int, BootstrapResult]:
        """
        Calculate bootstrap CI for multiple combinations in batch.
        
        Args:
            data: Cached combination data
            combination_ids: Encoded combination IDs
            target_mask: Boolean mask for target outcome
            combo_id_list: List of combination IDs to process
            n_bootstrap: Bootstrap iterations per combination
            confidence: Confidence level
            
        Returns:
            Dict mapping combo_id -> BootstrapResult
        """
        results = {}
        
        total_start = time.perf_counter()
        weights = data.get('uniqueness_weight')
        
        for combo_id in combo_id_list:
            xp = self.xp
            combo_mask = combination_ids == combo_id
            
            # Skip if too few samples (weighted sum)
            if weights is not None:
                n_with_combo = float(xp.sum(weights[combo_mask]))
            else:
                n_with_combo = float(xp.sum(combo_mask))
                
            if n_with_combo < 5:  # Lagere drempel voor gewogen data
                continue
            
            result = self.calculate_or_ci(
                combo_mask, target_mask, weights=weights, 
                n_bootstrap=n_bootstrap, confidence=confidence
            )
            results[combo_id] = result
        
        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(f"Batch bootstrap for {len(results)} combinations: {total_time:.1f}ms")
        
        return results


class ParametricORCalculator:
    """
    Parametric Odds Ratio calculator using logit method.
    
    Faster than bootstrap but less robust for non-normal data.
    Used as comparison/fallback.
    """
    
    @staticmethod
    def calculate_or_with_ci(
        a: float, b: float, c: float, d: float,
        alpha: float = 0.05,
        continuity_correction: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate Odds Ratio with 95% CI via logit method.
        
        Args:
            a, b, c, d: Contingency table cells (weighted counts)
            alpha: Significance level (default 0.05 for 95% CI)
            continuity_correction: Added to all cells
            
        Returns:
            Dict with OR, CI bounds, p-value
        """
        from scipy import stats
        
        # Apply continuity correction
        a_adj = a + continuity_correction
        b_adj = b + continuity_correction
        c_adj = c + continuity_correction
        d_adj = d + continuity_correction
        
        # Odds Ratio
        or_value = (a_adj * d_adj) / (b_adj * c_adj)
        
        # Standard Error of log(OR)
        se_log_or = np.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
        
        # Z-value for alpha
        z = stats.norm.ppf(1 - alpha/2)
        
        # CI in log-space, then transform back
        log_or = np.log(or_value)
        ci_lower = np.exp(log_or - z * se_log_or)
        ci_upper = np.exp(log_or + z * se_log_or)
        
        # P-value (two-sided)
        z_stat = log_or / se_log_or
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'odds_ratio': or_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'se_log_or': se_log_or,
            'significant': ci_lower > 1.0 or ci_upper < 1.0,
            'clinically_relevant': or_value > 2.0 and ci_lower > 1.0
        }


def create_bootstrap_calculator(
    use_gpu: bool = True,
    seed: Optional[int] = None
) -> GPUBootstrapCI:
    """Factory function voor GPUBootstrapCI."""
    return GPUBootstrapCI(use_gpu, seed)

