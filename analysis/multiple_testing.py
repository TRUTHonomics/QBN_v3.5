"""
Multiple Testing Correction Module - FDR-BH and Bonferroni correction.

ARCHITECTUUR NOOT:
- Benjamini-Hochberg FDR is standaard voor 81+ parallel tests
- Bonferroni is conservatiever maar kan te streng zijn
- Altijd beide methodes rapporteren voor volledigheid

Gebruik:
    from analysis.multiple_testing import MultipleTestingCorrector
    
    corrector = MultipleTestingCorrector(method='fdr_bh')
    results = corrector.correct(p_values)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of multiple testing correction."""
    
    # Original p-values
    original_p_values: np.ndarray
    
    # Corrected p-values
    corrected_p_values: np.ndarray
    
    # Significance after correction
    significant_mask: np.ndarray
    
    # Metadata
    method: str
    alpha: float
    n_tests: int
    n_significant: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'alpha': self.alpha,
            'n_tests': self.n_tests,
            'n_significant': self.n_significant,
            'fdr_rate': self.n_significant / self.n_tests if self.n_tests > 0 else 0
        }


class MultipleTestingCorrector:
    """
    Multiple testing correction to control false discovery rate.
    
    When testing 125 combinations × 3 horizons = 375 tests (or more),
    we need to account for increased Type I error probability.
    """
    
    METHODS = ['fdr_bh', 'fdr_by', 'bonferroni', 'holm', 'none']
    
    def __init__(self, method: str = 'fdr_bh', alpha: float = 0.05):
        """
        Initialize Multiple Testing Corrector.
        
        Args:
            method: Correction method
                - 'fdr_bh': Benjamini-Hochberg (recommended, controls FDR)
                - 'fdr_by': Benjamini-Yekutieli (conservative, for dependent tests)
                - 'bonferroni': Bonferroni (very conservative, controls FWER)
                - 'holm': Holm-Bonferroni (step-down, less conservative than Bonferroni)
                - 'none': No correction (not recommended for multiple tests)
            alpha: Significance level (default 0.05)
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Use one of {self.METHODS}")
        
        self.method = method
        self.alpha = alpha
        
        logger.info(f"MultipleTestingCorrector initialized with method={method}, alpha={alpha}")
    
    def correct(
        self,
        p_values: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> CorrectionResult:
        """
        Apply multiple testing correction.
        
        Args:
            p_values: Array of uncorrected p-values
            labels: Optional labels for each test
            
        Returns:
            CorrectionResult with corrected p-values and significance
        """
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        if n_tests == 0:
            return CorrectionResult(
                original_p_values=p_values,
                corrected_p_values=p_values,
                significant_mask=np.array([], dtype=bool),
                method=self.method,
                alpha=self.alpha,
                n_tests=0,
                n_significant=0
            )
        
        # Apply correction
        if self.method == 'fdr_bh':
            corrected = self._benjamini_hochberg(p_values)
        elif self.method == 'fdr_by':
            corrected = self._benjamini_yekutieli(p_values)
        elif self.method == 'bonferroni':
            corrected = self._bonferroni(p_values)
        elif self.method == 'holm':
            corrected = self._holm(p_values)
        else:  # 'none'
            corrected = p_values.copy()
        
        # Determine significance
        significant_mask = corrected < self.alpha
        n_significant = int(np.sum(significant_mask))
        
        logger.info(
            f"Multiple testing correction: {n_significant}/{n_tests} significant "
            f"after {self.method} (α={self.alpha})"
        )
        
        return CorrectionResult(
            original_p_values=p_values,
            corrected_p_values=corrected,
            significant_mask=significant_mask,
            method=self.method,
            alpha=self.alpha,
            n_tests=n_tests,
            n_significant=n_significant
        )
    
    def _benjamini_hochberg(self, p_values: np.ndarray) -> np.ndarray:
        """
        Benjamini-Hochberg procedure for FDR control.
        
        REASON: Standard method for controlling false discovery rate.
        Less conservative than Bonferroni, better for exploratory analysis.
        
        Algorithm:
        1. Sort p-values
        2. Calculate adjusted p-values: p_adj[i] = p[i] * n / rank[i]
        3. Ensure monotonicity (each p_adj >= previous)
        """
        n = len(p_values)
        if n == 0:
            return p_values
        
        # Sort indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Calculate ranks (1-indexed)
        ranks = np.arange(1, n + 1)
        
        # Adjust p-values
        adjusted = sorted_p * n / ranks
        
        # Ensure monotonicity (cumulative minimum from back to front)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        
        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)
        
        # Restore original order
        result = np.empty(n)
        result[sorted_indices] = adjusted
        
        return result
    
    def _benjamini_yekutieli(self, p_values: np.ndarray) -> np.ndarray:
        """
        Benjamini-Yekutieli procedure for FDR control under dependence.
        
        More conservative than B-H, valid even when tests are positively dependent.
        """
        n = len(p_values)
        if n == 0:
            return p_values
        
        # Correction factor for dependence
        cm = np.sum(1.0 / np.arange(1, n + 1))
        
        # Sort indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Calculate ranks
        ranks = np.arange(1, n + 1)
        
        # Adjust p-values with BY correction
        adjusted = sorted_p * n * cm / ranks
        
        # Ensure monotonicity
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        
        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)
        
        # Restore original order
        result = np.empty(n)
        result[sorted_indices] = adjusted
        
        return result
    
    def _bonferroni(self, p_values: np.ndarray) -> np.ndarray:
        """
        Bonferroni correction for FWER control.
        
        Most conservative method: p_adj = p * n
        Controls family-wise error rate but can be too strict.
        """
        n = len(p_values)
        adjusted = p_values * n
        return np.minimum(adjusted, 1.0)
    
    def _holm(self, p_values: np.ndarray) -> np.ndarray:
        """
        Holm-Bonferroni step-down procedure.
        
        Less conservative than Bonferroni but still controls FWER.
        """
        n = len(p_values)
        if n == 0:
            return p_values
        
        # Sort indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Step-down correction
        adjusted = sorted_p * np.arange(n, 0, -1)
        
        # Ensure monotonicity (cumulative maximum from front to back)
        adjusted = np.maximum.accumulate(adjusted)
        
        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)
        
        # Restore original order
        result = np.empty(n)
        result[sorted_indices] = adjusted
        
        return result
    
    def compare_methods(
        self,
        p_values: np.ndarray
    ) -> Dict[str, CorrectionResult]:
        """
        Compare all correction methods on same p-values.
        
        Useful for understanding the impact of different corrections.
        """
        original_method = self.method
        results = {}
        
        for method in self.METHODS:
            if method == 'none':
                continue  # Skip uncorrected
            
            self.method = method
            results[method] = self.correct(p_values)
        
        self.method = original_method
        return results


def apply_fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for FDR-BH correction.
    
    Returns:
        (corrected_p_values, significant_mask)
    """
    corrector = MultipleTestingCorrector(method='fdr_bh', alpha=alpha)
    result = corrector.correct(p_values)
    return result.corrected_p_values, result.significant_mask


def create_corrector(
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> MultipleTestingCorrector:
    """Factory function voor MultipleTestingCorrector."""
    return MultipleTestingCorrector(method, alpha)

