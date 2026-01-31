"""
Statistical Tests Module - OR, Chi2, Fisher, MCC, Cramer's V, Information Gain.

ARCHITECTUUR NOOT:
- Alle tests werken op 2x2 contingency tables
- Sommige tests zijn GPU-versnelbaar, andere niet
- Fisher's Exact gebruikt scipy (CPU only)

Gebruik:
    from analysis.statistical_tests import StatisticalTestSuite
    
    suite = StatisticalTestSuite()
    results = suite.run_all_tests(a, b, c, d)
"""

import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class OddsRatioResult:
    """Result of Odds Ratio calculation."""
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float
    se_log_or: float
    significant: bool
    clinically_relevant: bool


@dataclass 
class ChiSquareResult:
    """Result of Chi-square or Fisher's exact test."""
    test_type: str  # 'chi_square' or 'fisher_exact'
    statistic: float  # chi2 value or odds ratio
    p_value: float
    dof: int  # degrees of freedom (1 for 2x2)
    expected_frequencies: Optional[Tuple[float, float, float, float]]


@dataclass
class EffectSizeResult:
    """Effect size metrics."""
    mcc: float  # Matthews Correlation Coefficient
    cramers_v: float  # Cramér's V
    phi: float  # Phi coefficient
    information_gain: float
    effect_interpretation: str  # 'negligible', 'weak', 'moderate', 'strong'


@dataclass
class SensitivitySpecificityResult:
    """Sensitivity, Specificity and related metrics."""
    sensitivity: float  # True Positive Rate
    specificity: float  # True Negative Rate
    ppv: float  # Positive Predictive Value
    npv: float  # Negative Predictive Value
    lr_positive: float  # Positive Likelihood Ratio
    lr_negative: float  # Negative Likelihood Ratio
    accuracy: float
    f1_score: float
    diagnostic_power: str  # 'weak', 'moderate', 'strong'


@dataclass
class FullStatisticalResult:
    """Complete statistical analysis result."""
    # Basic info
    a: float
    b: float
    c: float
    d: float
    n_total: float
    n_with_combination: float
    
    # Test results
    odds_ratio: OddsRatioResult
    chi_square: ChiSquareResult
    effect_size: EffectSizeResult
    sens_spec: SensitivitySpecificityResult
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary."""
        result = {
            'a': self.a, 'b': self.b, 'c': self.c, 'd': self.d,
            'n_total': self.n_total,
            'n_with_combination': self.n_with_combination,
        }
        result.update({f'or_{k}': v for k, v in asdict(self.odds_ratio).items()})
        result.update({f'chi_{k}': v for k, v in asdict(self.chi_square).items() if k != 'expected_frequencies'})
        result.update({f'effect_{k}': v for k, v in asdict(self.effect_size).items()})
        result.update({f'ss_{k}': v for k, v in asdict(self.sens_spec).items()})
        return result


class StatisticalTestSuite:
    """
    Suite of statistical tests for 2x2 contingency tables.
    """
    
    # Continuity correction constant
    CONTINUITY_CORRECTION = 0.5
    
    # Significance level
    ALPHA = 0.05
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize Statistical Test Suite.
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
    
    def run_all_tests(
        self,
        a: float, b: float, c: float, d: float
    ) -> FullStatisticalResult:
        """
        Run all statistical tests on a 2x2 contingency table.
        
        Args:
            a: combination present AND target reached (weighted)
            b: combination present AND target NOT reached (weighted)
            c: combination NOT present AND target reached (weighted)
            d: combination NOT present AND target NOT reached (weighted)
            
        Returns:
            FullStatisticalResult with all metrics
        """
        n_total = a + b + c + d
        n_with_combination = a + b
        
        # Run individual tests
        or_result = self.calculate_odds_ratio(a, b, c, d)
        chi_result = self.calculate_chi_square(a, b, c, d)
        effect_result = self.calculate_effect_sizes(a, b, c, d)
        sens_spec_result = self.calculate_sensitivity_specificity(a, b, c, d)
        
        return FullStatisticalResult(
            a=a, b=b, c=c, d=d,
            n_total=n_total,
            n_with_combination=n_with_combination,
            odds_ratio=or_result,
            chi_square=chi_result,
            effect_size=effect_result,
            sens_spec=sens_spec_result
        )
    
    def calculate_odds_ratio(
        self,
        a: float, b: float, c: float, d: float
    ) -> OddsRatioResult:
        """
        Calculate Odds Ratio with 95% CI via logit method.
        
        OR = (a * d) / (b * c)
        """
        # Apply continuity correction
        cc = self.CONTINUITY_CORRECTION
        a_adj, b_adj, c_adj, d_adj = a + cc, b + cc, c + cc, d + cc
        
        # Odds Ratio
        or_value = (a_adj * d_adj) / (b_adj * c_adj)
        
        # Standard Error of log(OR)
        se_log_or = math.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
        
        # Z-value for alpha
        z = scipy_stats.norm.ppf(1 - self.alpha/2)
        
        # CI in log-space, then transform back
        log_or = math.log(or_value)
        ci_lower = math.exp(log_or - z * se_log_or)
        ci_upper = math.exp(log_or + z * se_log_or)
        
        # P-value (two-sided)
        z_stat = log_or / se_log_or
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
        
        return OddsRatioResult(
            odds_ratio=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            se_log_or=se_log_or,
            significant=ci_lower > 1.0 or ci_upper < 1.0,
            clinically_relevant=or_value > 2.0 and ci_lower > 1.0
        )
    
    def calculate_chi_square(
        self,
        a: float, b: float, c: float, d: float
    ) -> ChiSquareResult:
        """
        Calculate Chi-square or Fisher's exact test.
        
        Uses Fisher's exact when expected frequency < 5.
        REASON: Fisher strictly requires integers, so we round weighted counts.
        """
        contingency = [[a, b], [c, d]]
        n = a + b + c + d
        
        # Calculate expected frequencies
        row1 = a + b
        row2 = c + d
        col1 = a + c
        col2 = b + d
        
        exp_a = (row1 * col1) / n if n > 0 else 0
        exp_b = (row1 * col2) / n if n > 0 else 0
        exp_c = (row2 * col1) / n if n > 0 else 0
        exp_d = (row2 * col2) / n if n > 0 else 0
        
        expected = (exp_a, exp_b, exp_c, exp_d)
        min_expected = min(expected)
        
        if min_expected < 5:
            # Fisher's Exact Test for small samples
            # REASON: Round to nearest integer as Fisher requires discrete counts
            contingency_int = [
                [int(round(a)), int(round(b))],
                [int(round(c)), int(round(d))]
            ]
            odds_ratio, p_value = scipy_stats.fisher_exact(contingency_int)
            return ChiSquareResult(
                test_type='fisher_exact',
                statistic=odds_ratio,
                p_value=p_value,
                dof=1,
                expected_frequencies=expected
            )
        else:
            # Chi-square test (works with floats)
            chi2, p_value, dof, _ = scipy_stats.chi2_contingency(contingency)
            return ChiSquareResult(
                test_type='chi_square',
                statistic=chi2,
                p_value=p_value,
                dof=dof,
                expected_frequencies=expected
            )
    
    def calculate_effect_sizes(
        self,
        a: float, b: float, c: float, d: float
    ) -> EffectSizeResult:
        """
        Calculate effect size metrics: MCC, Cramér's V, Phi, Information Gain.
        """
        n = a + b + c + d
        
        # Matthews Correlation Coefficient
        mcc = self._calculate_mcc(a, b, c, d)
        
        # Phi coefficient (same as MCC for 2x2)
        phi = mcc
        
        # Chi-square for Cramér's V
        contingency = [[a, b], [c, d]]
        try:
            chi2, _, _, _ = scipy_stats.chi2_contingency(contingency)
            cramers_v = math.sqrt(chi2 / n) if n > 0 else 0
        except Exception:
            cramers_v = 0.0
        
        # Information Gain
        info_gain = self._calculate_information_gain(a, b, c, d)
        
        # Interpretation
        abs_effect = max(abs(mcc), cramers_v)
        if abs_effect < 0.1:
            interpretation = 'negligible'
        elif abs_effect < 0.3:
            interpretation = 'weak'
        elif abs_effect < 0.5:
            interpretation = 'moderate'
        else:
            interpretation = 'strong'
        
        return EffectSizeResult(
            mcc=mcc,
            cramers_v=cramers_v,
            phi=phi,
            information_gain=info_gain,
            effect_interpretation=interpretation
        )
    
    def calculate_sensitivity_specificity(
        self,
        a: float, b: float, c: float, d: float
    ) -> SensitivitySpecificityResult:
        """
        Calculate sensitivity, specificity, and related metrics.
        
        For 2x2 table:
        - Sensitivity = a / (a + c) = TPR
        - Specificity = d / (b + d) = TNR
        """
        # Sensitivity (True Positive Rate)
        sensitivity = a / (a + c) if (a + c) > 0 else 0.0
        
        # Specificity (True Negative Rate)
        specificity = d / (b + d) if (b + d) > 0 else 0.0
        
        # Positive Predictive Value
        ppv = a / (a + b) if (a + b) > 0 else 0.0
        
        # Negative Predictive Value
        npv = d / (c + d) if (c + d) > 0 else 0.0
        
        # Likelihood Ratios
        if specificity < 1.0:
            lr_positive = sensitivity / (1 - specificity)
        else:
            lr_positive = float('inf') if sensitivity > 0 else 0.0
        
        if specificity > 0:
            lr_negative = (1 - sensitivity) / specificity
        else:
            lr_negative = float('inf')
        
        # Accuracy
        n = a + b + c + d
        accuracy = (a + d) / n if n > 0 else 0.0
        
        # F1 Score
        if (ppv + sensitivity) > 0:
            f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity)
        else:
            f1_score = 0.0
        
        # Diagnostic power interpretation
        if lr_positive > 10 or lr_negative < 0.1:
            diagnostic_power = 'strong'
        elif lr_positive > 5 or lr_negative < 0.2:
            diagnostic_power = 'moderate'
        else:
            diagnostic_power = 'weak'
        
        return SensitivitySpecificityResult(
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            lr_positive=lr_positive,
            lr_negative=lr_negative,
            accuracy=accuracy,
            f1_score=f1_score,
            diagnostic_power=diagnostic_power
        )
    
    def _calculate_mcc(self, a: float, b: float, c: float, d: float) -> float:
        """
        Calculate Matthews Correlation Coefficient.
        
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        
        For our table:
        - TP = a (combination present, target reached)
        - FP = b (combination present, target not reached)
        - FN = c (combination not present, target reached)
        - TN = d (combination not present, target not reached)
        """
        numerator = (a * d) - (b * c)
        
        denominator_parts = [
            (a + b),  # TP + FP
            (a + c),  # TP + FN
            (d + b),  # TN + FP
            (d + c)   # TN + FN
        ]
        
        # If any part is zero, MCC is undefined (return 0)
        if any(p == 0 for p in denominator_parts):
            return 0.0
        
        denominator = math.sqrt(
            denominator_parts[0] * denominator_parts[1] * 
            denominator_parts[2] * denominator_parts[3]
        )
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_information_gain(
        self,
        a: float, b: float, c: float, d: float
    ) -> float:
        """
        Calculate Information Gain (reduction in entropy).
        
        IG = H(Y) - H(Y|X)
        Where X = combination present/absent, Y = target reached/not
        """
        n = a + b + c + d
        if n == 0:
            return 0.0
        
        # Parent entropy H(Y)
        p_target = (a + c) / n
        parent_entropy = self._entropy(p_target)
        
        # Weighted child entropy H(Y|X)
        n_with_combo = a + b
        n_without_combo = c + d
        
        if n_with_combo > 0:
            p_target_given_combo = a / n_with_combo
            entropy_with_combo = self._entropy(p_target_given_combo)
        else:
            entropy_with_combo = 0.0
        
        if n_without_combo > 0:
            p_target_given_no_combo = c / n_without_combo
            entropy_without_combo = self._entropy(p_target_given_no_combo)
        else:
            entropy_without_combo = 0.0
        
        # Weighted average
        weight_with = n_with_combo / n
        weight_without = n_without_combo / n
        
        child_entropy = (
            weight_with * entropy_with_combo + 
            weight_without * entropy_without_combo
        )
        
        return parent_entropy - child_entropy
    
    def _entropy(self, p: float) -> float:
        """Calculate binary Shannon entropy."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def create_test_suite(alpha: float = 0.05) -> StatisticalTestSuite:
    """Factory function voor StatisticalTestSuite."""
    return StatisticalTestSuite(alpha)

