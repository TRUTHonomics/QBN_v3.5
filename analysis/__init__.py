# analysis/__init__.py
"""
Analysis Module voor QBN v3.

Dit pakket bevat tools voor:
1. Threshold Optimalisatie (Phase 2.4)
   - Mutual Information Grid Search
   - Decision Tree (CART) Auto-Discovery
   - Logistic Regression Weight Optimization

2. Combination Alpha Analysis (Phase 2.5)
   - GPU-accelerated data loading and merging
   - Odds Ratio with 95% CI
   - Sensitivity/Specificity analysis
   - Multiple testing correction (FDR-BH)
"""

from .threshold_optimizer import ThresholdOptimizer, ThresholdAnalysisResult
from .mutual_information_analyzer import MutualInformationAnalyzer
from .decision_tree_analyzer import DecisionTreeAnalyzer
from .logistic_regression_analyzer import LogisticRegressionAnalyzer

# Combination Alpha Analysis (Phase 2.5)
from .gpu_combination_loader import GPUCombinationDataLoader, CachedCombinationData
from .gpu_contingency_builder import GPUContingencyBuilder, ContingencyTable
from .gpu_bootstrap import GPUBootstrapCI, BootstrapResult, ParametricORCalculator
from .statistical_tests import StatisticalTestSuite, FullStatisticalResult
from .multiple_testing import MultipleTestingCorrector, CorrectionResult
from .combination_alpha_analyzer import CombinationAlphaAnalyzer, AnalysisResult

__all__ = [
    # Threshold Optimization
    'ThresholdOptimizer',
    'ThresholdAnalysisResult',
    'MutualInformationAnalyzer',
    'DecisionTreeAnalyzer',
    'LogisticRegressionAnalyzer',
    
    # Combination Alpha Analysis
    'GPUCombinationDataLoader',
    'CachedCombinationData',
    'GPUContingencyBuilder',
    'ContingencyTable',
    'GPUBootstrapCI',
    'BootstrapResult',
    'ParametricORCalculator',
    'StatisticalTestSuite',
    'FullStatisticalResult',
    'MultipleTestingCorrector',
    'CorrectionResult',
    'CombinationAlphaAnalyzer',
    'AnalysisResult',
]

