"""
Combination Alpha Analyzer - Main orchestration class for combination analysis.

ARCHITECTUUR NOOT:
- Coördineert GPU data loader, contingency builder, bootstrap, en stats tests
- Classificeert combinaties als "Golden Rules", "Promising", of "Noise"
- Slaat resultaten op in database en genereert rapporten

Gebruik:
    analyzer = CombinationAlphaAnalyzer(gpu_config)
    results = analyzer.run_full_analysis(asset_id=1)
    analyzer.save_results(results)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from config.gpu_config import GPUConfig
from analysis.gpu_combination_loader import GPUCombinationDataLoader
from analysis.gpu_contingency_builder import (
    GPUContingencyBuilder, ContingencyTable, STATE_NAMES
)
from analysis.gpu_bootstrap import GPUBootstrapCI, BootstrapResult
from analysis.statistical_tests import StatisticalTestSuite, FullStatisticalResult
from analysis.multiple_testing import MultipleTestingCorrector
from database.db import get_cursor, insert_many

logger = logging.getLogger(__name__)


@dataclass
class CombinationResult:
    """Result for a single combination."""
    
    # Identification
    combination_key: str
    horizon: str
    target_type: str
    
    # Sample info (weighted sums)
    n_with_combination: float
    n_total: float
    
    # Odds Ratio
    odds_ratio: float
    or_ci_lower: float
    or_ci_upper: float
    or_p_value: float
    
    # Bootstrap (if computed)
    bootstrap_ci_lower: Optional[float] = None
    bootstrap_ci_upper: Optional[float] = None
    
    # Sensitivity/Specificity
    sensitivity: float = 0.0
    specificity: float = 0.0
    ppv: float = 0.0
    npv: float = 0.0
    lr_positive: float = 0.0
    lr_negative: float = 0.0
    
    # Effect sizes
    mcc: float = 0.0
    cramers_v: float = 0.0
    information_gain: float = 0.0
    
    # Chi-square/Fisher
    chi_statistic: float = 0.0
    chi_p_value: float = 0.0
    test_type: str = 'chi_square'
    
    # Classification
    classification: str = 'unknown'
    
    # Corrected p-value (after multiple testing)
    p_value_corrected: Optional[float] = None
    significant_after_correction: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Full analysis result for an asset."""
    
    asset_id: int
    target_type: str
    timestamp: datetime
    
    # Configuration
    n_bootstrap: int
    lookback_days: Optional[int]  # None = all data
    min_samples: int
    
    # Results per horizon
    results_1h: List[CombinationResult] = field(default_factory=list)
    results_4h: List[CombinationResult] = field(default_factory=list)
    results_1d: List[CombinationResult] = field(default_factory=list)
    
    # Summary
    n_golden_rules: int = 0
    n_promising: int = 0
    n_noise: int = 0
    
    # Performance
    total_time_seconds: float = 0.0
    
    def all_results(self) -> List[CombinationResult]:
        return self.results_1h + self.results_4h + self.results_1d
    
    @property
    def n_total_combinations(self) -> int:
        return len(self.all_results())


class CombinationAlphaAnalyzer:
    """
    Orchestrates full combination alpha analysis.
    
    Classification criteria:
    - Golden Rule: OR > 2.0, CI_lower > 1.0, n >= 30, MCC > 0.1
    - Promising: 1.5 < OR < 2.0, significant after FDR correction
    - Noise: Everything else
    """
    
    # Classification thresholds
    GOLDEN_RULE_OR = 1.5
    GOLDEN_RULE_MIN_CI = 1.0
    GOLDEN_RULE_MIN_N = 20
    GOLDEN_RULE_MIN_MCC = 0.05
    
    PROMISING_OR = 1.0
    
    HORIZONS = ['1h', '4h', '1d']
    
    def __init__(
        self,
        gpu_config: Optional[GPUConfig] = None,
        n_bootstrap: int = 10000,
        correction_method: str = 'fdr_bh',
        alpha: float = 0.05,
        run_id: Optional[str] = None
    ):
        """
        Initialize Combination Alpha Analyzer.
        
        Args:
            gpu_config: GPU configuration
            n_bootstrap: Number of bootstrap iterations (default 10000)
            correction_method: Multiple testing correction method
            alpha: Significance level
            run_id: Run identifier for traceability
        """
        self.config = gpu_config or GPUConfig.from_env()
        self.n_bootstrap = n_bootstrap
        self.correction_method = correction_method
        self.alpha = alpha
        self.run_id = run_id
        
        # Initialize components
        self.data_loader = GPUCombinationDataLoader(self.config)
        self.test_suite = StatisticalTestSuite(alpha)
        self.corrector = MultipleTestingCorrector(correction_method, alpha)
        
        # Bootstrap calculator will be initialized per analysis
        self.bootstrap: Optional[GPUBootstrapCI] = None
        
        logger.info(
            f"CombinationAlphaAnalyzer initialized: "
            f"n_bootstrap={n_bootstrap}, correction={correction_method}"
        )
    
    def run_full_analysis(
        self,
        asset_id: int,
        target_type: str = 'bullish',
        lookback_days: Optional[int] = None,
        min_samples: int = 30,
        run_bootstrap: bool = True
    ) -> AnalysisResult:
        """
        Run full combination alpha analysis for an asset.
        
        Args:
            asset_id: Asset to analyze
            target_type: 'bullish', 'bearish', or 'significant'
            lookback_days: Days of historical data (None = all data)
            min_samples: Minimum samples per combination
            run_bootstrap: Whether to run bootstrap CI (slower but more robust)
            
        Returns:
            AnalysisResult with all combination results
        """
        start_time = time.perf_counter()
        
        lookback_desc = f"{lookback_days}d" if lookback_days else "all data"
        logger.info(
            f"Starting combination analysis for asset {asset_id}, "
            f"target={target_type}, lookback={lookback_desc}"
        )
        
        # Step 1: Load and cache data
        cached_data = self.data_loader.load_and_cache(
            asset_id=asset_id,
            lookback_days=lookback_days
        )
        
        logger.info(f"Loaded {cached_data.n_rows} data points")
        
        # Step 2: Build contingency tables for all horizons
        builder = GPUContingencyBuilder(
            cached_data.to_dict(),
            use_gpu=self.data_loader.use_gpu
        )
        
        all_tables = builder.build_all_tables(
            target_type=target_type,
            min_samples=min_samples
        )
        
        logger.info(f"Built {len(all_tables)} contingency tables")
        
        # Step 3: Run statistical tests on all tables
        results = self._analyze_all_tables(
            all_tables,
            cached_data,
            builder,
            target_type,
            run_bootstrap
        )
        
        # Step 4: Apply multiple testing correction
        results = self._apply_correction(results)
        
        # Step 5: Classify combinations
        results = self._classify_combinations(results)
        
        # Step 6: Organize by horizon
        results_1h = [r for r in results if r.horizon == '1h']
        results_4h = [r for r in results if r.horizon == '4h']
        results_1d = [r for r in results if r.horizon == '1d']
        
        # Count classifications
        n_golden = sum(1 for r in results if r.classification == 'golden_rule')
        n_promising = sum(1 for r in results if r.classification == 'promising')
        n_noise = sum(1 for r in results if r.classification == 'noise')
        
        total_time = time.perf_counter() - start_time
        
        analysis_result = AnalysisResult(
            asset_id=asset_id,
            target_type=target_type,
            timestamp=datetime.now(timezone.utc),
            n_bootstrap=self.n_bootstrap if run_bootstrap else 0,
            lookback_days=lookback_days,
            min_samples=min_samples,
            results_1h=results_1h,
            results_4h=results_4h,
            results_1d=results_1d,
            n_golden_rules=n_golden,
            n_promising=n_promising,
            n_noise=n_noise,
            total_time_seconds=total_time
        )
        
        logger.info(
            f"Analysis complete: {n_golden} golden rules, {n_promising} promising, "
            f"{n_noise} noise in {total_time:.1f}s"
        )
        
        return analysis_result
    
    def _analyze_all_tables(
        self,
        tables: Dict[str, ContingencyTable],
        cached_data,
        builder: GPUContingencyBuilder,
        target_type: str,
        run_bootstrap: bool
    ) -> List[CombinationResult]:
        """
        Run statistical analysis on all contingency tables.
        """
        results = []
        
        # Initialize bootstrap if needed
        if run_bootstrap:
            self.bootstrap = GPUBootstrapCI(
                use_gpu=self.data_loader.use_gpu,
                seed=42
            )
        
        for key, table in tables.items():
            a, b, c, d = table.to_tuple()
            
            # Run all statistical tests
            stats = self.test_suite.run_all_tests(a, b, c, d)
            
            # Create result
            result = CombinationResult(
                combination_key=table.combination_key,
                horizon=table.horizon,
                target_type=target_type,
                n_with_combination=table.n_with_combination,
                n_total=table.n_total,
                
                # OR
                odds_ratio=stats.odds_ratio.odds_ratio,
                or_ci_lower=stats.odds_ratio.ci_lower,
                or_ci_upper=stats.odds_ratio.ci_upper,
                or_p_value=stats.odds_ratio.p_value,
                
                # Sens/Spec
                sensitivity=stats.sens_spec.sensitivity,
                specificity=stats.sens_spec.specificity,
                ppv=stats.sens_spec.ppv,
                npv=stats.sens_spec.npv,
                lr_positive=stats.sens_spec.lr_positive,
                lr_negative=stats.sens_spec.lr_negative,
                
                # Effect sizes
                mcc=stats.effect_size.mcc,
                cramers_v=stats.effect_size.cramers_v,
                information_gain=stats.effect_size.information_gain,
                
                # Chi-square
                chi_statistic=stats.chi_square.statistic,
                chi_p_value=stats.chi_square.p_value,
                test_type=stats.chi_square.test_type
            )
            
            results.append(result)
        
        # Run bootstrap for promising combinations (if enabled)
        if run_bootstrap and self.bootstrap:
            results = self._add_bootstrap_ci(results, cached_data, builder)
        
        return results
    
    def _add_bootstrap_ci(
        self,
        results: List[CombinationResult],
        cached_data,
        builder: GPUContingencyBuilder
    ) -> List[CombinationResult]:
        """
        Add bootstrap CI for combinations with OR > 1.5.
        
        REASON: Only bootstrap interesting combinations to save time.
        
        KRITIEK: Gebruikt horizon-specifieke composites voor correcte bootstrap!
        """
        xp = builder.xp
        
        # Pre-compute combo_ids per horizon
        # REASON: Elk horizon heeft zijn eigen composites
        combo_ids_by_horizon = {}
        
        for horizon in builder.HORIZONS:
            # Haal horizon-specifieke composites op
            leading, coincident, confirming = builder._get_composites_for_horizon(horizon)
            
            leading_states = builder._discretize_composite(leading)
            coincident_states = builder._discretize_composite(coincident)
            confirming_states = builder._discretize_composite(confirming)
            
            combo_ids_by_horizon[horizon] = builder._encode_combination_ids(
                leading_states, coincident_states, confirming_states
            )
        
        n_bootstrapped = 0
        for result in results:
            # Only bootstrap if OR is interesting
            if result.odds_ratio < self.PROMISING_OR:
                continue
            
            # Get horizon-specific combo_ids
            combo_ids = combo_ids_by_horizon.get(result.horizon)
            if combo_ids is None:
                continue
            
            # Get masks for this combination
            combo_key = result.combination_key
            combo_id = self._key_to_id(combo_key)
            
            combo_mask = combo_ids == combo_id
            target_mask, valid_mask = builder._build_target_mask(
                result.horizon, result.target_type
            )
            
            # Apply valid mask
            valid_combo_mask = combo_mask & valid_mask
            valid_target_mask = target_mask & valid_mask
            
            # Run bootstrap
            try:
                bootstrap_result = self.bootstrap.calculate_or_ci(
                    valid_combo_mask,
                    valid_target_mask,
                    n_bootstrap=self.n_bootstrap
                )
                
                result.bootstrap_ci_lower = bootstrap_result.ci_lower
                result.bootstrap_ci_upper = bootstrap_result.ci_upper
                n_bootstrapped += 1
            except Exception as e:
                logger.warning(f"Bootstrap failed for {combo_key}: {e}")
        
        logger.info(f"Added bootstrap CI to {n_bootstrapped} combinations")
        return results
    
    def _key_to_id(self, key: str) -> int:
        """Convert combination key back to ID."""
        parts = key.split('|')
        state_to_int = {v: k for k, v in STATE_NAMES.items()}
        l = state_to_int[parts[0]]
        c = state_to_int[parts[1]]
        f = state_to_int[parts[2]]
        return l * 25 + c * 5 + f
    
    def _apply_correction(
        self,
        results: List[CombinationResult]
    ) -> List[CombinationResult]:
        """
        Apply multiple testing correction to p-values.
        """
        if not results:
            return results
        
        # Extract p-values
        p_values = np.array([r.or_p_value for r in results])
        
        # Apply correction
        correction_result = self.corrector.correct(p_values)
        
        # Update results
        for i, result in enumerate(results):
            result.p_value_corrected = float(correction_result.corrected_p_values[i])
            result.significant_after_correction = bool(
                correction_result.significant_mask[i]
            )
        
        return results
    
    def _classify_combinations(
        self,
        results: List[CombinationResult]
    ) -> List[CombinationResult]:
        """
        Classify combinations as golden_rule, promising, or noise.
        REASON: Drempels verlaagd op verzoek gebruiker voor meer sensitiviteit.
        """
        for result in results:
            # 1. Check basic significance (95% BI > 1.0)
            ci_lower = result.or_ci_lower
            if result.bootstrap_ci_lower is not None:
                ci_lower = result.bootstrap_ci_lower
            
            is_significant = ci_lower > self.GOLDEN_RULE_MIN_CI
            
            # 2. Golden Rule criteria (OR > 1.5 & BI > 1.0)
            is_golden = (
                result.odds_ratio > self.GOLDEN_RULE_OR and
                is_significant and
                result.n_with_combination >= self.GOLDEN_RULE_MIN_N and
                result.mcc > self.GOLDEN_RULE_MIN_MCC
            )
            
            # 3. Apply classification
            if is_golden:
                result.classification = 'golden_rule'
            elif (
                result.odds_ratio > self.PROMISING_OR and
                is_significant
            ):
                result.classification = 'promising'
            else:
                result.classification = 'noise'
        
        return results
    
    def save_to_database(self, result: AnalysisResult) -> int:
        """
        Save analysis results to database.
        
        Returns:
            Number of rows inserted
        """
        rows = []
        columns = [
            'asset_id', 'horizon', 'combination_key', 'target_type',
            'n_with_combination', 'n_total',
            'odds_ratio', 'or_ci_lower', 'or_ci_upper',
            'bootstrap_ci_lower', 'bootstrap_ci_upper',
            'sensitivity', 'specificity', 'ppv', 'npv',
            'lr_positive', 'lr_negative',
            'mcc', 'cramers_v', 'information_gain',
            'chi_statistic', 'chi_p_value', 'test_type',
            'p_value_corrected', 'classification',
            'analyzed_at', 'run_id'
        ]
        
        import math

        def safe_float(val, default=None):
            """Convert to float and handle inf/nan for Postgres NUMERIC."""
            if val is None:
                return default
            try:
                f_val = float(val)
                if math.isinf(f_val) or math.isnan(f_val):
                    return default
                # Postgres NUMERIC(12,4) max value is 99,999,999.9999
                if f_val > 99999999:
                    return 99999999.0
                if f_val < -99999999:
                    return -99999999.0
                return f_val
            except (ValueError, TypeError):
                return default

        for r in result.all_results():
            rows.append((
                int(result.asset_id),
                str(r.horizon),
                str(r.combination_key),
                str(r.target_type),
                int(r.n_with_combination),
                int(r.n_total),
                safe_float(r.odds_ratio, 0.0),
                safe_float(r.or_ci_lower, 0.0),
                safe_float(r.or_ci_upper, 0.0),
                safe_float(r.bootstrap_ci_lower),
                safe_float(r.bootstrap_ci_upper),
                safe_float(r.sensitivity, 0.0),
                safe_float(r.specificity, 0.0),
                safe_float(r.ppv, 0.0),
                safe_float(r.npv, 0.0),
                safe_float(r.lr_positive),
                safe_float(r.lr_negative),
                safe_float(r.mcc, 0.0),
                safe_float(r.cramers_v, 0.0),
                safe_float(r.information_gain, 0.0),
                safe_float(r.chi_statistic, 0.0),
                safe_float(r.chi_p_value, 1.0),
                str(r.test_type),
                safe_float(r.p_value_corrected),
                str(r.classification),
                result.timestamp,
                self.run_id
            ))
        
        if rows:
            insert_many(
                table='qbn.combination_alpha',
                columns=columns,
                rows=rows,
                page_size=500
            )
        
        saved_count = len(rows)
        logger.info(f"Saved {saved_count} combination results to database")
        
        # HANDSHAKE_OUT logging
        from core.step_validation import log_handshake_out
        from core.run_retention import retain_recent_runs_auto
        log_handshake_out(
            step="run_combination_analysis",
            target="qbn.combination_alpha",
            run_id=self.run_id or "N/A",
            rows=saved_count,
            operation="INSERT"
        )
        
        # Retentie: bewaar 3 meest recente runs
        if self.run_id:
            from database.db import get_cursor
            with get_cursor() as cur:
                retain_recent_runs_auto(cur.connection, "qbn.combination_alpha", result.asset_id)
        
        return saved_count
    
    def save_to_json(
        self,
        result: AnalysisResult,
        output_path: Path
    ) -> Path:
        """
        Save analysis results to JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'asset_id': result.asset_id,
            'target_type': result.target_type,
            'timestamp': result.timestamp.isoformat(),
            'n_bootstrap': result.n_bootstrap,
            'lookback_days': result.lookback_days,
            'min_samples': result.min_samples,
            'summary': {
                'n_golden_rules': result.n_golden_rules,
                'n_promising': result.n_promising,
                'n_noise': result.n_noise,
                'total_time_seconds': result.total_time_seconds
            },
            'results_1h': [r.to_dict() for r in result.results_1h],
            'results_4h': [r.to_dict() for r in result.results_4h],
            'results_1d': [r.to_dict() for r in result.results_1d]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved results to {output_path}")
        return output_path
    
    def get_golden_rules(self, result: AnalysisResult) -> List[CombinationResult]:
        """Extract only golden rule combinations."""
        return [r for r in result.all_results() if r.classification == 'golden_rule']
    
    def get_promising(self, result: AnalysisResult) -> List[CombinationResult]:
        """Extract only promising combinations."""
        return [r for r in result.all_results() if r.classification == 'promising']


def create_analyzer(
    gpu_config: Optional[GPUConfig] = None,
    n_bootstrap: int = 10000
) -> CombinationAlphaAnalyzer:
    """Factory function voor CombinationAlphaAnalyzer."""
    return CombinationAlphaAnalyzer(gpu_config, n_bootstrap)

