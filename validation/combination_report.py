"""
Combination Alpha Report Generator - Markdown + CSV rapportage.

ARCHITECTUUR NOOT:
- Genereert gestructureerde markdown rapporten
- Exporteert ruwe data naar CSV voor verder analyse
- Bevat interpretatie en aanbevelingen

Gebruik:
    from validation.combination_report import CombinationReportGenerator
    
    generator = CombinationReportGenerator(output_dir)
    report_path = generator.generate_full_report(analysis_result)
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import csv

logger = logging.getLogger(__name__)

# Import types
try:
    from analysis.combination_alpha_analyzer import CombinationResult, AnalysisResult
except ImportError:
    CombinationResult = Any
    AnalysisResult = Any


class CombinationReportGenerator:
    """
    Generates comprehensive markdown reports for combination alpha analysis.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize Report Generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        if output_dir is None:
            output_dir = Path('_validation/combination_alpha')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CombinationReportGenerator initialized: {self.output_dir}")
    
    def generate_full_report(self, result: AnalysisResult) -> Path:
        """
        Generate comprehensive markdown report.
        
        Args:
            result: AnalysisResult from CombinationAlphaAnalyzer
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'report_{result.target_type}_{timestamp}.md'
        
        sections = [
            self._generate_header(result),
            self._generate_executive_summary(result),
            self._generate_methodology(result),
            self._generate_golden_rules_section(result),
            self._generate_promising_section(result),
            self._generate_horizon_breakdown(result),
            self._generate_statistical_notes(result),
            self._generate_recommendations(result),
            self._generate_appendix(result)
        ]
        
        content = '\n\n'.join(sections)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Also export CSV
        self._export_csv(result, report_path.with_suffix('.csv'))
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    
    def _generate_header(self, result: AnalysisResult) -> str:
        """Generate report header."""
        return f"""# Combination Alpha Analysis Report

**Asset ID:** {result.asset_id}  
**Target Type:** {result.target_type.title()}  
**Analysis Date:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Lookback Period:** {result.lookback_days} days  
**Bootstrap Iterations:** {result.n_bootstrap if result.n_bootstrap > 0 else 'Disabled'}  

---"""
    
    def _generate_executive_summary(self, result: AnalysisResult) -> str:
        """Generate executive summary section."""
        all_results = result.all_results()
        
        # Calculate key metrics
        n_total = result.n_total_combinations
        pct_golden = 100 * result.n_golden_rules / max(1, n_total)
        pct_promising = 100 * result.n_promising / max(1, n_total)
        
        # Get best combination
        golden_rules = [r for r in all_results if r.classification == 'golden_rule']
        best_combo = max(golden_rules, key=lambda x: x.odds_ratio) if golden_rules else None
        
        summary = f"""## Executive Summary

### Key Findings

| Metric | Value |
|--------|-------|
| Total Combinations | {n_total} |
| Golden Rules | {result.n_golden_rules} ({pct_golden:.1f}%) |
| Promising | {result.n_promising} ({pct_promising:.1f}%) |
| Noise | {result.n_noise} ({100-pct_golden-pct_promising:.1f}%) |
| Analysis Duration | {result.total_time_seconds:.1f}s |
"""
        
        if best_combo:
            summary += f"""
### Top Golden Rule

**Combination:** `{best_combo.combination_key}`  
**Horizon:** {best_combo.horizon}  
**Odds Ratio:** {best_combo.odds_ratio:.2f} (95% CI: {best_combo.or_ci_lower:.2f} - {best_combo.or_ci_upper:.2f})  
**Sensitivity:** {best_combo.sensitivity:.1%}  
**Specificity:** {best_combo.specificity:.1%}  
**Sample Size:** {best_combo.n_with_combination}
"""
        
        return summary
    
    def _generate_methodology(self, result: AnalysisResult) -> str:
        """Generate methodology section."""
        return f"""## Methodology

### Classification Criteria

| Classification | Criteria |
|----------------|----------|
| **Golden Rule** | OR > 2.0 AND CI_lower > 1.0 AND n ≥ 30 AND MCC > 0.1 |
| **Promising** | OR > 1.5 AND significant after FDR correction |
| **Noise** | Does not meet above criteria |

### Statistical Tests Applied

1. **Odds Ratio** with 95% CI (logit method + bootstrap if enabled)
2. **Chi-square / Fisher's Exact** (Fisher's for expected freq < 5)
3. **Matthews Correlation Coefficient (MCC)** - effect size
4. **Cramér's V** - association strength
5. **Information Gain** - entropy reduction
6. **Sensitivity & Specificity** - diagnostic performance
7. **Likelihood Ratios** - clinical utility

### Multiple Testing Correction

- **Method:** Benjamini-Hochberg FDR
- **Alpha:** 0.05
- **Rationale:** Controls false discovery rate, less conservative than Bonferroni

### Data Parameters

- **Lookback:** {result.lookback_days} days
- **Minimum samples per combination:** {result.min_samples}
- **Bootstrap iterations:** {result.n_bootstrap if result.n_bootstrap > 0 else 'Disabled'}
"""
    
    def _generate_golden_rules_section(self, result: AnalysisResult) -> str:
        """Generate golden rules detailed section."""
        all_results = result.all_results()
        golden_rules = [r for r in all_results if r.classification == 'golden_rule']
        golden_rules = sorted(golden_rules, key=lambda x: x.odds_ratio, reverse=True)
        
        if not golden_rules:
            return """## Golden Rules

*No combinations met Golden Rule criteria.*
"""
        
        content = f"""## Golden Rules ({len(golden_rules)} combinations)

These combinations show strong, statistically significant predictive power.

| Horizon | Combination | OR | 95% CI | Sens | Spec | MCC | n |
|---------|-------------|----:|--------|------|------|-----|---|
"""
        
        for r in golden_rules[:20]:  # Top 20
            ci_str = f"{r.or_ci_lower:.2f}-{r.or_ci_upper:.2f}"
            content += f"| {r.horizon} | `{r.combination_key}` | {r.odds_ratio:.2f} | {ci_str} | {r.sensitivity:.1%} | {r.specificity:.1%} | {r.mcc:.3f} | {r.n_with_combination} |\n"
        
        if len(golden_rules) > 20:
            content += f"\n*...and {len(golden_rules) - 20} more. See CSV export for full list.*\n"
        
        return content
    
    def _generate_promising_section(self, result: AnalysisResult) -> str:
        """Generate promising combinations section."""
        all_results = result.all_results()
        promising = [r for r in all_results if r.classification == 'promising']
        promising = sorted(promising, key=lambda x: x.odds_ratio, reverse=True)
        
        if not promising:
            return """## Promising Combinations

*No combinations met Promising criteria.*
"""
        
        content = f"""## Promising Combinations ({len(promising)} combinations)

These combinations show potential but need further validation.

| Horizon | Combination | OR | 95% CI | p-adj | n |
|---------|-------------|----:|--------|-------|---|
"""
        
        for r in promising[:15]:
            ci_str = f"{r.or_ci_lower:.2f}-{r.or_ci_upper:.2f}"
            p_str = f"{r.p_value_corrected:.4f}" if r.p_value_corrected else "N/A"
            content += f"| {r.horizon} | `{r.combination_key}` | {r.odds_ratio:.2f} | {ci_str} | {p_str} | {r.n_with_combination} |\n"
        
        return content
    
    def _generate_horizon_breakdown(self, result: AnalysisResult) -> str:
        """Generate per-horizon breakdown."""
        content = """## Horizon Breakdown

### Summary by Horizon

| Horizon | Total | Golden | Promising | Noise | Best OR |
|---------|-------|--------|-----------|-------|---------|
"""
        
        for horizon, results in [
            ('1h', result.results_1h),
            ('4h', result.results_4h),
            ('1d', result.results_1d)
        ]:
            n_golden = sum(1 for r in results if r.classification == 'golden_rule')
            n_promising = sum(1 for r in results if r.classification == 'promising')
            n_noise = len(results) - n_golden - n_promising
            best_or = max((r.odds_ratio for r in results), default=0)
            
            content += f"| {horizon} | {len(results)} | {n_golden} | {n_promising} | {n_noise} | {best_or:.2f} |\n"
        
        return content
    
    def _generate_statistical_notes(self, result: AnalysisResult) -> str:
        """Generate statistical interpretation notes."""
        return """## Statistical Interpretation

### Odds Ratio Interpretation

| OR Range | Interpretation |
|----------|----------------|
| 0.5 - 0.7 | Moderate protective effect |
| 0.7 - 1.0 | Weak protective effect |
| 1.0 | No effect |
| 1.0 - 1.5 | Weak risk increase |
| 1.5 - 2.0 | Moderate risk increase |
| > 2.0 | Strong risk increase (clinically relevant) |

### Sensitivity vs Specificity Trade-off

- **High Sensitivity:** Few false negatives, catches most true positives
- **High Specificity:** Few false positives, reliable when positive
- **Ideal:** Both > 70% for clinical utility

### Likelihood Ratio Interpretation

| LR+ | LR- | Diagnostic Power |
|-----|-----|------------------|
| > 10 | < 0.1 | Strong |
| 5-10 | 0.1-0.2 | Moderate |
| 2-5 | 0.2-0.5 | Weak |
| 1-2 | 0.5-1.0 | Negligible |

### MCC Interpretation

| MCC Range | Interpretation |
|-----------|----------------|
| > 0.7 | Strong correlation |
| 0.4 - 0.7 | Moderate correlation |
| 0.1 - 0.4 | Weak correlation |
| < 0.1 | Negligible |
"""
    
    def _generate_recommendations(self, result: AnalysisResult) -> str:
        """Generate actionable recommendations."""
        all_results = result.all_results()
        golden_rules = [r for r in all_results if r.classification == 'golden_rule']
        
        content = """## Recommendations

### For Trading Strategy
"""
        
        if result.n_golden_rules > 0:
            content += f"""
1. **Implement Golden Rules**: {result.n_golden_rules} combinations show strong predictive power
2. **Prioritize high-sensitivity rules** for entry signals
3. **Prioritize high-specificity rules** for exit/confirmation signals
"""
        else:
            content += """
1. **Caution**: No Golden Rules found for this target type
2. Consider analyzing different time periods or target types
3. Promising combinations may need larger sample sizes
"""
        
        content += """
### For Further Analysis

1. **Validate on out-of-sample data** (walk-forward analysis)
2. **Test stability across market regimes** (bull/bear/sideways)
3. **Combine multiple Golden Rules** for ensemble signals
4. **Monitor for alpha decay** over time

### Caveats

- Past performance does not guarantee future results
- Market conditions may invalidate historical patterns
- Bootstrap CI provides robustness but is not immune to bias
- Always use proper risk management
"""
        
        return content
    
    def _generate_appendix(self, result: AnalysisResult) -> str:
        """Generate appendix with technical details."""
        return f"""## Appendix

### Analysis Configuration

```
Asset ID:           {result.asset_id}
Target Type:        {result.target_type}
Lookback Days:      {result.lookback_days}
Min Samples:        {result.min_samples}
Bootstrap N:        {result.n_bootstrap}
Correction Method:  FDR-BH
Alpha:              0.05
```

### Files Generated

- `report_*.md` - This report
- `report_*.csv` - Raw results for all combinations
- `forest_plot_*.png` - Forest plots per horizon
- `sens_spec_scatter_*.png` - Sensitivity/Specificity plots
- `or_heatmap_*.png` - OR heatmaps
- `dashboard_*.png` - Summary dashboard

### References

1. Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate.
2. Matthews, B.W. (1975). Comparison of the predicted and observed secondary structure.
3. Fisher, R.A. (1922). On the interpretation of χ² from contingency tables.

---

*Report generated by QBN Combination Alpha Analysis v2.5*
"""
    
    def _export_csv(self, result: AnalysisResult, csv_path: Path):
        """Export all results to CSV."""
        all_results = result.all_results()
        
        if not all_results:
            return
        
        fieldnames = [
            'horizon', 'combination_key', 'classification', 'target_type',
            'n_with_combination', 'n_total',
            'odds_ratio', 'or_ci_lower', 'or_ci_upper', 'or_p_value',
            'bootstrap_ci_lower', 'bootstrap_ci_upper',
            'sensitivity', 'specificity', 'ppv', 'npv',
            'lr_positive', 'lr_negative',
            'mcc', 'cramers_v', 'information_gain',
            'chi_statistic', 'chi_p_value', 'test_type',
            'p_value_corrected', 'significant_after_correction'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in all_results:
                row = r.to_dict()
                # Handle infinity values
                for key in ['lr_positive', 'lr_negative']:
                    if row.get(key) == float('inf'):
                        row[key] = 'inf'
                writer.writerow({k: row.get(k, '') for k in fieldnames})
        
        logger.info(f"CSV exported: {csv_path}")


def create_report_generator(output_dir: Optional[Path] = None) -> CombinationReportGenerator:
    """Factory function voor CombinationReportGenerator."""
    return CombinationReportGenerator(output_dir)

