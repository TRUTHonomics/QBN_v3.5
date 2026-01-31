# Combination Alpha Analysis Report

**Asset ID:** 1  
**Target Type:** Significant  
**Analysis Date:** 2026-01-24 10:46:45 UTC  
**Lookback Period:** None days  
**Bootstrap Iterations:** 10000  

---

## Executive Summary

### Key Findings

| Metric | Value |
|--------|-------|
| Total Combinations | 24 |
| Golden Rules | 2 (8.3%) |
| Promising | 5 (20.8%) |
| Noise | 17 (70.8%) |
| Analysis Duration | 37.5s |

### Top Golden Rule

**Combination:** `neutral|bullish|bullish`  
**Horizon:** 1d  
**Odds Ratio:** 2.02 (95% CI: 1.81 - 2.25)  
**Sensitivity:** 26.9%  
**Specificity:** 84.6%  
**Sample Size:** 7525.3642578125


## Methodology

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

- **Lookback:** None days
- **Minimum samples per combination:** 30
- **Bootstrap iterations:** 10000


## Golden Rules (2 combinations)

These combinations show strong, statistically significant predictive power.

| Horizon | Combination | OR | 95% CI | Sens | Spec | MCC | n |
|---------|-------------|----:|--------|------|------|-----|---|
| 1d | `neutral|bullish|bullish` | 2.02 | 1.81-2.25 | 26.9% | 84.6% | 0.076 | 7525.3642578125 |
| 1h | `neutral|bullish|bullish` | 1.71 | 1.56-1.86 | 32.2% | 78.2% | 0.071 | 6607.174560546875 |


## Promising Combinations (5 combinations)

These combinations show potential but need further validation.

| Horizon | Combination | OR | 95% CI | p-adj | n |
|---------|-------------|----:|--------|-------|---|
| 1d | `bullish|neutral|bullish` | 29.10 | 1.81-467.68 | 0.0231 | 141.4130859375 |
| 1d | `bullish|bullish|bullish` | 2.10 | 1.33-3.31 | 0.0023 | 416.4674987792969 |
| 1h | `bullish|bullish|bullish` | 1.32 | 1.12-1.55 | 0.0014 | 1594.9861450195312 |
| 4h | `neutral|bullish|bullish` | 1.27 | 1.20-1.35 | N/A | 7293.05224609375 |
| 4h | `neutral|neutral|bullish` | 1.07 | 1.02-1.13 | 0.0101 | 11260.828857421875 |


## Horizon Breakdown

### Summary by Horizon

| Horizon | Total | Golden | Promising | Noise | Best OR |
|---------|-------|--------|-----------|-------|---------|
| 1h | 8 | 1 | 1 | 6 | 1.71 |
| 4h | 8 | 0 | 2 | 6 | 1.27 |
| 1d | 8 | 1 | 2 | 5 | 29.10 |


## Statistical Interpretation

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


## Recommendations

### For Trading Strategy

1. **Implement Golden Rules**: 2 combinations show strong predictive power
2. **Prioritize high-sensitivity rules** for entry signals
3. **Prioritize high-specificity rules** for exit/confirmation signals

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


## Appendix

### Analysis Configuration

```
Asset ID:           1
Target Type:        significant
Lookback Days:      None
Min Samples:        30
Bootstrap N:        10000
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
