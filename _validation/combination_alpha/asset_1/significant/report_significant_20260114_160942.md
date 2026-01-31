# Combination Alpha Analysis Report

**Asset ID:** 1  
**Target Type:** Significant  
**Analysis Date:** 2026-01-14 16:09:41 UTC  
**Lookback Period:** None days  
**Bootstrap Iterations:** 10000  

---

## Executive Summary

### Key Findings

| Metric | Value |
|--------|-------|
| Total Combinations | 24 |
| Golden Rules | 23 (95.8%) |
| Promising | 1 (4.2%) |
| Noise | 0 (0.0%) |
| Analysis Duration | 84.7s |

### Top Golden Rule

**Combination:** `bullish|bullish|neutral`  
**Horizon:** 1d  
**Odds Ratio:** 235.37 (95% CI: 4.66 - 11895.58)  
**Sensitivity:** 0.0%  
**Specificity:** 99.6%  
**Sample Size:** 171.86428833007812


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


## Golden Rules (23 combinations)

These combinations show strong, statistically significant predictive power.

| Horizon | Combination | OR | 95% CI | Sens | Spec | MCC | n |
|---------|-------------|----:|--------|------|------|-----|---|
| 1d | `bullish|bullish|neutral` | 235.37 | 4.66-11895.58 | 0.0% | 99.6% | 0.000 | 171.86428833007812 |
| 1d | `bullish|neutral|bullish` | 195.85 | 3.88-9893.97 | 0.0% | 99.5% | 0.000 | 206.4595184326172 |
| 1d | `bullish|neutral|neutral` | 147.19 | 2.92-7431.03 | 0.0% | 99.3% | 0.000 | 274.431396484375 |
| 4h | `bullish|bullish|neutral` | 91.41 | 1.81-4612.00 | 0.0% | 98.9% | 0.000 | 440.36785888671875 |
| 1d | `bullish|bullish|bullish` | 73.09 | 1.45-3686.67 | 0.0% | 98.7% | 0.000 | 549.4091186523438 |
| 4h | `bullish|bullish|bullish` | 43.40 | 0.86-2188.37 | 0.0% | 97.7% | 0.000 | 917.108642578125 |
| 4h | `bullish|neutral|neutral` | 42.61 | 0.85-2148.80 | 0.0% | 97.7% | 0.000 | 933.6122436523438 |
| 1h | `bullish|bullish|neutral` | 35.79 | 0.71-1804.70 | 0.0% | 97.3% | 0.000 | 1106.787353515625 |
| 4h | `bullish|neutral|bullish` | 23.26 | 0.46-1172.45 | 0.0% | 95.9% | 0.000 | 1679.0372314453125 |
| 1h | `bullish|bullish|bullish` | 17.06 | 0.34-859.84 | 0.0% | 94.5% | 0.000 | 2255.6884765625 |
| 1h | `bullish|neutral|neutral` | 14.30 | 0.28-720.68 | 0.0% | 93.5% | 0.000 | 2662.7783203125 |
| 1h | `neutral|bullish|neutral` | 12.42 | 0.25-626.10 | 0.0% | 92.6% | 0.000 | 3035.071044921875 |
| 4h | `neutral|bullish|neutral` | 9.92 | 0.20-499.93 | 0.0% | 90.8% | 0.000 | 3730.9150390625 |
| 1d | `neutral|bullish|neutral` | 9.82 | 0.19-495.19 | 0.0% | 90.8% | 0.000 | 3763.274169921875 |
| 1h | `bullish|neutral|bullish` | 8.51 | 0.17-429.03 | 0.0% | 89.5% | 0.000 | 4282.599609375 |
| 1h | `neutral|neutral|neutral` | 6.30 | 0.13-317.71 | 0.0% | 86.3% | 0.000 | 5577.830078125 |
| 4h | `neutral|neutral|neutral` | 4.43 | 0.09-223.12 | 0.0% | 81.6% | 0.000 | 7506.83251953125 |
| 1d | `neutral|neutral|neutral` | 3.91 | 0.08-197.01 | 0.0% | 79.6% | 0.000 | 8298.98828125 |
| 1h | `neutral|bullish|bullish` | 3.87 | 0.08-194.82 | 0.0% | 79.4% | 0.000 | 8373.046875 |
| 4h | `neutral|bullish|bullish` | 3.25 | 0.06-163.67 | 0.0% | 76.5% | 0.000 | 9591.484375 |

*...and 3 more. See CSV export for full list.*


## Promising Combinations (1 combinations)

These combinations show potential but need further validation.

| Horizon | Combination | OR | 95% CI | p-adj | n |
|---------|-------------|----:|--------|-------|---|
| 1d | `neutral|neutral|bullish` | 1.35 | 0.03-67.89 | 0.8816 | 17357.41796875 |


## Horizon Breakdown

### Summary by Horizon

| Horizon | Total | Golden | Promising | Noise | Best OR |
|---------|-------|--------|-----------|-------|---------|
| 1h | 8 | 8 | 0 | 0 | 35.79 |
| 4h | 8 | 8 | 0 | 0 | 91.41 |
| 1d | 8 | 7 | 1 | 0 | 235.37 |


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

1. **Implement Golden Rules**: 23 combinations show strong predictive power
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
