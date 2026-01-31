# FULL VALIDATION CYCLE REPORT
Generated: 2026-01-09T16:11:17.383841

## INHOUDSOPGAVE
- [concordance_analysis_20260109_161053.md](#concordance-analysis-20260109-161053)
- [cpt_cache_status_20260109_161033.md](#cpt-cache-status-20260109-161033)
- [cpt_validation_asset_1_20260109_161050.md](#cpt-validation-asset-1-20260109-161050)
- [db_stats_20260109_161031.md](#db-stats-20260109-161031)
- [gpu_benchmark_20260109_161031.md](#gpu-benchmark-20260109-161031)
- [latency_profiling_20260109_161117.md](#latency-profiling-20260109-161117)
- [outcome_analysis_selected_20260109_161035.md](#outcome-analysis-selected-20260109-161035)
- [outcome_status_20260109_161033.md](#outcome-status-20260109-161033)
- [wf_report_asset_1_20260109_161112.md](#wf-report-asset-1-20260109-161112)

---

<a name="concordance-analysis-20260109-161053"></a>
## concordance_analysis_20260109_161053.md

# Concordance Analysis Report

**Timestamp:** 2026-01-09T16:10:53.470385

```text
Asset ID: 1
Rows analyzed: 43200

📊 Concordance scenario analyse (RSI signals):

Scenario                       Count          %
---------------------------------------------
neutral                        42761      99.0%
moderate_bullish                 437       1.0%
strong_bullish                     2       0.0%

📊 Concordance score statistieken (concordance_score_d):
   Mean:   0.079
   Median: 0.080
   Std:    0.056
   Min:    -0.140
   Max:    0.190
```


---

<a name="cpt-cache-status-20260109-161033"></a>
## cpt_cache_status_20260109_161033.md

# CPT Cache Status Report

**Timestamp:** 2026-01-09T16:10:33.279046

```text
📭 Geen CPT's in cache
   Tip: Gebruik optie 4 om CPT's te genereren
```


---

<a name="cpt-validation-asset-1-20260109-161050"></a>
## cpt_validation_asset_1_20260109_161050.md

# CPT Validation & Health Report

**Asset ID:** 1
**Lookback:** 30 dagen
**Timestamp:** 2026-01-09T16:10:50.960495

## Health Metrics
```text
📭 Geen CPT's in cache voor kwaliteitsrapportage.
```


---

<a name="db-stats-20260109-161031"></a>
## db_stats_20260109_161031.md

# Database Statistics Report

**Timestamp:** 2026-01-09T16:10:31.047729

```text
Tabel                                                   Rijen
------------------------------------------------------------
kfl.mtf_signals_current_lead                                1
kfl.mtf_signals_lead (historical)                   3,046,555
qbn.signal_outcomes                                         0
qbn.cpt_cache                                               0
qbn.bayesian_predictions                                    0

------------------------------------------------------------
Laatste MTF signal: 2026-01-09 16:09:00+00:00
```


---

<a name="gpu-benchmark-20260109-161031"></a>
## gpu_benchmark_20260109_161031.md

# GPU Performance Benchmark Report

**Timestamp:** 2026-01-09T16:10:31.754674

```text
GPU: NVIDIA GeForce RTX 5080
Memory: 15.9 GB
Compute capability: 12.0
CUDA cores: 10752

🔄 Benchmarks:

Matrix Size        Time (ms)     TFLOPS
----------------------------------------
1000x1000            0.09ms     21.41
2000x2000            0.50ms     31.84
4000x4000            3.69ms     34.71
8000x8000           29.48ms     34.73

Bandwidth: 202.7 GB/s
```


---

<a name="latency-profiling-20260109-161117"></a>
## latency_profiling_20260109_161117.md

# Inference Latency Profiling Report

**Timestamp:** 2026-01-09T16:11:17.376674

```text
🚀 Start profiling voor Asset 1 (1000 iteraties)...

========================================
📊 INFERENCE LATENCY PROFIEL
========================================
  Gemiddelde: 0.01 ms
  Mediaan:    0.01 ms
  P95:        0.01 ms
  P99:        0.01 ms
  Maximum:    0.04 ms
----------------------------------------
  Target:     25.00 ms
  ✅ STATUS: Binnen target (P99: 0.01ms)
========================================


```


---

<a name="outcome-analysis-selected-20260109-161035"></a>
## outcome_analysis_selected_20260109_161035.md

# Outcome Analyse Rapport (FOUT)

**Scope:** selected
**Timestamp:** 2026-01-09T16:10:40.981737+00:00
**Exit code:** 1

---

## Output

```
2026-01-09 16:10:37,956 - INFO - Validating 1 selected assets...
2026-01-09 16:10:37,956 - INFO - 
================================================================================
2026-01-09 16:10:37,956 - INFO - OUTCOME BACKFILL VALIDATION REPORT
2026-01-09 16:10:37,956 - INFO - Target Table: qbn.signal_outcomes
2026-01-09 16:10:37,956 - INFO - Asset ID: 1
2026-01-09 16:10:37,956 - INFO - Generated: 2026-01-09 16:10:37.956396
2026-01-09 16:10:37,956 - INFO - ================================================================================

2026-01-09 16:10:37,956 - INFO - ================================================================================
2026-01-09 16:10:37,956 - INFO - CHECK 1: COMPLETENESS (qbn.signal_outcomes)
2026-01-09 16:10:37,956 - INFO - ================================================================================
2026-01-09 16:10:38,814 - WARNING -   ⚠️  1h (via time_60): Coverage 0.0% (missing 50,776 rows)
2026-01-09 16:10:39,374 - WARNING -   ⚠️  4h (via time_240): Coverage 0.0% (missing 12,694 rows)
2026-01-09 16:10:39,923 - WARNING -   ⚠️  1d (via time_d): Coverage 0.0% (missing 2,115 rows)
2026-01-09 16:10:39,923 - INFO - 
================================================================================
2026-01-09 16:10:39,923 - INFO - CHECK 2: LOOKAHEAD BIAS DETECTION (CRITICAL)
2026-01-09 16:10:39,923 - INFO - ================================================================================
2026-01-09 16:10:39,934 - INFO -   ✅ 1h (via time_60): No lookahead bias violations
2026-01-09 16:10:39,945 - INFO -   ✅ 4h (via time_240): No lookahead bias violations
2026-01-09 16:10:39,956 - INFO -   ✅ 1d (via time_d): No lookahead bias violations
2026-01-09 16:10:39,956 - INFO - 
================================================================================
2026-01-09 16:10:39,956 - INFO - CHECK 3: DISTRIBUTION CHECK
2026-01-09 16:10:39,956 - INFO - ================================================================================
2026-01-09 16:10:39,959 - INFO - 
  1h Distribution:
2026-01-09 16:10:39,959 - WARNING -     -3: Missing data
2026-01-09 16:10:39,960 - WARNING -     -2: Missing data
2026-01-09 16:10:39,960 - WARNING -     -1: Missing data
2026-01-09 16:10:39,960 - WARNING -      0: Missing data
2026-01-09 16:10:39,960 - WARNING -      1: Missing data
2026-01-09 16:10:39,960 - WARNING -      2: Missing data
2026-01-09 16:10:39,960 - WARNING -      3: Missing data
2026-01-09 16:10:39,963 - INFO - 
  4h Distribution:
2026-01-09 16:10:39,963 - WARNING -     -3: Missing data
2026-01-09 16:10:39,963 - WARNING -     -2: Missing data
2026-01-09 16:10:39,963 - WARNING -     -1: Missing data
2026-01-09 16:10:39,963 - WARNING -      0: Missing data
2026-01-09 16:10:39,963 - WARNING -      1: Missing data
2026-01-09 16:10:39,963 - WARNING -      2: Missing data
2026-01-09 16:10:39,963 - WARNING -      3: Missing data
2026-01-09 16:10:39,965 - INFO - 
  1d Distribution:
2026-01-09 16:10:39,966 - WARNING -     -3: Missing data
2026-01-09 16:10:39,966 - WARNING -     -2: Missing data
2026-01-09 16:10:39,966 - WARNING -     -1: Missing data
2026-01-09 16:10:39,966 - WARNING -      0: Missing data
2026-01-09 16:10:39,966 - WARNING -      1: Missing data
2026-01-09 16:10:39,966 - WARNING -      2: Missing data
2026-01-09 16:10:39,966 - WARNING -      3: Missing data
2026-01-09 16:10:39,966 - INFO - 
================================================================================
2026-01-09 16:10:39,966 - INFO - CHECK 4: ATR CORRELATION CHECK
2026-01-09 16:10:39,966 - INFO - ================================================================================
2026-01-09 16:10:40,052 - INFO - 
  1h ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-09 16:10:40,085 - INFO - 
  4h ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-09 16:10:40,118 - INFO - 
  1d ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-09 16:10:40,118 - INFO - 
================================================================================
2026-01-09 16:10:40,118 - ERROR - ❌ VALIDATION FAILURES DETECTED
2026-01-09 16:10:40,118 - INFO - ================================================================================

```



---

<a name="outcome-status-20260109-161033"></a>
## outcome_status_20260109_161033.md

# Outcome Coverage Status Report

**Timestamp:** 2026-01-09T16:10:33.498839

```text

📊 Asset 1 Training Data Status:
   Totaal records:  3,046,555
   Coverage 1h:     0.0%
   Coverage 4h:     0.0%
   Coverage 1d:     0.0%
   Coverage ATR 1h: 0.0%
   Coverage ATR 4h: 0.0%
   Coverage ATR 1d: 0.0%

   Recommendation:  wait_for_horizon
   Ready:           ❌ No
```


---

<a name="wf-report-asset-1-20260109-161112"></a>
## wf_report_asset_1_20260109_161112.md

# Walk-Forward Validation Report

**Asset ID:** 1
**Timestamp:** 2026-01-09T16:11:12.427688

## Parameters
- **Start Date:** 2025-09-11 16:11:11.374767
- **End Date:** 2026-01-09 16:11:11.374767
- **Train Window:** 90 days
- **Test Step:** 7 days
- **ATR Thresholds:** [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]

### Signal Weights (1h/4h/1d)
| Signal | 1h | 4h | 1d |
|--------|----|----|----|

## Summary Metrics
```text
Geen resultaten om te rapporteren.
```

## Step Details
| Window | Acc (1h) | Dir Acc (1h) | Brier (1h) | Inferences |
|--------|----------|--------------|------------|------------|


---

