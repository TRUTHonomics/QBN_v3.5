# FULL VALIDATION CYCLE REPORT
Generated: 2026-01-04T22:57:52.240935

## INHOUDSOPGAVE
- [concordance_analysis_20260104_225739.md](#concordance-analysis-20260104-225739)
- [cpt_cache_status_20260104_225701.md](#cpt-cache-status-20260104-225701)
- [cpt_validation_asset_1_20260104_225737.md](#cpt-validation-asset-1-20260104-225737)
- [db_stats_20260104_225659.md](#db-stats-20260104-225659)
- [gpu_benchmark_20260104_225700.md](#gpu-benchmark-20260104-225700)
- [latency_profiling_20260104_225752.md](#latency-profiling-20260104-225752)
- [outcome_analysis_selected_20260104_225706.md](#outcome-analysis-selected-20260104-225706)
- [outcome_status_20260104_225702.md](#outcome-status-20260104-225702)
- [wf_report_asset_1_20260104_225746.md](#wf-report-asset-1-20260104-225746)

---

<a name="concordance-analysis-20260104-225739"></a>
## concordance_analysis_20260104_225739.md

# Concordance Analysis Report

**Timestamp:** 2026-01-04T22:57:39.888608

```text
Asset ID: 1
Rows analyzed: 43200

📊 Concordance scenario analyse (RSI signals):

Scenario                       Count          %
---------------------------------------------
neutral                        42765      99.0%
moderate_bullish                 433       1.0%
strong_bullish                     2       0.0%

📊 Concordance score statistieken (concordance_score_d):
   Mean:   0.075
   Median: 0.060
   Std:    0.046
   Min:    0.000
   Max:    0.190
```


---

<a name="cpt-cache-status-20260104-225701"></a>
## cpt_cache_status_20260104_225701.md

# CPT Cache Status Report

**Timestamp:** 2026-01-04T22:57:01.287899

```text
📊 1 assets in cache:

   Asset      Nodes    Obs          Gegenereerd
   -------------------------------------------------------
   1          8        580665       2026-01-04T22:49:46
```


---

<a name="cpt-validation-asset-1-20260104-225737"></a>
## cpt_validation_asset_1_20260104_225737.md

# CPT Validation & Health Report

**Asset ID:** 1
**Lookback:** 30 dagen
**Timestamp:** 2026-01-04T22:57:37.980390

## Health Metrics
```text
Asset  Node                        Cov  Entr  Gain  Stab   Sem      Obs
---------------------------------------------------------------------------
1      Coincident_Composite         0%   0.0  0.00  1.00  0.00        0 ⚠️
1      Confirming_Composite         0%   0.0  0.00  1.00  0.00        0 ⚠️
1      Entry_Timing               100%   1.7  0.30  0.76  1.00  116,133 ✅
1      HTF_Regime                 100%   2.2  0.00  0.52  1.00  116,133 🔴
1      Leading_Composite            0%   0.0  0.00  1.00  0.00        0 ⚠️
1      Prediction_1d              100%   0.7  2.06  0.00  0.67  116,133 🔴
1      Prediction_1h              100%   2.3  0.48  0.00  0.67  116,133 🔴
1      Prediction_4h              100%   2.4  0.45  0.00  1.00  116,133 🔴
```


---

<a name="db-stats-20260104-225659"></a>
## db_stats_20260104_225659.md

# Database Statistics Report

**Timestamp:** 2026-01-04T22:56:59.870619

```text
Tabel                                                   Rijen
------------------------------------------------------------
kfl.mtf_signals_current_lead                                1
kfl.mtf_signals_lead (historical)                   3,039,776
qbn.signal_outcomes                                 3,039,707
qbn.cpt_cache                                               8
qbn.bayesian_predictions                                    7

------------------------------------------------------------
Laatste MTF signal: 2026-01-04 22:55:00+00:00
Outcome coverage: 1h=100.0%, 4h=99.9%, 1d=99.3%
ATR coverage:     1h=100.0%, 4h=99.9%, 1d=99.3%
```


---

<a name="gpu-benchmark-20260104-225700"></a>
## gpu_benchmark_20260104_225700.md

# GPU Performance Benchmark Report

**Timestamp:** 2026-01-04T22:57:00.497690

```text
GPU: NVIDIA GeForce RTX 5080
Memory: 15.9 GB
Compute capability: 12.0
CUDA cores: 10752

🔄 Benchmarks:

Matrix Size        Time (ms)     TFLOPS
----------------------------------------
1000x1000            0.09ms     21.65
2000x2000            0.47ms     33.89
4000x4000            3.53ms     36.23
8000x8000           27.65ms     37.04

Bandwidth: 214.8 GB/s
```


---

<a name="latency-profiling-20260104-225752"></a>
## latency_profiling_20260104_225752.md

# Inference Latency Profiling Report

**Timestamp:** 2026-01-04T22:57:52.234457

```text
🚀 Start profiling voor Asset 1 (1000 iteraties)...

========================================
📊 INFERENCE LATENCY PROFIEL
========================================
  Gemiddelde: 0.01 ms
  Mediaan:    0.01 ms
  P95:        0.02 ms
  P99:        0.03 ms
  Maximum:    0.06 ms
----------------------------------------
  Target:     25.00 ms
  ✅ STATUS: Binnen target (P99: 0.03ms)
========================================


```


---

<a name="outcome-analysis-selected-20260104-225706"></a>
## outcome_analysis_selected_20260104_225706.md

# Outcome Analyse Rapport

**Scope:** selected
**Timestamp:** 2026-01-04T22:57:19.503132+00:00
**Aantal assets:** selected

---

## Output

```
2026-01-04 22:57:08,301 - INFO - Validating 1 selected assets...
2026-01-04 22:57:08,301 - INFO - 
================================================================================
2026-01-04 22:57:08,301 - INFO - OUTCOME BACKFILL VALIDATION REPORT
2026-01-04 22:57:08,301 - INFO - Target Table: qbn.signal_outcomes
2026-01-04 22:57:08,301 - INFO - Asset ID: 1
2026-01-04 22:57:08,301 - INFO - Generated: 2026-01-04 22:57:08.301493
2026-01-04 22:57:08,301 - INFO - ================================================================================

2026-01-04 22:57:08,301 - INFO - ================================================================================
2026-01-04 22:57:08,301 - INFO - CHECK 1: COMPLETENESS (qbn.signal_outcomes)
2026-01-04 22:57:08,301 - INFO - ================================================================================
2026-01-04 22:57:09,497 - INFO -   ✅ 1h: Coverage 100.0% (3,039,707/3,039,718)
2026-01-04 22:57:10,467 - INFO -   ✅ 4h: Coverage 99.9% (3,037,128/3,039,538)
2026-01-04 22:57:11,441 - INFO -   ✅ 1d: Coverage 99.4% (3,019,610/3,038,338)
2026-01-04 22:57:11,442 - INFO - 
================================================================================
2026-01-04 22:57:11,442 - INFO - CHECK 2: LOOKAHEAD BIAS DETECTION (CRITICAL)
2026-01-04 22:57:11,442 - INFO - ================================================================================
2026-01-04 22:57:11,678 - INFO -   ✅ 1h: No lookahead bias violations
2026-01-04 22:57:11,912 - INFO -   ✅ 4h: No lookahead bias violations
2026-01-04 22:57:12,129 - INFO -   ✅ 1d: No lookahead bias violations
2026-01-04 22:57:12,129 - INFO - 
================================================================================
2026-01-04 22:57:12,129 - INFO - CHECK 3: DISTRIBUTION CHECK
2026-01-04 22:57:12,129 - INFO - ================================================================================
2026-01-04 22:57:12,759 - INFO - 
  1h Distribution:
2026-01-04 22:57:12,759 - INFO -     -3:     98,804 (  3.25%)
2026-01-04 22:57:12,759 - INFO -     -2:    162,584 (  5.35%)
2026-01-04 22:57:12,759 - INFO -     -1:    626,925 ( 20.62%)
2026-01-04 22:57:12,759 - INFO -      0:  1,226,510 ( 40.35%)
2026-01-04 22:57:12,759 - INFO -      1:    655,575 ( 21.57%)
2026-01-04 22:57:12,759 - INFO -      2:    169,051 (  5.56%)
2026-01-04 22:57:12,759 - INFO -      3:    100,258 (  3.30%)
2026-01-04 22:57:13,408 - INFO - 
  4h Distribution:
2026-01-04 22:57:13,408 - INFO -     -3:    104,054 (  3.42%)
2026-01-04 22:57:13,408 - INFO -     -2:    156,921 (  5.16%)
2026-01-04 22:57:13,408 - INFO -     -1:    564,289 ( 18.56%)
2026-01-04 22:57:13,408 - INFO -      0:  1,320,853 ( 43.45%)
2026-01-04 22:57:13,408 - INFO -      1:    614,893 ( 20.23%)
2026-01-04 22:57:13,408 - INFO -      2:    167,482 (  5.51%)
2026-01-04 22:57:13,408 - INFO -      3:    108,636 (  3.57%)
2026-01-04 22:57:14,059 - INFO - 
  1d Distribution:
2026-01-04 22:57:14,059 - INFO -     -3:     88,542 (  2.91%)
2026-01-04 22:57:14,059 - INFO -     -2:    176,676 (  5.81%)
2026-01-04 22:57:14,059 - INFO -     -1:    568,289 ( 18.70%)
2026-01-04 22:57:14,059 - INFO -      0:  1,239,013 ( 40.76%)
2026-01-04 22:57:14,059 - INFO -      1:    615,293 ( 20.24%)
2026-01-04 22:57:14,059 - INFO -      2:    208,531 (  6.86%)
2026-01-04 22:57:14,059 - INFO -      3:    123,266 (  4.06%)
2026-01-04 22:57:14,059 - INFO - 
================================================================================
2026-01-04 22:57:14,060 - INFO - CHECK 4: ATR CORRELATION CHECK
2026-01-04 22:57:14,060 - INFO - ================================================================================
2026-01-04 22:57:15,939 - INFO - 
  1h ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-04 22:57:15,939 - INFO -     -3: Avg Norm Return =  -2.01
2026-01-04 22:57:15,939 - INFO -     -2: Avg Norm Return =  -0.95
2026-01-04 22:57:15,939 - INFO -     -1: Avg Norm Return =  -0.45
2026-01-04 22:57:15,939 - INFO -      0: Avg Norm Return =   0.00
2026-01-04 22:57:15,939 - INFO -      1: Avg Norm Return =   0.45
2026-01-04 22:57:15,939 - INFO -      2: Avg Norm Return =   0.95
2026-01-04 22:57:15,939 - INFO -      3: Avg Norm Return =   2.05
2026-01-04 22:57:17,595 - INFO - 
  4h ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-04 22:57:17,595 - INFO -     -3: Avg Norm Return =  -1.91
2026-01-04 22:57:17,595 - INFO -     -2: Avg Norm Return =  -0.95
2026-01-04 22:57:17,595 - INFO -     -1: Avg Norm Return =  -0.44
2026-01-04 22:57:17,595 - INFO -      0: Avg Norm Return =   0.00
2026-01-04 22:57:17,595 - INFO -      1: Avg Norm Return =   0.45
2026-01-04 22:57:17,595 - INFO -      2: Avg Norm Return =   0.95
2026-01-04 22:57:17,595 - INFO -      3: Avg Norm Return =   1.99
2026-01-04 22:57:19,107 - INFO - 
  1d ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-04 22:57:19,107 - INFO -     -3: Avg Norm Return =  -1.80
2026-01-04 22:57:19,107 - INFO -     -2: Avg Norm Return =  -0.95
2026-01-04 22:57:19,107 - INFO -     -1: Avg Norm Return =  -0.45
2026-01-04 22:57:19,107 - INFO -      0: Avg Norm Return =   0.00
2026-01-04 22:57:19,107 - INFO -      1: Avg Norm Return =   0.46
2026-01-04 22:57:19,107 - INFO -      2: Avg Norm Return =   0.95
2026-01-04 22:57:19,107 - INFO -      3: Avg Norm Return =   1.89
2026-01-04 22:57:19,107 - INFO - 
================================================================================
2026-01-04 22:57:19,107 - INFO - ✅ ALL VALIDATIONS PASSED
2026-01-04 22:57:19,107 - INFO - ================================================================================

```


---

<a name="outcome-status-20260104-225702"></a>
## outcome_status_20260104_225702.md

# Outcome Coverage Status Report

**Timestamp:** 2026-01-04T22:57:02.562234

```text

📊 Asset 1 Training Data Status:
   Totaal records:  3,039,777
   Coverage 1h:     100.0%
   Coverage 4h:     99.9%
   Coverage 1d:     99.3%
   Coverage ATR 1h: 100.0%
   Coverage ATR 4h: 99.9%
   Coverage ATR 1d: 99.3%

   Recommendation:  ready
   Ready:           ✅ Yes
```


---

<a name="wf-report-asset-1-20260104-225746"></a>
## wf_report_asset_1_20260104_225746.md

# Walk-Forward Validation Report

**Asset ID:** 1
**Timestamp:** 2026-01-04T22:57:46.498224

## Parameters
- **Start Date:** 2025-09-06 22:57:43.951300
- **End Date:** 2026-01-04 22:57:43.951300
- **Train Window:** 90 days
- **Test Step:** 7 days
- **ATR Thresholds:** [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]

### Signal Weights (1h/4h/1d)
| Signal | 1h | 4h | 1d |
|--------|----|----|----|

## Summary Metrics
```text
============================================================
📊 WALK-FORWARD VALIDATION SUMMARY - ASSET 1
============================================================
Totaal aantal inferences: 672
Aantal windows:          4
----------------------------------------
Horizon 1h:
  Accuracy:      47.92%
  Dir Accuracy:  47.92%
  Brier Score:   0.7136
  Distributie:
    Neutral        : 100.0% ████████████████████

Horizon 4h:
  Accuracy:      47.92%
  Dir Accuracy:  47.92%
  Brier Score:   0.7160
  Distributie:
    Neutral        : 100.0% ████████████████████

Horizon 1d:
  Accuracy:      47.02%
  Dir Accuracy:  47.02%
  Brier Score:   0.7896
  Distributie:
    Neutral        : 100.0% ████████████████████

============================================================
```

## Step Details
| Window | Acc (1h) | Dir Acc (1h) | Brier (1h) | Inferences |
|--------|----------|--------------|------------|------------|
| 2025-12-05 to 2025-12-12 | 48.81% | 48.81% | 0.7120 | 168 |
| 2025-12-12 to 2025-12-19 | 47.62% | 47.62% | 0.7138 | 168 |
| 2025-12-19 to 2025-12-26 | 49.40% | 49.40% | 0.7103 | 168 |
| 2025-12-26 to 2026-01-02 | 45.83% | 45.83% | 0.7181 | 168 |


---

