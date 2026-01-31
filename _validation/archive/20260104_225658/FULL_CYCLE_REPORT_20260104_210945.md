# FULL VALIDATION CYCLE REPORT
Generated: 2026-01-04T21:09:45.602202

## INHOUDSOPGAVE
- [concordance_analysis_20260104_210810.md](#concordance-analysis-20260104-210810)
- [cpt_cache_status_20260104_210738.md](#cpt-cache-status-20260104-210738)
- [cpt_validation_asset_1_20260104_210809.md](#cpt-validation-asset-1-20260104-210809)
- [db_stats_20260104_210737.md](#db-stats-20260104-210737)
- [gpu_benchmark_20260104_210737.md](#gpu-benchmark-20260104-210737)
- [latency_profiling_20260104_210945.md](#latency-profiling-20260104-210945)
- [outcome_analysis_selected_20260104_210753.md](#outcome-analysis-selected-20260104-210753)
- [outcome_status_20260104_210740.md](#outcome-status-20260104-210740)
- [wf_report_asset_1_20260104_210823.md](#wf-report-asset-1-20260104-210823)

---

<a name="concordance-analysis-20260104-210810"></a>
## concordance_analysis_20260104_210810.md

# Concordance Analysis Report

**Timestamp:** 2026-01-04T21:08:10.701322

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

<a name="cpt-cache-status-20260104-210738"></a>
## cpt_cache_status_20260104_210738.md

# CPT Cache Status Report

**Timestamp:** 2026-01-04T21:07:38.553484

```text
📊 1 assets in cache:

   Asset      Nodes    Obs          Gegenereerd
   -------------------------------------------------------
   1          8        929008       2026-01-04T19:34:20
```


---

<a name="cpt-validation-asset-1-20260104-210809"></a>
## cpt_validation_asset_1_20260104_210809.md

# CPT Validation & Health Report

**Asset ID:** 1
**Lookback:** 30 dagen
**Timestamp:** 2026-01-04T21:08:09.007450

## Health Metrics
```text
Asset  Node                        Cov  Entr  Gain  Stab   Sem      Obs
---------------------------------------------------------------------------
1      Coincident_Composite       100%   0.3  2.03  0.89  1.00  116,126 ✅
1      Confirming_Composite       100%   0.8  1.51  0.84  1.00  116,126 ✅
1      Entry_Timing               100%   1.8  0.22  0.88  0.29  116,126 ✅
1      HTF_Regime                 100%   3.2  0.00  0.40  1.00  116,126 🔴
1      Leading_Composite          100%   0.1  2.24  0.89  1.00  116,126 ✅
1      Prediction_1d              100%   2.4  0.45  0.00  0.67  116,126 🔴
1      Prediction_1h              100%   0.8  2.05  0.00  0.67  116,126 🔴
1      Prediction_4h              100%   2.4  0.44  0.00  0.67  116,126 🔴
```


---

<a name="db-stats-20260104-210737"></a>
## db_stats_20260104_210737.md

# Database Statistics Report

**Timestamp:** 2026-01-04T21:07:37.208776

```text
Tabel                                                   Rijen
------------------------------------------------------------
kfl.mtf_signals_current_lead                                1
kfl.mtf_signals_lead (historical)                   3,039,643
qbn.signal_outcomes                                 3,039,501
qbn.cpt_cache                                               8
qbn.bayesian_predictions                                   77

------------------------------------------------------------
Laatste MTF signal: 2026-01-04 20:42:00+00:00
Outcome coverage: 1h=100.0%, 4h=99.9%, 1d=99.3%
ATR coverage:     1h=100.0%, 4h=99.9%, 1d=99.3%
```


---

<a name="gpu-benchmark-20260104-210737"></a>
## gpu_benchmark_20260104_210737.md

# GPU Performance Benchmark Report

**Timestamp:** 2026-01-04T21:07:37.789887

```text
GPU: NVIDIA GeForce RTX 5080
Memory: 15.9 GB
Compute capability: 12.0
CUDA cores: 10752

🔄 Benchmarks:

Matrix Size        Time (ms)     TFLOPS
----------------------------------------
1000x1000            0.09ms     22.64
2000x2000            0.47ms     33.91
4000x4000            3.48ms     36.82
8000x8000           27.46ms     37.29

Bandwidth: 266.5 GB/s
```


---

<a name="latency-profiling-20260104-210945"></a>
## latency_profiling_20260104_210945.md

# Inference Latency Profiling Report

**Timestamp:** 2026-01-04T21:09:45.595596

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

<a name="outcome-analysis-selected-20260104-210753"></a>
## outcome_analysis_selected_20260104_210753.md

# Outcome Analyse Rapport

**Scope:** selected
**Timestamp:** 2026-01-04T21:08:01.439734+00:00
**Aantal assets:** selected

---

## Output

```
2026-01-04 21:07:55,323 - INFO - Validating 1 selected assets...
2026-01-04 21:07:55,324 - INFO - 
================================================================================
2026-01-04 21:07:55,324 - INFO - OUTCOME BACKFILL VALIDATION REPORT
2026-01-04 21:07:55,324 - INFO - Target Table: qbn.signal_outcomes
2026-01-04 21:07:55,324 - INFO - Asset ID: 1
2026-01-04 21:07:55,324 - INFO - Generated: 2026-01-04 21:07:55.324241
2026-01-04 21:07:55,324 - INFO - ================================================================================

2026-01-04 21:07:55,324 - INFO - ================================================================================
2026-01-04 21:07:55,324 - INFO - CHECK 1: COMPLETENESS (qbn.signal_outcomes)
2026-01-04 21:07:55,324 - INFO - ================================================================================
2026-01-04 21:07:56,260 - INFO -   ✅ 1h: Coverage 100.0% (3,039,501/3,039,608)
2026-01-04 21:07:56,978 - INFO -   ✅ 4h: Coverage 99.9% (3,036,924/3,039,428)
2026-01-04 21:07:57,637 - INFO -   ✅ 1d: Coverage 99.4% (3,019,406/3,038,228)
2026-01-04 21:07:57,638 - INFO - 
================================================================================
2026-01-04 21:07:57,638 - INFO - CHECK 2: LOOKAHEAD BIAS DETECTION (CRITICAL)
2026-01-04 21:07:57,638 - INFO - ================================================================================
2026-01-04 21:07:57,850 - INFO -   ✅ 1h: No lookahead bias violations
2026-01-04 21:07:58,066 - INFO -   ✅ 4h: No lookahead bias violations
2026-01-04 21:07:58,277 - INFO -   ✅ 1d: No lookahead bias violations
2026-01-04 21:07:58,277 - INFO - 
================================================================================
2026-01-04 21:07:58,277 - INFO - CHECK 3: DISTRIBUTION CHECK
2026-01-04 21:07:58,277 - INFO - ================================================================================
2026-01-04 21:07:58,365 - INFO - 
  1h Distribution:
2026-01-04 21:07:58,365 - INFO -     -3:     98,804 (  3.25%)
2026-01-04 21:07:58,365 - INFO -     -2:    162,583 (  5.35%)
2026-01-04 21:07:58,365 - INFO -     -1:    626,879 ( 20.62%)
2026-01-04 21:07:58,365 - INFO -      0:  1,226,426 ( 40.35%)
2026-01-04 21:07:58,365 - INFO -      1:    655,520 ( 21.57%)
2026-01-04 21:07:58,365 - INFO -      2:    169,031 (  5.56%)
2026-01-04 21:07:58,365 - INFO -      3:    100,258 (  3.30%)
2026-01-04 21:07:58,449 - INFO - 
  4h Distribution:
2026-01-04 21:07:58,450 - INFO -     -3:    104,054 (  3.42%)
2026-01-04 21:07:58,450 - INFO -     -2:    156,921 (  5.16%)
2026-01-04 21:07:58,450 - INFO -     -1:    564,269 ( 18.56%)
2026-01-04 21:07:58,450 - INFO -      0:  1,320,669 ( 43.45%)
2026-01-04 21:07:58,450 - INFO -      1:    614,893 ( 20.23%)
2026-01-04 21:07:58,450 - INFO -      2:    167,482 (  5.51%)
2026-01-04 21:07:58,450 - INFO -      3:    108,636 (  3.57%)
2026-01-04 21:07:58,534 - INFO - 
  1d Distribution:
2026-01-04 21:07:58,534 - INFO -     -3:     88,542 (  2.91%)
2026-01-04 21:07:58,534 - INFO -     -2:    176,676 (  5.81%)
2026-01-04 21:07:58,534 - INFO -     -1:    568,289 ( 18.70%)
2026-01-04 21:07:58,534 - INFO -      0:  1,238,979 ( 40.76%)
2026-01-04 21:07:58,534 - INFO -      1:    615,123 ( 20.24%)
2026-01-04 21:07:58,534 - INFO -      2:    208,531 (  6.86%)
2026-01-04 21:07:58,534 - INFO -      3:    123,266 (  4.06%)
2026-01-04 21:07:58,534 - INFO - 
================================================================================
2026-01-04 21:07:58,534 - INFO - CHECK 4: ATR CORRELATION CHECK
2026-01-04 21:07:58,534 - INFO - ================================================================================
2026-01-04 21:07:59,408 - INFO - 
  1h ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-04 21:07:59,409 - INFO -     -3: Avg Norm Return =  -2.01
2026-01-04 21:07:59,409 - INFO -     -2: Avg Norm Return =  -0.95
2026-01-04 21:07:59,409 - INFO -     -1: Avg Norm Return =  -0.45
2026-01-04 21:07:59,409 - INFO -      0: Avg Norm Return =   0.00
2026-01-04 21:07:59,409 - INFO -      1: Avg Norm Return =   0.45
2026-01-04 21:07:59,409 - INFO -      2: Avg Norm Return =   0.95
2026-01-04 21:07:59,409 - INFO -      3: Avg Norm Return =   2.05
2026-01-04 21:08:00,247 - INFO - 
  4h ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-04 21:08:00,247 - INFO -     -3: Avg Norm Return =  -1.91
2026-01-04 21:08:00,247 - INFO -     -2: Avg Norm Return =  -0.95
2026-01-04 21:08:00,247 - INFO -     -1: Avg Norm Return =  -0.44
2026-01-04 21:08:00,247 - INFO -      0: Avg Norm Return =   0.00
2026-01-04 21:08:00,247 - INFO -      1: Avg Norm Return =   0.45
2026-01-04 21:08:00,247 - INFO -      2: Avg Norm Return =   0.95
2026-01-04 21:08:00,247 - INFO -      3: Avg Norm Return =   1.99
2026-01-04 21:08:01,058 - INFO - 
  1d ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):
2026-01-04 21:08:01,058 - INFO -     -3: Avg Norm Return =  -1.80
2026-01-04 21:08:01,058 - INFO -     -2: Avg Norm Return =  -0.95
2026-01-04 21:08:01,058 - INFO -     -1: Avg Norm Return =  -0.45
2026-01-04 21:08:01,058 - INFO -      0: Avg Norm Return =   0.00
2026-01-04 21:08:01,058 - INFO -      1: Avg Norm Return =   0.46
2026-01-04 21:08:01,058 - INFO -      2: Avg Norm Return =   0.95
2026-01-04 21:08:01,058 - INFO -      3: Avg Norm Return =   1.89
2026-01-04 21:08:01,058 - INFO - 
================================================================================
2026-01-04 21:08:01,058 - INFO - ✅ ALL VALIDATIONS PASSED
2026-01-04 21:08:01,058 - INFO - ================================================================================

```


---

<a name="outcome-status-20260104-210740"></a>
## outcome_status_20260104_210740.md

# Outcome Coverage Status Report

**Timestamp:** 2026-01-04T21:07:40.195866

```text

📊 Asset 1 Training Data Status:
   Totaal records:  3,039,643
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

<a name="wf-report-asset-1-20260104-210823"></a>
## wf_report_asset_1_20260104_210823.md

# Walk-Forward Validation Report

**Asset ID:** 1
**Timestamp:** 2026-01-04T21:08:23.555688

## Parameters
- **Start Date:** 2025-09-06 21:08:14.969085
- **End Date:** 2026-01-04 21:08:14.969085
- **Train Window:** 90 days
- **Test Step:** 7 days
- **ATR Thresholds:** [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]

### Signal Weights (1h/4h/1d)
| Signal | 1h | 4h | 1d |
|--------|----|----|----|
| adx_non_trending_regime_60 | 0.13794787 | 1.0082277 | 0.484964 |
| adx_peak_reversal_60 | 0.16457275 | 0.19917025 | 0.2287673 |
| adx_strong_trend_60 | 0.065061994 | 1.2796576 | 2.5 |
| adx_trend_confirm_60 | 0.2735301 | 1.0697507 | 1.5714971 |
| adx_trend_exhaustion_60 | 0.41819137 | 0.48602176 | 1.0385393 |
| adx_weak_trend_60 | 0.12102563 | 0.5103965 | 1.0588542 |
| ao_bearish_zero_cross_60 | 0.3511275 | 0.16403916 | 0.24643886 |
| ao_bullish_zero_cross_60 | 0.21847625 | 0.14948364 | 0.13014963 |
| ao_saucer_bearish_60 | 0.72590446 | 0.6121616 | 0.6594295 |
| ao_saucer_bullish_60 | 1.0246695 | 0.62927026 | 0.59111357 |
| ao_twin_peaks_bearish_60 | 0.0 | 0.0 | 0.0 |
| ao_twin_peaks_bullish_60 | 0.0 | 0.0 | 0.0 |
| atr_high_volatility_60 | 1.6436623 | 2.4570484 | 2.365583 |
| atr_low_volatility_60 | 0.5058898 | 2.1245356 | 2.5 |
| bb_breakout_long_60 | 2.5 | 2.5 | 1.000459 |
| bb_breakout_short_60 | 2.5 | 2.5 | 0.3944482 |
| bb_mean_reversion_long_60 | 2.5 | 2.5 | 1.5559481 |
| bb_squeeze_60 | 0.37146038 | 2.5 | 2.5 |
| bearish_confluence_strong_60 | 0.91777766 | 2.5 | 2.5 |
| bullish_confluence_strong_60 | 0.50949425 | 2.5 | 2.5 |
| cmf_bearish_bias_60 | 1.5644877 | 0.9933882 | 0.79623824 |
| cmf_bullish_bias_60 | 1.5669041 | 0.9933882 | 0.79623824 |
| cmf_divergence_bearish_60 | 2.5 | 2.5 | 1.9985596 |
| cmf_divergence_bullish_60 | 2.5 | 2.5 | 2.5 |
| cmf_strong_buying_60 | 0.601846 | 1.3974338 | 0.87998384 |
| cmf_strong_selling_60 | 0.19917385 | 0.56380326 | 0.6161201 |
| di_bearish_cross_60 | 0.9747318 | 0.14207552 | 0.1106109 |
| di_bullish_cross_60 | 0.6613181 | 0.2480249 | 0.08401964 |
| di_strong_bearish_60 | 0.8963935 | 2.5 | 2.5 |
| di_strong_bullish_60 | 0.31645298 | 2.3615458 | 2.5 |
| ichi_10_30_60_kumo_breakout_long_60 | 0.90945107 | 0.17755012 | 0.27489424 |
| ichi_10_30_60_kumo_breakout_short_60 | 0.53071356 | 0.32467502 | 0.20714968 |
| ichi_10_30_60_tk_cross_bear_60 | 0.4843363 | 2.5 | 0.12850182 |
| ichi_10_30_60_tk_cross_bull_60 | 0.85490936 | 2.5 | 0.20728923 |
| ichi_6_20_52_kumo_breakout_long_60 | 1.2542726 | 0.8293728 | 0.5129374 |
| ichi_6_20_52_kumo_breakout_short_60 | 1.0203081 | 0.4899539 | 0.23891729 |
| ichi_6_20_52_tk_cross_bear_60 | 2.5 | 2.5 | 0.8042629 |
| ichi_6_20_52_tk_cross_bull_60 | 2.5 | 2.5 | 0.78053975 |
| ichi_7_22_44_tk_cross_bear_60 | 1.6963813 | 2.5 | 0.59837455 |
| ichi_7_22_44_tk_cross_bull_60 | 2.5 | 2.5 | 0.3777139 |
| ichimoku_kumo_breakout_long_60 | 0.95477545 | 0.38308862 | 0.47273588 |
| ichimoku_kumo_breakout_short_60 | 0.6821264 | 0.4330421 | 0.31348684 |
| ichimoku_kumo_twist_bear_60 | 0.13553002 | 0.2545514 | 0.18083689 |
| ichimoku_kumo_twist_bull_60 | 0.46118355 | 0.08170132 | 0.16915135 |
| ichimoku_price_above_kumo_60 | 2.5 | 2.5 | 2.5 |
| ichimoku_price_below_kumo_60 | 2.5 | 2.5 | 2.5 |
| ichimoku_price_in_kumo_60 | 0.39080516 | 1.0229015 | 1.0841613 |
| ichimoku_tenkan_kijun_cross_bear_60 | 0.6143184 | 2.5 | 0.29107833 |
| ichimoku_tenkan_kijun_cross_bull_60 | 1.045842 | 2.5 | 0.28587157 |
| kc_dynamic_resistance_60 | 2.19296 | 2.5 | 2.5 |
| kc_dynamic_support_60 | 2.5 | 2.5 | 2.472049 |
| kc_mean_reversion_long_60 | 2.0372887 | 2.123257 | 2.2711473 |
| kc_mean_reversion_short_60 | 2.5 | 2.355458 | 1.9009567 |
| kc_pullback_long_60 | 1.4021804 | 2.5 | 2.5 |
| kc_pullback_short_60 | 1.4021804 | 2.5 | 2.5 |
| kc_squeeze_60 | 0.39835858 | 2.5 | 2.5 |
| kc_trend_breakout_long_60 | 2.5 | 2.5 | 1.018879 |
| kc_trend_breakout_short_60 | 1.9720105 | 1.1733066 | 0.4472767 |
| macd_20_50_15_bearish_cross_60 | 1.3327386 | 0.5345852 | 0.058272608 |
| macd_20_50_15_bullish_cross_60 | 0.54598707 | 0.3779531 | 0.058937576 |
| macd_5_35_5_bearish_cross_60 | 0.54533046 | 0.17700139 | 0.022346627 |
| macd_5_35_5_bullish_cross_60 | 0.6576963 | 0.44001094 | 0.044788707 |
| macd_6_13_4_bearish_cross_60 | 0.57053477 | 0.19172677 | 0.12794197 |
| macd_6_13_4_bullish_cross_60 | 1.0497751 | 0.30671474 | 0.017199764 |
| macd_8_24_9_bearish_cross_60 | 0.47275516 | 0.72247684 | 0.09348233 |
| macd_8_24_9_bullish_cross_60 | 1.1398228 | 0.42297366 | 0.11164663 |
| macd_bearish_cross_60 | 0.47600153 | 0.2295443 | 0.20376952 |
| macd_bullish_cross_60 | 0.4385348 | 0.83082104 | 0.17286167 |
| macd_divergence_bearish_60 | 2.5 | 2.5 | 2.5 |
| macd_divergence_bullish_60 | 2.5 | 2.5 | 1.6066997 |
| macd_histogram_negative_60 | 2.5 | 2.5 | 0.5403298 |
| macd_histogram_positive_60 | 2.5 | 2.5 | 0.5403298 |
| macd_zero_line_cross_bear_60 | 0.392557 | 0.19451667 | 0.046944268 |
| macd_zero_line_cross_bull_60 | 0.7304132 | 0.24489462 | 0.113364756 |
| mean_reversion_setup_long_60 | 2.5 | 2.5 | 1.0254935 |
| mean_reversion_setup_short_60 | 2.5 | 2.5 | 1.6969827 |
| momentum_bearish_confluence_60 | 2.5 | 2.5 | 2.5 |
| momentum_bullish_confluence_60 | 2.5 | 2.5 | 2.5 |
| mtf_bearish_alignment_60 | 0.0 | 0.0 | 0.0 |
| mtf_bullish_alignment_60 | 0.0 | 0.0 | 0.0 |
| obv_bearish_divergence_60 | 2.5 | 2.5 | 0.79385877 |
| obv_bullish_divergence_60 | 2.5 | 2.5 | 2.2106028 |
| obv_trend_confirm_bear_60 | 2.5 | 2.155991 | 0.23798627 |
| obv_trend_confirm_bull_60 | 2.5 | 2.1552243 | 0.24082308 |
| obv_trend_strength_bear_60 | 2.5 | 2.5 | 0.6743997 |
| obv_trend_strength_bull_60 | 2.310541 | 2.333393 | 1.2877989 |
| regime_ranging_60 | 0.13794787 | 1.0082277 | 0.484964 |
| regime_trending_bearish_60 | 0.8963935 | 2.5 | 2.5 |
| regime_trending_bullish_60 | 0.31645298 | 2.3615458 | 2.5 |
| rsi_center_bearish_60 | 2.5 | 2.5 | 2.5 |
| rsi_center_bullish_60 | 2.5 | 2.5 | 2.5 |
| rsi_center_cross_bear_60 | 0.5013502 | 1.0962327 | 0.26317936 |
| rsi_center_cross_bull_60 | 0.42645207 | 0.42257074 | 0.22655162 |
| rsi_divergence_bearish_60 | 2.5 | 2.5 | 2.5 |
| rsi_divergence_bullish_60 | 2.5 | 2.5 | 2.3924859 |
| rsi_extreme_overbought_60 | 0.14979906 | 0.7966274 | 1.4134293 |
| rsi_extreme_oversold_60 | 0.13864978 | 0.5154417 | 0.24429359 |
| rsi_overbought_60 | 2.1926463 | 2.290422 | 2.3958795 |
| rsi_oversold_60 | 1.5367082 | 2.5 | 1.4806789 |
| stoch_bullish_cross_60 | 0.5560335 | 1.9605547 | 0.16201663 |
| stoch_divergence_bear_60 | 2.5 | 2.5 | 2.5 |
| stoch_divergence_bull_60 | 2.5 | 2.5 | 2.5 |
| stoch_hidden_divergence_bear_60 | 2.5 | 2.5 | 2.5 |
| stoch_hidden_divergence_bull_60 | 2.5 | 2.5 | 2.5 |
| stoch_overbought_60 | 2.5 | 2.5 | 2.5 |
| stoch_oversold_60 | 2.5 | 2.5 | 2.5 |
| super_trend_bearish_60 | 0.6270126 | 2.5 | 2.5 |
| super_trend_bullish_60 | 0.6270126 | 2.5 | 2.5 |
| super_trend_flip_bear_60 | 1.5833613 | 0.78018844 | 0.1273288 |
| super_trend_flip_bull_60 | 1.6993121 | 0.71754074 | 0.051549673 |
| trend_following_breakout_long_60 | 0.27497032 | 2.183333 | 2.5 |
| trend_following_breakout_short_60 | 0.75236064 | 2.5 | 2.5 |
| volatility_breakout_long_60 | 2.5 | 2.5 | 2.495059 |
| volatility_breakout_short_60 | 2.5 | 2.5 | 1.4489331 |
| volume_bearish_confluence_60 | 2.5 | 2.2546341 | 0.6720165 |
| volume_bullish_confluence_60 | 2.5 | 2.5 | 0.42732072 |
| vpvr_hvn_resistance_60 | 1.5386217 | 2.5 | 2.5 |
| vpvr_hvn_support_60 | 0.7402372 | 2.2812335 | 2.484325 |
| vpvr_lvn_breakout_down_60 | 1.3744899 | 0.7041924 | 0.4272917 |
| vpvr_lvn_breakout_up_60 | 2.5 | 1.4234225 | 0.75362647 |
| vpvr_poc_touch_60 | 0.5516699 | 2.5 | 2.5 |
| vpvr_value_area_inside_60 | 1.5495067 | 2.5 | 2.5 |

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
  Brier Score:   0.7073
  Distributie:
    Neutral        : 100.0% ████████████████████

Horizon 4h:
  Accuracy:      48.07%
  Dir Accuracy:  48.07%
  Brier Score:   0.7077
  Distributie:
    Neutral        : 100.0% ████████████████████

Horizon 1d:
  Accuracy:      47.02%
  Dir Accuracy:  47.02%
  Brier Score:   0.6770
  Distributie:
    Neutral        : 100.0% ████████████████████

============================================================
```

## Step Details
| Window | Acc (1h) | Dir Acc (1h) | Brier (1h) | Inferences |
|--------|----------|--------------|------------|------------|
| 2025-12-05 to 2025-12-12 | 48.81% | 48.81% | 0.7065 | 168 |
| 2025-12-12 to 2025-12-19 | 48.21% | 48.21% | 0.7047 | 168 |
| 2025-12-19 to 2025-12-26 | 48.81% | 48.81% | 0.7054 | 168 |
| 2025-12-26 to 2026-01-02 | 45.83% | 45.83% | 0.7127 | 168 |


---

