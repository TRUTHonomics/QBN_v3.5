# Node-Level Diagnostic Report - Asset 1

**Generated:** 2026-01-14 15:19:04

## Executive Summary

| Status | Count |
|--------|-------|
| ✅ PASS | 1 |
| ⚠️ WARN | 1 |
| ❌ FAIL | 8 |

**Verdict:** ❌ **8 node(s) failing** - Critical issues found

## Results Overview

| Node | Status | MI (best) | Key Issue |
|------|--------|-----------|-----------|
| HTF_Regime | ✅ PASS | 0.228 | - |
| Leading_Composite | ❌ FAIL | 0.013 | HIGH NEUTRAL: 81% (signalen activeren... |
| Coincident_Composite | ❌ FAIL | 0.000 | HIGH NEUTRAL: 100% (signalen activere... |
| Confirming_Composite | ❌ FAIL | 0.000 | HIGH NEUTRAL: 100% (signalen activere... |
| Trade_Hypothesis | ⚠️ WARN | 0.013 | HIGH no_setup: 81% |
| Entry_Confidence | ❌ FAIL | 0.000 | STUCK: medium = 100% |
| Position_Confidence | ❌ FAIL | 0.000 | STUCK: low = 100% |
| Prediction_1h | ❌ FAIL | 0.000 | Accuracy: 0.0%, Dir: 100%, Brier: 0.143 |
| Prediction_4h | ❌ FAIL | 0.004 | Accuracy: 0.0%, Dir: 98%, Brier: 0.143 |
| Prediction_1d | ❌ FAIL | 0.005 | Accuracy: 0.0%, Dir: 64%, Brier: 0.143 |

## Detailed Analysis

### Structural Layer

#### HTF_Regime

**Status:** ✅ PASS
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| bearish_transition | 47.4% |
| macro_ranging | 32.6% |
| bullish_transition | 13.3% |
| full_bearish | 6.7% |

**Mutual Information:**

- 1h: 0.0000
- 4h: 0.0119
- 1d: 0.2279

### Tactical Layer

#### Leading_Composite

**Status:** ❌ FAIL
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| neutral | 81.4% |
| bearish | 9.7% |
| bullish | 9.0% |

**Mutual Information:**

- 1h: 0.0000
- 4h: 0.0104
- 1d: 0.0133

**Issues:**

- HIGH NEUTRAL: 81% (signalen activeren niet)
- LOW MI: max=0.013
- POOR BULLISH ALIGNMENT: 0%
- POOR BEARISH ALIGNMENT: 0%

#### Coincident_Composite

**Status:** ❌ FAIL
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| neutral | 100.0% ⚠️ |

**Issues:**

- HIGH NEUTRAL: 100% (signalen activeren niet)
- STUCK: neutral = 100%
- LOW MI: max=0.000
- POOR BULLISH ALIGNMENT: 0%
- POOR BEARISH ALIGNMENT: 0%

#### Confirming_Composite

**Status:** ❌ FAIL
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| neutral | 100.0% ⚠️ |

**Issues:**

- HIGH NEUTRAL: 100% (signalen activeren niet)
- STUCK: neutral = 100%
- LOW MI: max=0.000
- POOR BULLISH ALIGNMENT: 0%
- POOR BEARISH ALIGNMENT: 0%

### Entry Layer

#### Trade_Hypothesis

**Status:** ⚠️ WARN
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| no_setup | 81.4% |
| weak_short | 9.7% |
| weak_long | 9.0% |

**Mutual Information:**

- 1h: 0.0000
- 4h: 0.0104
- 1d: 0.0133

**Issues:**

- HIGH no_setup: 81%
- LOW MI: max=0.013 (hypothesis geeft geen info)
- weak_long → wrong direction 0% (verwacht >60%)
- weak_short → wrong direction 0% (verwacht >60%)

### Timing Layer

#### Entry_Confidence

**Status:** ❌ FAIL
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| medium | 100.0% ⚠️ |

**Issues:**

- STUCK: medium = 100%
- LOW VARIATIE: slechts 1 actieve states
- LOW MI: 0.000

### Management Layer

#### Position_Confidence

**Status:** ❌ FAIL
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| low | 100.0% ⚠️ |

**Issues:**

- STUCK: low = 100%
- LOW MI: 0.000

### Prediction Layer

#### Prediction_1h

**Status:** ❌ FAIL
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| neutral | 81.4% |
| up_strong | 18.6% |

**Issues:**

- Accuracy: 0.0%, Dir: 100%, Brier: 0.143
- VERY LOW ACCURACY: 0.0%
- POOR CALIBRATION: reliability=0.815

#### Prediction_4h

**Status:** ❌ FAIL
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| neutral | 81.4% |
| up_strong | 18.6% |

**Mutual Information:**

- 4h: 0.0039

**Issues:**

- Accuracy: 0.0%, Dir: 98%, Brier: 0.143
- VERY LOW ACCURACY: 0.0%
- POOR CALIBRATION: reliability=0.788

#### Prediction_1d

**Status:** ❌ FAIL
**Samples:** 601

**State Distribution:**

| State | Percentage |
|-------|------------|
| neutral | 81.4% |
| up_strong | 18.6% |

**Mutual Information:**

- 1d: 0.0048

**Issues:**

- Accuracy: 0.0%, Dir: 64%, Brier: 0.143
- VERY LOW ACCURACY: 0.0%
- POOR CALIBRATION: reliability=0.308

## Recommendations

### Critical Issues to Fix

**Leading_Composite:**
- HIGH NEUTRAL: 81% (signalen activeren niet)
- LOW MI: max=0.013
- POOR BULLISH ALIGNMENT: 0%
- POOR BEARISH ALIGNMENT: 0%
- **Fix:** Node has no predictive power. May need better feature engineering or more data.

**Coincident_Composite:**
- HIGH NEUTRAL: 100% (signalen activeren niet)
- STUCK: neutral = 100%
- LOW MI: max=0.000
- POOR BULLISH ALIGNMENT: 0%
- POOR BEARISH ALIGNMENT: 0%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Confirming_Composite:**
- HIGH NEUTRAL: 100% (signalen activeren niet)
- STUCK: neutral = 100%
- LOW MI: max=0.000
- POOR BULLISH ALIGNMENT: 0%
- POOR BEARISH ALIGNMENT: 0%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Entry_Confidence:**
- STUCK: medium = 100%
- LOW VARIATIE: slechts 1 actieve states
- LOW MI: 0.000
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Position_Confidence:**
- STUCK: low = 100%
- LOW MI: 0.000
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_1h:**
- Accuracy: 0.0%, Dir: 100%, Brier: 0.143
- VERY LOW ACCURACY: 0.0%
- POOR CALIBRATION: reliability=0.815
- **Fix:** Predicted probabilities do not match observed frequencies. CPT needs retraining.

**Prediction_4h:**
- Accuracy: 0.0%, Dir: 98%, Brier: 0.143
- VERY LOW ACCURACY: 0.0%
- POOR CALIBRATION: reliability=0.788
- **Fix:** Predicted probabilities do not match observed frequencies. CPT needs retraining.

**Prediction_1d:**
- Accuracy: 0.0%, Dir: 64%, Brier: 0.143
- VERY LOW ACCURACY: 0.0%
- POOR CALIBRATION: reliability=0.308
- **Fix:** Predicted probabilities do not match observed frequencies. CPT needs retraining.

### Warnings to Review

- **Trade_Hypothesis:** HIGH no_setup: 81%

---
*Report generated by QBN v3 Node Diagnostic Validator*