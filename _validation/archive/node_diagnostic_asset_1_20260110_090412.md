# Node-Level Diagnostic Report - Asset 1

**Generated:** 2026-01-10 09:04:12

## Executive Summary

| Status | Count |
|--------|-------|
| ✅ PASS | 0 |
| ⚠️ WARN | 0 |
| ❌ FAIL | 10 |

**Verdict:** ❌ **10 node(s) failing** - Critical issues found

## Results Overview

| Node | Status | MI (best) | Key Issue |
|------|--------|-----------|-----------|
| HTF_Regime | ❌ FAIL | 0.011 | LOW MI: max=0.011 (geen predictive po... |
| Leading_Composite | ❌ FAIL | 0.009 | LOW MI: max=0.009 |
| Coincident_Composite | ❌ FAIL | 0.000 | STUCK: bullish = 100% |
| Confirming_Composite | ❌ FAIL | 0.000 | STUCK: bullish = 100% |
| Trade_Hypothesis | ❌ FAIL | 0.009 | VERY FEW SHORT SIGNALS: 0.0% |
| Entry_Confidence | ❌ FAIL | 0.000 | STUCK: high = 100% |
| Position_Confidence | ❌ FAIL | 0.000 | STUCK: high = 100% |
| Prediction_1h | ❌ FAIL | 0.000 | Accuracy: 3.3%, Dir: 29%, Brier: 0.122 |
| Prediction_4h | ❌ FAIL | 0.000 | Accuracy: 3.8%, Dir: 27%, Brier: 0.122 |
| Prediction_1d | ❌ FAIL | 0.000 | Accuracy: 2.9%, Dir: 28%, Brier: 0.122 |

## Detailed Analysis

### Structural Layer

#### HTF_Regime

**Status:** ❌ FAIL
**Samples:** 23983

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish_transition | 29.4% |
| bearish_transition | 28.6% |
| macro_ranging | 24.1% |
| full_bullish | 10.1% |
| full_bearish | 7.9% |

**Mutual Information:**

- 1h: 0.0008
- 4h: 0.0030
- 1d: 0.0112

**Issues:**

- LOW MI: max=0.011 (geen predictive power)

### Tactical Layer

#### Leading_Composite

**Status:** ❌ FAIL
**Samples:** 23983

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 87.6% |
| strong_bullish | 12.1% |
| neutral | 0.2% |
| bearish | 0.0% |

**Mutual Information:**

- 1h: 0.0031
- 4h: 0.0026
- 1d: 0.0088

**Issues:**

- LOW MI: max=0.009
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 40%

#### Coincident_Composite

**Status:** ❌ FAIL
**Samples:** 23983

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 100.0% ⚠️ |
| bearish | 0.0% |

**Mutual Information:**

- 1h: 0.0002
- 4h: 0.0000
- 1d: 0.0000

**Issues:**

- STUCK: bullish = 100%
- LOW MI: max=0.000
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 14%

#### Confirming_Composite

**Status:** ❌ FAIL
**Samples:** 23983

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 100.0% ⚠️ |
| bearish | 0.0% |

**Mutual Information:**

- 1h: 0.0001
- 4h: 0.0004
- 1d: 0.0000

**Issues:**

- STUCK: bullish = 100%
- LOW MI: max=0.000
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 30%

### Entry Layer

#### Trade_Hypothesis

**Status:** ❌ FAIL
**Samples:** 23983

**State Distribution:**

| State | Percentage |
|-------|------------|
| weak_long | 87.6% |
| strong_long | 12.1% |
| no_setup | 0.2% |
| weak_short | 0.0% |

**Mutual Information:**

- 1h: 0.0031
- 4h: 0.0026
- 1d: 0.0088

**Issues:**

- VERY FEW SHORT SIGNALS: 0.0%
- LOW MI: max=0.009 (hypothesis geeft geen info)
- strong_long → wrong direction 33% (verwacht >60%)
- weak_long → wrong direction 31% (verwacht >60%)
- weak_short → wrong direction 40% (verwacht >60%)

### Timing Layer

#### Entry_Confidence

**Status:** ❌ FAIL
**Samples:** 23983

**State Distribution:**

| State | Percentage |
|-------|------------|
| high | 100.0% ⚠️ |
| low | 0.0% |

**Mutual Information:**

- 1h: 0.0001
- 4h: 0.0004
- 1d: 0.0000

**Issues:**

- STUCK: high = 100%
- LOW VARIATIE: slechts 1 actieve states
- LOW MI: 0.000
- NO MAGNITUDE CORRELATION: r=0.00

### Management Layer

#### Position_Confidence

**Status:** ❌ FAIL
**Samples:** 23983

**State Distribution:**

| State | Percentage |
|-------|------------|
| high | 100.0% ⚠️ |
| low | 0.0% |

**Mutual Information:**

- 1h: 0.0001
- 4h: 0.0004
- 1d: 0.0000

**Issues:**

- STUCK: high = 100%
- LOW MI: 0.000

### Prediction Layer

#### Prediction_1h

**Status:** ❌ FAIL
**Samples:** 23982

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 3.3%, Dir: 29%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 3.3%
- WORSE THAN RANDOM DIR: 29%

#### Prediction_4h

**Status:** ❌ FAIL
**Samples:** 5994

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 3.8%, Dir: 27%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 3.8%
- WORSE THAN RANDOM DIR: 27%

#### Prediction_1d

**Status:** ❌ FAIL
**Samples:** 997

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 2.9%, Dir: 28%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 2.9%
- WORSE THAN RANDOM DIR: 28%

## Recommendations

### Critical Issues to Fix

**HTF_Regime:**
- LOW MI: max=0.011 (geen predictive power)
- **Fix:** Node has no predictive power. May need better feature engineering or more data.

**Leading_Composite:**
- LOW MI: max=0.009
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 40%
- **Fix:** Node has no predictive power. May need better feature engineering or more data.

**Coincident_Composite:**
- STUCK: bullish = 100%
- LOW MI: max=0.000
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 14%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Confirming_Composite:**
- STUCK: bullish = 100%
- LOW MI: max=0.000
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 30%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Trade_Hypothesis:**
- VERY FEW SHORT SIGNALS: 0.0%
- LOW MI: max=0.009 (hypothesis geeft geen info)
- strong_long → wrong direction 33% (verwacht >60%)
- weak_long → wrong direction 31% (verwacht >60%)
- weak_short → wrong direction 40% (verwacht >60%)
- **Fix:** Node has no predictive power. May need better feature engineering or more data.

**Entry_Confidence:**
- STUCK: high = 100%
- LOW VARIATIE: slechts 1 actieve states
- LOW MI: 0.000
- NO MAGNITUDE CORRELATION: r=0.00
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Position_Confidence:**
- STUCK: high = 100%
- LOW MI: 0.000
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_1h:**
- Accuracy: 3.3%, Dir: 29%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 3.3%
- WORSE THAN RANDOM DIR: 29%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_4h:**
- Accuracy: 3.8%, Dir: 27%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 3.8%
- WORSE THAN RANDOM DIR: 27%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_1d:**
- Accuracy: 2.9%, Dir: 28%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 2.9%
- WORSE THAN RANDOM DIR: 28%
- **Fix:** Node is stuck in one state. Check input data and thresholds.


---
*Report generated by QBN v3 Node Diagnostic Validator*