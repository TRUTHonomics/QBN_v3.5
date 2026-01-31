# Node-Level Diagnostic Report - Asset 1

**Generated:** 2026-01-10 09:18:31

## Executive Summary

| Status | Count |
|--------|-------|
| ✅ PASS | 1 |
| ⚠️ WARN | 0 |
| ❌ FAIL | 9 |

**Verdict:** ❌ **9 node(s) failing** - Critical issues found

## Results Overview

| Node | Status | MI (best) | Key Issue |
|------|--------|-----------|-----------|
| HTF_Regime | ✅ PASS | 0.029 | - |
| Leading_Composite | ❌ FAIL | 0.010 | LOW MI: max=0.010 |
| Coincident_Composite | ❌ FAIL | 0.001 | STUCK: bullish = 100% |
| Confirming_Composite | ❌ FAIL | 0.001 | STUCK: bullish = 100% |
| Trade_Hypothesis | ❌ FAIL | 0.010 | VERY FEW SHORT SIGNALS: 0.1% |
| Entry_Confidence | ❌ FAIL | 0.001 | STUCK: high = 100% |
| Position_Confidence | ❌ FAIL | 0.001 | STUCK: high = 100% |
| Prediction_1h | ❌ FAIL | 0.000 | Accuracy: 3.6%, Dir: 30%, Brier: 0.122 |
| Prediction_4h | ❌ FAIL | 0.000 | Accuracy: 4.0%, Dir: 28%, Brier: 0.122 |
| Prediction_1d | ❌ FAIL | 0.000 | Accuracy: 2.5%, Dir: 30%, Brier: 0.122 |

## Detailed Analysis

### Structural Layer

#### HTF_Regime

**Status:** ✅ PASS
**Samples:** 8743

**State Distribution:**

| State | Percentage |
|-------|------------|
| bearish_transition | 33.2% |
| macro_ranging | 28.6% |
| bullish_transition | 21.2% |
| full_bearish | 11.4% |
| full_bullish | 5.5% |

**Mutual Information:**

- 1h: 0.0030
- 4h: 0.0063
- 1d: 0.0291

### Tactical Layer

#### Leading_Composite

**Status:** ❌ FAIL
**Samples:** 8743

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 86.6% |
| strong_bullish | 13.0% |
| neutral | 0.3% |
| bearish | 0.1% |

**Mutual Information:**

- 1h: 0.0029
- 4h: 0.0053
- 1d: 0.0096

**Issues:**

- LOW MI: max=0.010
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 40%

#### Coincident_Composite

**Status:** ❌ FAIL
**Samples:** 8743

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 99.9% ⚠️ |
| bearish | 0.1% |

**Mutual Information:**

- 1h: 0.0006
- 4h: 0.0000
- 1d: 0.0000

**Issues:**

- STUCK: bullish = 100%
- LOW MI: max=0.001
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 14%

#### Confirming_Composite

**Status:** ❌ FAIL
**Samples:** 8743

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 99.9% ⚠️ |
| bearish | 0.1% |

**Mutual Information:**

- 1h: 0.0003
- 4h: 0.0010
- 1d: 0.0000

**Issues:**

- STUCK: bullish = 100%
- LOW MI: max=0.001
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 30%

### Entry Layer

#### Trade_Hypothesis

**Status:** ❌ FAIL
**Samples:** 8743

**State Distribution:**

| State | Percentage |
|-------|------------|
| weak_long | 86.6% |
| strong_long | 13.0% |
| no_setup | 0.3% |
| weak_short | 0.1% |

**Mutual Information:**

- 1h: 0.0029
- 4h: 0.0053
- 1d: 0.0096

**Issues:**

- VERY FEW SHORT SIGNALS: 0.1%
- LOW MI: max=0.010 (hypothesis geeft geen info)
- strong_long → wrong direction 35% (verwacht >60%)
- weak_long → wrong direction 30% (verwacht >60%)
- weak_short → wrong direction 40% (verwacht >60%)

### Timing Layer

#### Entry_Confidence

**Status:** ❌ FAIL
**Samples:** 8743

**State Distribution:**

| State | Percentage |
|-------|------------|
| high | 100.0% ⚠️ |
| low | 0.0% |

**Mutual Information:**

- 1h: 0.0004
- 4h: 0.0010
- 1d: 0.0000

**Issues:**

- STUCK: high = 100%
- LOW VARIATIE: slechts 1 actieve states
- LOW MI: 0.001
- NO MAGNITUDE CORRELATION: r=0.00

### Management Layer

#### Position_Confidence

**Status:** ❌ FAIL
**Samples:** 8743

**State Distribution:**

| State | Percentage |
|-------|------------|
| high | 99.9% ⚠️ |
| low | 0.1% |

**Mutual Information:**

- 1h: 0.0003
- 4h: 0.0010
- 1d: 0.0000

**Issues:**

- STUCK: high = 100%
- LOW MI: 0.001

### Prediction Layer

#### Prediction_1h

**Status:** ❌ FAIL
**Samples:** 8742

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 3.6%, Dir: 30%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 3.6%
- WORSE THAN RANDOM DIR: 30%

#### Prediction_4h

**Status:** ❌ FAIL
**Samples:** 2184

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 4.0%, Dir: 28%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 4.0%
- WORSE THAN RANDOM DIR: 28%

#### Prediction_1d

**Status:** ❌ FAIL
**Samples:** 362

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 2.5%, Dir: 30%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 2.5%
- WORSE THAN RANDOM DIR: 30%

## Recommendations

### Critical Issues to Fix

**Leading_Composite:**
- LOW MI: max=0.010
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 40%
- **Fix:** Node has no predictive power. May need better feature engineering or more data.

**Coincident_Composite:**
- STUCK: bullish = 100%
- LOW MI: max=0.001
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 14%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Confirming_Composite:**
- STUCK: bullish = 100%
- LOW MI: max=0.001
- POOR BULLISH ALIGNMENT: 31%
- POOR BEARISH ALIGNMENT: 30%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Trade_Hypothesis:**
- VERY FEW SHORT SIGNALS: 0.1%
- LOW MI: max=0.010 (hypothesis geeft geen info)
- strong_long → wrong direction 35% (verwacht >60%)
- weak_long → wrong direction 30% (verwacht >60%)
- weak_short → wrong direction 40% (verwacht >60%)
- **Fix:** Node has no predictive power. May need better feature engineering or more data.

**Entry_Confidence:**
- STUCK: high = 100%
- LOW VARIATIE: slechts 1 actieve states
- LOW MI: 0.001
- NO MAGNITUDE CORRELATION: r=0.00
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Position_Confidence:**
- STUCK: high = 100%
- LOW MI: 0.001
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_1h:**
- Accuracy: 3.6%, Dir: 30%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 3.6%
- WORSE THAN RANDOM DIR: 30%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_4h:**
- Accuracy: 4.0%, Dir: 28%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 4.0%
- WORSE THAN RANDOM DIR: 28%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_1d:**
- Accuracy: 2.5%, Dir: 30%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 2.5%
- WORSE THAN RANDOM DIR: 30%
- **Fix:** Node is stuck in one state. Check input data and thresholds.


---
*Report generated by QBN v3 Node Diagnostic Validator*