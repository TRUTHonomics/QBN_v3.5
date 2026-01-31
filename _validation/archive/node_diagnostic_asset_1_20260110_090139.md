# Node-Level Diagnostic Report - Asset 1

**Generated:** 2026-01-10 09:01:39

## Executive Summary

| Status | Count |
|--------|-------|
| ✅ PASS | 2 |
| ⚠️ WARN | 1 |
| ❌ FAIL | 7 |

**Verdict:** ❌ **7 node(s) failing** - Critical issues found

## Results Overview

| Node | Status | MI (best) | Key Issue |
|------|--------|-----------|-----------|
| HTF_Regime | ✅ PASS | 0.446 | - |
| Leading_Composite | ⚠️ WARN | 0.090 | POOR BULLISH ALIGNMENT: 27% |
| Coincident_Composite | ❌ FAIL | 0.006 | STUCK: bullish = 99% |
| Confirming_Composite | ❌ FAIL | 0.013 | STUCK: bullish = 99% |
| Trade_Hypothesis | ✅ PASS | 0.090 | VERY FEW SHORT SIGNALS: 1.4% |
| Entry_Confidence | ❌ FAIL | 0.013 | STUCK: high = 100% |
| Position_Confidence | ❌ FAIL | 0.013 | STUCK: high = 99% |
| Prediction_1h | ❌ FAIL | 0.000 | Accuracy: 3.4%, Dir: 25%, Brier: 0.122 |
| Prediction_4h | ❌ FAIL | 0.000 | Accuracy: 4.6%, Dir: 25%, Brier: 0.122 |
| Prediction_1d | ❌ FAIL | 0.000 | Accuracy: 0.0%, Dir: 22%, Brier: 0.122 |

## Detailed Analysis

### Structural Layer

#### HTF_Regime

**Status:** ✅ PASS
**Samples:** 703

**State Distribution:**

| State | Percentage |
|-------|------------|
| bearish_transition | 55.0% |
| macro_ranging | 27.9% |
| bullish_transition | 11.4% |
| full_bearish | 5.7% |

**Mutual Information:**

- 1h: 0.0245
- 4h: 0.0800
- 1d: 0.4458

### Tactical Layer

#### Leading_Composite

**Status:** ⚠️ WARN
**Samples:** 703

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 84.6% |
| strong_bullish | 13.9% |
| bearish | 1.4% |

**Mutual Information:**

- 1h: 0.0100
- 4h: 0.0448
- 1d: 0.0905

**Issues:**

- POOR BULLISH ALIGNMENT: 27%
- POOR BEARISH ALIGNMENT: 40%

#### Coincident_Composite

**Status:** ❌ FAIL
**Samples:** 703

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 99.0% ⚠️ |
| bearish | 1.0% |

**Mutual Information:**

- 1h: 0.0061
- 4h: 0.0000
- 1d: 0.0000

**Issues:**

- STUCK: bullish = 99%
- LOW MI: max=0.006
- POOR BULLISH ALIGNMENT: 27%
- POOR BEARISH ALIGNMENT: 14%

#### Confirming_Composite

**Status:** ❌ FAIL
**Samples:** 703

**State Distribution:**

| State | Percentage |
|-------|------------|
| bullish | 98.6% ⚠️ |
| bearish | 1.4% |

**Mutual Information:**

- 1h: 0.0029
- 4h: 0.0127
- 1d: 0.0000

**Issues:**

- STUCK: bullish = 99%
- LOW MI: max=0.013
- POOR BULLISH ALIGNMENT: 27%
- POOR BEARISH ALIGNMENT: 30%

### Entry Layer

#### Trade_Hypothesis

**Status:** ✅ PASS
**Samples:** 703

**State Distribution:**

| State | Percentage |
|-------|------------|
| weak_long | 84.6% |
| strong_long | 13.9% |
| weak_short | 1.4% |

**Mutual Information:**

- 1h: 0.0100
- 4h: 0.0448
- 1d: 0.0905

**Issues:**

- VERY FEW SHORT SIGNALS: 1.4%
- strong_long → wrong direction 30% (verwacht >60%)
- weak_long → wrong direction 27% (verwacht >60%)
- weak_short → wrong direction 40% (verwacht >60%)

### Timing Layer

#### Entry_Confidence

**Status:** ❌ FAIL
**Samples:** 703

**State Distribution:**

| State | Percentage |
|-------|------------|
| high | 99.6% ⚠️ |
| low | 0.4% |

**Mutual Information:**

- 1h: 0.0048
- 4h: 0.0127
- 1d: 0.0000

**Issues:**

- STUCK: high = 100%
- LOW VARIATIE: slechts 1 actieve states
- LOW MI: 0.013
- NO MAGNITUDE CORRELATION: r=-0.00

### Management Layer

#### Position_Confidence

**Status:** ❌ FAIL
**Samples:** 703

**State Distribution:**

| State | Percentage |
|-------|------------|
| high | 98.6% ⚠️ |
| low | 1.4% |

**Mutual Information:**

- 1h: 0.0029
- 4h: 0.0127
- 1d: 0.0000

**Issues:**

- STUCK: high = 99%
- LOW MI: 0.013

### Prediction Layer

#### Prediction_1h

**Status:** ❌ FAIL
**Samples:** 702

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 3.4%, Dir: 25%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 3.4%
- WORSE THAN RANDOM DIR: 25%

#### Prediction_4h

**Status:** ❌ FAIL
**Samples:** 174

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 4.6%, Dir: 25%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 4.6%
- WORSE THAN RANDOM DIR: 25%

#### Prediction_1d

**Status:** ❌ FAIL
**Samples:** 27

**State Distribution:**

| State | Percentage |
|-------|------------|
| Strong_Bearish | 100.0% ⚠️ |

**Issues:**

- Accuracy: 0.0%, Dir: 22%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 0.0%
- WORSE THAN RANDOM DIR: 22%

## Recommendations

### Critical Issues to Fix

**Coincident_Composite:**
- STUCK: bullish = 99%
- LOW MI: max=0.006
- POOR BULLISH ALIGNMENT: 27%
- POOR BEARISH ALIGNMENT: 14%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Confirming_Composite:**
- STUCK: bullish = 99%
- LOW MI: max=0.013
- POOR BULLISH ALIGNMENT: 27%
- POOR BEARISH ALIGNMENT: 30%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Entry_Confidence:**
- STUCK: high = 100%
- LOW VARIATIE: slechts 1 actieve states
- LOW MI: 0.013
- NO MAGNITUDE CORRELATION: r=-0.00
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Position_Confidence:**
- STUCK: high = 99%
- LOW MI: 0.013
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_1h:**
- Accuracy: 3.4%, Dir: 25%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 3.4%
- WORSE THAN RANDOM DIR: 25%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_4h:**
- Accuracy: 4.6%, Dir: 25%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 4.6%
- WORSE THAN RANDOM DIR: 25%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

**Prediction_1d:**
- Accuracy: 0.0%, Dir: 22%, Brier: 0.122
- STUCK: Strong_Bearish = 100%
- VERY LOW ACCURACY: 0.0%
- WORSE THAN RANDOM DIR: 22%
- **Fix:** Node is stuck in one state. Check input data and thresholds.

### Warnings to Review

- **Leading_Composite:** POOR BULLISH ALIGNMENT: 27%

---
*Report generated by QBN v3 Node Diagnostic Validator*