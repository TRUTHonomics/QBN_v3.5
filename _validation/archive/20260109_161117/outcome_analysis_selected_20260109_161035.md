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

