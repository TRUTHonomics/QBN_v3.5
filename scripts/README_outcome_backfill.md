# Outcome Backfill Scripts - QBN_v3

GPU-accelerated outcome backfill system voor ML training data met **zero lookahead bias**.

## Overview

Deze scripts vullen historische outcome labels in voor `qbn.signal_outcomes`. De outcomes worden gebruikt voor QBN_v3 Conditional Probability Table (CPT) training.

**Kritieke Features:**
- ✅ Zero lookahead bias via strikte timestamp filtering
- ✅ ATR-relative binning (-3 tot +3)
- ✅ GPU acceleration met CuPy
- ✅ Resume capability (kan onderbroken worden)
- ✅ **Multi-Timeframe Architecture**: Elke horizon gebruikt eigen time anchor
- ✅ Comprehensive validation checks

## Multi-Timeframe Architecture (v3.3)

**UPDATE 2026-01-06**: Elke horizon heeft nu zijn eigen time anchor om autocorrelatie te voorkomen.

| Horizon | Time Anchor | Signal Suffix | Verwachte Rijen (BTC) |
|---------|-------------|---------------|----------------------|
| 1h      | time_60     | _60           | ~50.000              |
| 4h      | time_240    | _240          | ~12.500              |
| 1d      | time_d      | _d            | ~2.100               |

Zie `260106_MTF_outcome_architecture.md` voor volledige details.

---

## Scripts

### 1. outcome_backfill.py

Hoofd backfill script met GPU acceleration.

**Features:**
- Vectorized ATR-relative binning op GPU
- Binary COPY voor bulk database updates
- Lookahead-safe data fetching
- Automatic resume capability

**Usage:**

```bash
# Status check
python scripts/outcome_backfill.py --status

# Specifiek asset, specifieke horizon
python scripts/outcome_backfill.py --asset BTCUSDT --horizon 1h

# Alle assets, alle horizons
python scripts/outcome_backfill.py --all

# Met validation
python scripts/outcome_backfill.py --all --verify

# CPU mode (geen GPU)
python scripts/outcome_backfill.py --all --no-gpu
```

**Parameters:**
- `--asset SYMBOL`: Specifiek asset (bijv. BTCUSDT)
- `--all`: Alle assets processeren
- `--horizon {1h|4h|1d|all}`: Specifieke horizon (default: all)
- `--timeframe TIMEFRAME`: Timeframe filter (default: 1h)
- `--batch-size N`: Database batch size (default: 50000)
- `--status`: Toon huidige backfill status
- `--verify`: Run validation na backfill
- `--no-gpu`: Disable GPU (CPU fallback)

---

### 2. validate_outcome_backfill.py

Validation script om correctheid te verifiëren en lookahead bias te detecteren.

**Checks:**
1. **Completeness**: Alle eligible rows hebben outcomes
2. **Lookahead Bias Detection**: GEEN rows met outcomes die te recent zijn
3. **Distribution**: Redelijke verdeling over bins (-3 tot +3)
4. **ATR Correlation**: Extreme outcomes correleren met grote ATR-normalized returns

**Usage:**

```bash
# Quick check (alle assets)
python scripts/validate_outcome_backfill.py --quick

# Specifiek asset
python scripts/validate_outcome_backfill.py --asset BTCUSDT

# Met report
python scripts/validate_outcome_backfill.py --all --report validation_report.txt
```

**Parameters:**
- `--asset SYMBOL`: Specifiek asset
- `--all`: Alle assets valideren
- `--quick`: Quick check (alias voor --all)
- `--report FILE`: Save report to file

---

## Lookahead Bias Prevention

**KRITIEK:** Deze implementatie voorkomt lookahead bias door strikte timestamp filtering.

### Wat is Lookahead Bias?

Lookahead bias treedt op wanneer een model toegang heeft tot toekomstige informatie tijdens training. Dit resulteert in:
- Onrealistisch hoge backtest performance
- Catastrofaal falen in productie
- Worthless model

### Hoe Wordt Het Voorkomen?

**Per Horizon Time Anchor (v3.3):**
```python
HORIZON_CONFIG = {
    '1h': {'time_col': 'time_60', 'time_close_col': 'time_close_60'},
    '4h': {'time_col': 'time_240', 'time_close_col': 'time_close_240'},
    '1d': {'time_col': 'time_d', 'time_close_col': 'time_close_d'}
}
```

**Database Query Filter:**
```sql
-- Entry: signaal is pas bekend bij time_close
-- Exit: entry + horizon minutes later
WHERE {time_close_col} < NOW() - INTERVAL '{horizon_minutes} minutes'
```

Dit betekent:
- Voor 1h horizon: alleen signals met time_close_60 > 1 uur geleden krijgen outcomes
- Voor 4h horizon: alleen signals met time_close_240 > 4 uur geleden
- Voor 1d horizon: alleen signals met time_close_d > 1 dag geleden

**Validation Check:**
```sql
SELECT COUNT(*) FROM qbn.signal_outcomes so
JOIN kfl.mtf_signals_lead mtf ON so.asset_id = mtf.asset_id 
    AND so.time_1 = mtf.{time_col}
WHERE mtf.{time_close_col} + INTERVAL '{horizon_minutes} minutes' > NOW()
  AND so.outcome_{horizon} IS NOT NULL
```

Expected result: **0 rows**

Elke non-zero waarde = LOOKAHEAD BIAS VIOLATION

---

## ATR-Relative Binning

Outcomes worden gediscretiseerd naar 7 bins op basis van ATR-normalized returns:

```
Return / ATR         → Outcome    → Label
─────────────────────────────────────────────
< -2.0              → -3         → Strong Bearish
-2.0 to -1.0        → -2         → Bearish
-1.0 to -0.5        → -1         → Slight Bearish
-0.5 to +0.5        → 0          → Neutral
+0.5 to +1.0        → +1         → Slight Bullish
+1.0 to +2.0        → +2         → Bullish
> +2.0              → +3         → Strong Bullish
```

**Waarom ATR-relative?**
- Market-adaptive (volatiele periodes krijgen hogere thresholds)
- Asset-agnostic (werkt voor BTC, ETH, altcoins)
- Robust tegen regime changes

**Thresholds:** `[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]` (van `target_generator.py`)

---

## Resume Capability

De backfill kan veilig worden onderbroken en hervat:

### Hoe Werkt Het?

```sql
WHERE outcome_{horizon} IS NULL  -- Skip already processed rows
```

### Workflow:

1. **Start backfill:**
   ```bash
   python scripts/outcome_backfill.py --all
   ```

2. **Onderbreek (Ctrl+C):**
   ```
   ^C
   [INFO] Interrupted - 50,000 rows processed
   ```

3. **Hervat:**
   ```bash
   python scripts/outcome_backfill.py --all
   ```
   → Begint automatisch waar gestopt
   → Geen duplicates
   → Geen data loss

---

## GPU Acceleration

### Requirement

- CUDA-enabled GPU
- CuPy installed: `pip install cupy-cuda12x`
- Binnen QBN Docker container: **CUDA is al geïnstalleerd** ✅

### Performance

**Vectorized Operations:**
- Return calculation: `(future_close - close) / close * 100`
- ATR normalization: `return / ATR`
- Binning: Vectorized conditions op GPU

**Verwachte Speedup:**
- GPU mode: ~5-10x sneller dan CPU
- Batch processing: 50,000 rows per batch
- GPU batch: 100,000 rows (configurable)

### CPU Fallback

Als GPU niet beschikbaar:
```bash
python scripts/outcome_backfill.py --no-gpu
```

De code heeft automatic fallback naar numpy (CPU).

---

## Validation Checks Explained

### 1. Completeness Check

**Doel:** Verifieer dat alle eligible rows outcomes hebben.

**Query:**
```sql
SELECT COUNT(*) as missing
FROM qbn.ml_multi_timeframe_signals
WHERE time < NOW() - INTERVAL '{horizon}'
  AND outcome_{horizon} IS NULL
  AND atr_14 IS NOT NULL
```

**Expected:** missing = 0 (of < 5% van totaal)

---

### 2. Lookahead Bias Detection

**Doel:** Detecteer of er outcomes zijn voor te recente rows.

**Query:**
```sql
SELECT COUNT(*) as violations
FROM qbn.ml_multi_timeframe_signals
WHERE outcome_{horizon} IS NOT NULL
  AND time >= NOW() - INTERVAL '{horizon}'
```

**Expected:** violations = **0** (CRITICAL!)

**Als violations > 0:**
- ❌ Model heeft toekomstige informatie gezien
- ❌ Training data is gecompromitteerd
- ❌ Backtest resultaten zijn niet betrouwbaar
- ❌ FIX IMMEDIATELY

---

### 3. Distribution Check

**Doel:** Verifieer redelijke verdeling over bins.

**Expected:**
- Niet 100% in één bin
- Niet 0% in alle bins behalve één
- Critical bins (-2, -1, 0, 1, 2) hebben data
- Extremes (-3, +3) mogen minder voorkomen

**Voorbeeld Goede Verdeling:**
```
Outcome    Count      Percentage
-3         1,250      2.5%
-2         8,400      16.8%
-1         12,600     25.2%
 0         13,200     26.4%
+1         11,800     23.6%
+2         2,550      5.1%
+3         200        0.4%
```

---

### 4. ATR Correlation Check

**Doel:** Verifieer dat binning correct is.

**Query:**
```sql
SELECT
    outcome_{horizon},
    AVG(return_{horizon}_pct / atr_at_signal) as avg_atr_normalized
FROM qbn.ml_multi_timeframe_signals
WHERE outcome_{horizon} IS NOT NULL
GROUP BY outcome_{horizon}
```

**Expected:**
- Outcome -3: avg_atr_normalized < -2.0
- Outcome +3: avg_atr_normalized > +2.0
- Outcome 0: avg_atr_normalized ≈ 0.0

**Als niet correct:**
- Binning logic is fout
- ATR calculation is fout
- Data integrity issue

---

## Workflow: Complete Backfill

### Step 1: Check Status

```bash
python scripts/outcome_backfill.py --status
```

Output:
```
Total records:     125,000
Filled:
  outcome_1h:      98,000 (78.4%)
  outcome_4h:      85,000 (68.0%)
  outcome_1d:      60,000 (48.0%)
Processable (eligible but missing):
  outcome_1h:      15,000
  outcome_4h:      28,000
  outcome_1d:      52,000
```

---

### Step 2: Run Backfill

```bash
# Test met één asset eerst
python scripts/outcome_backfill.py --asset BTCUSDT --horizon 1h

# Als succesvol: full backfill
python scripts/outcome_backfill.py --all
```

Output:
```
[INFO] ✅ GPU mode enabled (CuPy)
[INFO] Processing all 25 assets
...
[INFO] [1h] Batch 1: 50,000 rows
[INFO] [1h] Batch 1 complete - Total: 50,000
...
[INFO] ✅ Backfill complete - Elapsed time: 0:15:32
```

---

### Step 3: Validate

```bash
python scripts/validate_outcome_backfill.py --all
```

Output:
```
✅ PASS: Completeness
✅ PASS: Lookahead Bias Detection
✅ PASS: Distribution
✅ PASS: ATR Correlation

✅ ALL VALIDATIONS PASSED - Safe to proceed with training
```

---

### Step 4: Spot Check Database

```sql
-- Check fills per asset
SELECT
    asset_id,
    COUNT(*) FILTER (WHERE outcome_1h IS NOT NULL) as h1_filled,
    COUNT(*) FILTER (WHERE outcome_4h IS NOT NULL) as h4_filled,
    COUNT(*) FILTER (WHERE outcome_1d IS NOT NULL) as d1_filled
FROM qbn.ml_multi_timeframe_signals
GROUP BY asset_id
ORDER BY asset_id;

-- Check distribution
SELECT outcome_1h, COUNT(*)
FROM qbn.ml_multi_timeframe_signals
WHERE outcome_1h IS NOT NULL
GROUP BY outcome_1h
ORDER BY outcome_1h;
```

---

## Troubleshooting

### Problem: "CuPy not available"

**Solution:**
```bash
# Binnen QBN Docker container
pip install cupy-cuda12x

# Of gebruik CPU fallback
python scripts/outcome_backfill.py --all --no-gpu
```

---

### Problem: "Lookahead bias violations detected"

**Severity:** 🔴 CRITICAL

**Cause:** Recent rows hebben outcomes gekregen

**Solution:**
1. Stop alle training immediately
2. Identify violating rows:
   ```sql
   SELECT time, asset_id, outcome_1h
   FROM qbn.ml_multi_timeframe_signals
   WHERE outcome_1h IS NOT NULL
     AND time >= NOW() - INTERVAL '60 minutes'
   LIMIT 20;
   ```
3. Clear violating outcomes:
   ```sql
   UPDATE qbn.ml_multi_timeframe_signals
   SET outcome_1h = NULL
   WHERE time >= NOW() - INTERVAL '60 minutes';
   ```
4. Re-run backfill
5. Re-validate

---

### Problem: "Low completeness (<95%)"

**Cause:** Missing future price data

**Solution:**
1. Check if sufficient time has passed:
   - 1h horizon: need data from 1h ago
   - 1d horizon: need data from 24h ago

2. Check klines data availability:
   ```sql
   SELECT MAX(time) FROM kfl.klines_raw WHERE asset_id = 1 AND interval_min = 1;
   ```

3. Wait for more data or accept lower coverage

---

### Problem: "Distribution extremely skewed"

**Cause:** Market regime or data quality issue

**Solution:**
1. Check asset:
   ```bash
   python scripts/validate_outcome_backfill.py --asset BTCUSDT
   ```

2. Verify ATR calculations:
   ```sql
   SELECT AVG(atr_14), MIN(atr_14), MAX(atr_14)
   FROM qbn.ml_multi_timeframe_signals
   WHERE asset_id = 'BTCUSDT';
   ```

3. Check for outliers in returns:
   ```sql
   SELECT time, return_1h_pct, atr_at_signal, outcome_1h
   FROM qbn.ml_multi_timeframe_signals
   WHERE asset = 'BTCUSDT'
     AND ABS(return_1h_pct) > 10  -- Large moves
   ORDER BY ABS(return_1h_pct) DESC
   LIMIT 20;
   ```

---

## Database Schema Reference

### Table: qbn.ml_multi_timeframe_signals

**Outcome Columns:**
```sql
outcome_1h      SMALLINT,    -- -3 to +3
outcome_4h      SMALLINT,    -- -3 to +3
outcome_1d      SMALLINT,    -- -3 to +3
```

**Return Columns:**
```sql
return_1h_pct   REAL,        -- Percentage return
return_4h_pct   REAL,
return_1d_pct   REAL,
```

**ATR Column:**
```sql
atr_at_signal   REAL,        -- ATR percentage at signal time
```

**Constraints:**
```sql
CHECK (outcome_1h >= -3 AND outcome_1h <= 3)
CHECK (outcome_4h >= -3 AND outcome_4h <= 3)
CHECK (outcome_1d >= -3 AND outcome_1d <= 3)
```

**Indexes:**
```sql
-- For backfill queries
CREATE INDEX idx_mtf_outcome_1h_null ON qbn.ml_multi_timeframe_signals (time)
WHERE outcome_1h IS NULL;

-- For training queries
CREATE INDEX idx_mtf_outcome_1h_not_null ON qbn.ml_multi_timeframe_signals (asset_id, time)
WHERE outcome_1h IS NOT NULL;
```

---

## Performance Benchmarks

**Test Environment:**
- GPU: NVIDIA RTX 3090
- Assets: 25 crypto pairs
- Total rows: 5M
- Timeframe: 1h

**Results:**

| Operation | GPU Time | CPU Time | Speedup |
|-----------|----------|----------|---------|
| Fetch batch (50k rows) | 2.1s | 2.1s | 1x |
| Calculate outcomes | 0.3s | 2.8s | 9.3x |
| Bulk update | 1.2s | 1.2s | 1x |
| **Total per batch** | **3.6s** | **6.1s** | **1.7x** |

**Full Backfill (5M rows):**
- GPU: ~6 minutes
- CPU: ~10 minutes

Note: Speedup is batch-size dependent. Larger batches → higher speedup.

---

## Next Steps After Backfill

1. ✅ **Verify completeness**: `--status` shows >95% coverage
2. ✅ **Validate correctness**: All validation checks pass
3. ➡️ **PvA 1.4**: QBN_v3 Training Pipeline
4. ➡️ **PvA 1.5**: QBN_v3 Inference Integration

---

## Support

**Issues:**
- Check validation output carefully
- Review database logs: `/var/log/postgresql/`
- Check GPU logs: `nvidia-smi`

**Contact:**
- Project documentation: `f:\Containers\QBN\.docs\`
- Migration plans: `f:\Containers\QBN\.docs\v2_migration_plans\`

---

**Version:** 2.0
**Last Updated:** 2026-01-06
**Part of:** QBN_v3 Migration (Phase 2.5 - MTF Outcome Architecture)
