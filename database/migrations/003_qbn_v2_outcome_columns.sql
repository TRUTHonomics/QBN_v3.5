-- Migration: 003_qbn_v2_outcome_columns.sql
-- Purpose: Add outcome target columns for multi-horizon predictions
-- Date: 2025-12-10
-- Priority: CRITICAL
-- Description: Add outcome columns (1h, 4h, 1d), raw return columns, and ATR value to hypertable and cache

-- ============================================================================
-- PART 1: Add columns to main hypertable (qbn.ml_multi_timeframe_signals)
-- ============================================================================

BEGIN;

-- Add outcome bin columns (-3 to +3, ATR-relative)
ALTER TABLE qbn.ml_multi_timeframe_signals
ADD COLUMN IF NOT EXISTS outcome_1h SMALLINT,
ADD COLUMN IF NOT EXISTS outcome_4h SMALLINT,
ADD COLUMN IF NOT EXISTS outcome_1d SMALLINT;

-- Add raw return percentage columns for analysis
ALTER TABLE qbn.ml_multi_timeframe_signals
ADD COLUMN IF NOT EXISTS return_1h_pct REAL,
ADD COLUMN IF NOT EXISTS return_4h_pct REAL,
ADD COLUMN IF NOT EXISTS return_1d_pct REAL;

-- Add ATR value at signal time (for return normalization)
ALTER TABLE qbn.ml_multi_timeframe_signals
ADD COLUMN IF NOT EXISTS atr_at_signal REAL;

-- Add constraints to ensure outcome values are within valid range
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_outcome_1h') THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals
        ADD CONSTRAINT chk_outcome_1h CHECK (outcome_1h IS NULL OR (outcome_1h >= -3 AND outcome_1h <= 3));
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_outcome_4h') THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals
        ADD CONSTRAINT chk_outcome_4h CHECK (outcome_4h IS NULL OR (outcome_4h >= -3 AND outcome_4h <= 3));
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_outcome_1d') THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals
        ADD CONSTRAINT chk_outcome_1d CHECK (outcome_1d IS NULL OR (outcome_1d >= -3 AND outcome_1d <= 3));
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_atr_positive') THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals
        ADD CONSTRAINT chk_atr_positive CHECK (atr_at_signal IS NULL OR atr_at_signal > 0);
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- PART 2: Add columns to cache table (qbn.ml_multi_timeframe_signals_cache)
-- ============================================================================

BEGIN;

-- Add outcome bin columns to cache table
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
ADD COLUMN IF NOT EXISTS outcome_1h SMALLINT,
ADD COLUMN IF NOT EXISTS outcome_4h SMALLINT,
ADD COLUMN IF NOT EXISTS outcome_1d SMALLINT;

-- Add raw return percentage columns to cache table
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
ADD COLUMN IF NOT EXISTS return_1h_pct REAL,
ADD COLUMN IF NOT EXISTS return_4h_pct REAL,
ADD COLUMN IF NOT EXISTS return_1d_pct REAL;

-- Add ATR value to cache table
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
ADD COLUMN IF NOT EXISTS atr_at_signal REAL;

-- Add constraints to cache table (same as hypertable)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_cache_outcome_1h') THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals_cache
        ADD CONSTRAINT chk_cache_outcome_1h CHECK (outcome_1h IS NULL OR (outcome_1h >= -3 AND outcome_1h <= 3));
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_cache_outcome_4h') THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals_cache
        ADD CONSTRAINT chk_cache_outcome_4h CHECK (outcome_4h IS NULL OR (outcome_4h >= -3 AND outcome_4h <= 3));
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_cache_outcome_1d') THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals_cache
        ADD CONSTRAINT chk_cache_outcome_1d CHECK (outcome_1d IS NULL OR (outcome_1d >= -3 AND outcome_1d <= 3));
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_cache_atr_positive') THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals_cache
        ADD CONSTRAINT chk_cache_atr_positive CHECK (atr_at_signal IS NULL OR atr_at_signal > 0);
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES (for immediate post-migration check)
-- ============================================================================

-- Verify columns exist in hypertable
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'ml_multi_timeframe_signals'
AND column_name IN ('outcome_1h', 'outcome_4h', 'outcome_1d', 'return_1h_pct', 'return_4h_pct', 'return_1d_pct', 'atr_at_signal')
ORDER BY column_name;

-- Verify columns exist in cache table
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'ml_multi_timeframe_signals_cache'
AND column_name IN ('outcome_1h', 'outcome_4h', 'outcome_1d', 'return_1h_pct', 'return_4h_pct', 'return_1d_pct', 'atr_at_signal')
ORDER BY column_name;

-- Verify constraints
SELECT conname, contype, pg_get_constraintdef(oid) as definition
FROM pg_constraint
WHERE conrelid IN ('qbn.ml_multi_timeframe_signals'::regclass, 'qbn.ml_multi_timeframe_signals_cache'::regclass)
AND conname LIKE '%outcome%' OR conname LIKE '%atr%'
ORDER BY conname;

-- Verify all values are NULL initially (before backfill)
SELECT
    COUNT(*) as total_rows,
    COUNT(outcome_1h) as outcome_1h_count,
    COUNT(outcome_4h) as outcome_4h_count,
    COUNT(outcome_1d) as outcome_1d_count,
    COUNT(atr_at_signal) as atr_count
FROM qbn.ml_multi_timeframe_signals;

-- Expected: all outcome and ATR counts should be 0

-- ============================================================================
-- NOTES
-- ============================================================================

-- Outcome Classification:
-- -3: Strong_Bearish (< -1.5 ATR)
-- -2: Moderate_Bearish (-1.5 to -0.5 ATR)
-- -1: Weak_Bearish (-0.5 to -0.1 ATR)
--  0: Neutral (-0.1 to +0.1 ATR)
-- +1: Weak_Bullish (+0.1 to +0.5 ATR)
-- +2: Moderate_Bullish (+0.5 to +1.5 ATR)
-- +3: Strong_Bullish (> +1.5 ATR)

-- Timing:
-- - Estimated execution time: 20-30 minutes (depending on table size)
-- - Requires decompression of recent chunks if compressed
-- - Execute during off-hours (02:00-05:00 UTC) to minimize lock contention

-- Next Steps:
-- 1. Run 003_qbn_v2_signal_classification.sql (create signal classification table)
-- 2. Run 003_qbn_v2_indexes.sql (create performance indexes)
-- 3. Execute Phase 1.3 outcome backfill script to populate historical values
