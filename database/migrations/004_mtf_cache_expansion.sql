-- Migration: 004_mtf_cache_expansion.sql
-- Purpose: Expand MTF cache with additional technical indicators across all timeframes
-- Date: 2025-12-10
-- Priority: MEDIUM
-- Description: Add ~48 new signal columns (4 timeframes × 12 indicators) + 4 ATR values

-- ============================================================================
-- PART 1: Add STRUCTURAL layer signals (Daily, 1440m)
-- ============================================================================

BEGIN;

-- ADX signals
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
ADD COLUMN IF NOT EXISTS structural_adx_signal SMALLINT,
ADD COLUMN IF NOT EXISTS structural_adx_value REAL,

-- CMF signals
ADD COLUMN IF NOT EXISTS structural_cmf_signal SMALLINT,
ADD COLUMN IF NOT EXISTS structural_cmf_value REAL,

-- OBV signals
ADD COLUMN IF NOT EXISTS structural_obv_signal SMALLINT,
ADD COLUMN IF NOT EXISTS structural_obv_value REAL,

-- Stochastic signals
ADD COLUMN IF NOT EXISTS structural_stoch_k_signal SMALLINT,
ADD COLUMN IF NOT EXISTS structural_stoch_d_signal SMALLINT,

-- MACD histogram
ADD COLUMN IF NOT EXISTS structural_macd_histogram_signal SMALLINT,

-- Ichimoku signal
ADD COLUMN IF NOT EXISTS structural_ichimoku_signal SMALLINT,

-- ATR value
ADD COLUMN IF NOT EXISTS structural_atr_value REAL;

COMMIT;

-- ============================================================================
-- PART 2: Add TACTICAL layer signals (4H, 240m)
-- ============================================================================

BEGIN;

-- ADX signals
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
ADD COLUMN IF NOT EXISTS tactical_adx_signal SMALLINT,
ADD COLUMN IF NOT EXISTS tactical_adx_value REAL,

-- CMF signals
ADD COLUMN IF NOT EXISTS tactical_cmf_signal SMALLINT,
ADD COLUMN IF NOT EXISTS tactical_cmf_value REAL,

-- OBV signals
ADD COLUMN IF NOT EXISTS tactical_obv_signal SMALLINT,
ADD COLUMN IF NOT EXISTS tactical_obv_value REAL,

-- Stochastic signals
ADD COLUMN IF NOT EXISTS tactical_stoch_k_signal SMALLINT,
ADD COLUMN IF NOT EXISTS tactical_stoch_d_signal SMALLINT,

-- MACD histogram
ADD COLUMN IF NOT EXISTS tactical_macd_histogram_signal SMALLINT,

-- Ichimoku signal
ADD COLUMN IF NOT EXISTS tactical_ichimoku_signal SMALLINT,

-- ATR value
ADD COLUMN IF NOT EXISTS tactical_atr_value REAL;

COMMIT;

-- ============================================================================
-- PART 3: Add ENTRY layer signals (1H, 60m)
-- ============================================================================

BEGIN;

-- ADX signals
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
ADD COLUMN IF NOT EXISTS entry_adx_signal SMALLINT,
ADD COLUMN IF NOT EXISTS entry_adx_value REAL,

-- CMF signals
ADD COLUMN IF NOT EXISTS entry_cmf_signal SMALLINT,
ADD COLUMN IF NOT EXISTS entry_cmf_value REAL,

-- OBV signals
ADD COLUMN IF NOT EXISTS entry_obv_signal SMALLINT,
ADD COLUMN IF NOT EXISTS entry_obv_value REAL,

-- Stochastic signals
ADD COLUMN IF NOT EXISTS entry_stoch_k_signal SMALLINT,
ADD COLUMN IF NOT EXISTS entry_stoch_d_signal SMALLINT,

-- MACD histogram
ADD COLUMN IF NOT EXISTS entry_macd_histogram_signal SMALLINT,

-- Ichimoku signal
ADD COLUMN IF NOT EXISTS entry_ichimoku_signal SMALLINT,

-- ATR value
ADD COLUMN IF NOT EXISTS entry_atr_value REAL;

COMMIT;

-- ============================================================================
-- PART 4: Add UTF layer signals (Micro, 1m)
-- ============================================================================

BEGIN;

-- ADX signals
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
ADD COLUMN IF NOT EXISTS utf_adx_signal SMALLINT,
ADD COLUMN IF NOT EXISTS utf_adx_value REAL,

-- CMF signals
ADD COLUMN IF NOT EXISTS utf_cmf_signal SMALLINT,
ADD COLUMN IF NOT EXISTS utf_cmf_value REAL,

-- OBV signals
ADD COLUMN IF NOT EXISTS utf_obv_signal SMALLINT,
ADD COLUMN IF NOT EXISTS utf_obv_value REAL,

-- Stochastic signals
ADD COLUMN IF NOT EXISTS utf_stoch_k_signal SMALLINT,
ADD COLUMN IF NOT EXISTS utf_stoch_d_signal SMALLINT,

-- MACD histogram
ADD COLUMN IF NOT EXISTS utf_macd_histogram_signal SMALLINT,

-- Ichimoku signal
ADD COLUMN IF NOT EXISTS utf_ichimoku_signal SMALLINT,

-- ATR value
ADD COLUMN IF NOT EXISTS utf_atr_value REAL;

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES (for immediate post-migration check)
-- ============================================================================

-- Count all new columns
SELECT COUNT(*) as new_columns_count
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'ml_multi_timeframe_signals_cache'
AND (
    column_name LIKE '%_adx_%'
    OR column_name LIKE '%_cmf_%'
    OR column_name LIKE '%_obv_%'
    OR column_name LIKE '%_stoch_%'
    OR column_name LIKE '%_macd_histogram_%'
    OR column_name LIKE '%_ichimoku_%'
    OR column_name LIKE '%_atr_value'
);

-- Expected: 48 new columns (4 timeframes × 12 indicators)

-- Verify columns exist per timeframe
SELECT
    COUNT(*) FILTER (WHERE column_name LIKE 'structural_%') as structural_columns,
    COUNT(*) FILTER (WHERE column_name LIKE 'tactical_%') as tactical_columns,
    COUNT(*) FILTER (WHERE column_name LIKE 'entry_%') as entry_columns,
    COUNT(*) FILTER (WHERE column_name LIKE 'utf_%') as utf_columns
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'ml_multi_timeframe_signals_cache'
AND (
    column_name LIKE '%_adx_%'
    OR column_name LIKE '%_cmf_%'
    OR column_name LIKE '%_obv_%'
    OR column_name LIKE '%_stoch_%'
    OR column_name LIKE '%_macd_histogram_%'
    OR column_name LIKE '%_ichimoku_%'
    OR column_name LIKE '%_atr_value'
);

-- Expected: 12 columns per timeframe

-- List all new columns by timeframe
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'ml_multi_timeframe_signals_cache'
AND (
    column_name LIKE '%_adx_%'
    OR column_name LIKE '%_cmf_%'
    OR column_name LIKE '%_obv_%'
    OR column_name LIKE '%_stoch_%'
    OR column_name LIKE '%_macd_histogram_%'
    OR column_name LIKE '%_ichimoku_%'
    OR column_name LIKE '%_atr_value'
)
ORDER BY
    CASE
        WHEN column_name LIKE 'structural_%' THEN 1
        WHEN column_name LIKE 'tactical_%' THEN 2
        WHEN column_name LIKE 'entry_%' THEN 3
        WHEN column_name LIKE 'utf_%' THEN 4
    END,
    column_name;

-- Verify all values are NULL initially
SELECT
    COUNT(*) as total_rows,
    COUNT(structural_adx_signal) as structural_adx_count,
    COUNT(tactical_adx_signal) as tactical_adx_count,
    COUNT(entry_adx_signal) as entry_adx_count,
    COUNT(utf_adx_signal) as utf_adx_count,
    COUNT(structural_atr_value) as structural_atr_count,
    COUNT(tactical_atr_value) as tactical_atr_count,
    COUNT(entry_atr_value) as entry_atr_count,
    COUNT(utf_atr_value) as utf_atr_count
FROM qbn.ml_multi_timeframe_signals_cache;

-- Expected: all signal counts should be 0 initially

-- ============================================================================
-- NOTES
-- ============================================================================

-- New Columns Summary:
--
-- Per timeframe (4 total: structural, tactical, entry, utf):
-- - ADX: 2 columns (signal, value)
-- - CMF: 2 columns (signal, value)
-- - OBV: 2 columns (signal, value)
-- - Stochastic: 2 columns (K signal, D signal)
-- - MACD histogram: 1 column (histogram signal)
-- - Ichimoku: 1 column (cloud position signal)
-- - ATR: 1 column (value)
-- = 11 columns per timeframe
--
-- However, we add ATR at timeframe level (4 total)
-- Total: 4 timeframes × 11 signals + 4 ATR values = 48 columns

-- Column Naming Convention:
-- {timeframe}_{indicator}_{variant}
-- Examples:
-- - structural_adx_signal
-- - tactical_cmf_value
-- - entry_stoch_k_signal
-- - utf_atr_value

-- Signal Propagation:
-- - These columns will be populated by MTF triggers from kfl.signals_current
-- - Triggers aggregate signals across timeframes (structural, tactical, entry, utf)
-- - Values are cached for efficient BN inference

-- Timing:
-- - Estimated execution time: 5-10 minutes
-- - ALTER operations on cache table (smaller than hypertable)
-- - Execute sequentially after 004_signals_current_expansion.sql

-- Integration:
-- - MTF triggers will automatically propagate new signals to cache
-- - No changes needed to existing trigger logic (column-agnostic)
-- - BN model will access signals via cache for real-time inference

-- Next Steps:
-- 1. Verify all 48 columns created successfully
-- 2. Update MTF triggers if needed (should be automatic)
-- 3. Test signal propagation from signals_current to MTF cache
-- 4. Validate BN model can access new signals
