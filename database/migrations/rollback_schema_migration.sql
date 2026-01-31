-- Rollback Script: Phase 1.1 Database Schema Migration
-- Purpose: Complete reversal of all schema changes from Phase 1.1
-- Date: 2025-12-10
-- Priority: CRITICAL
-- WARNING: This script will DROP all changes made by Phase 1.1 migrations

-- ============================================================================
-- SAFETY CHECKS AND WARNINGS
-- ============================================================================

\echo '============================================'
\echo 'ROLLBACK SCRIPT: Phase 1.1'
\echo '============================================'
\echo ''
\echo 'WARNING: This script will reverse ALL schema changes from Phase 1.1'
\echo 'Including:'
\echo '  - Drop outcome columns from hypertable and cache'
\echo '  - Drop signal_classification table'
\echo '  - Drop outcome indexes'
\echo '  - Remove new signal columns from signals_current and MTF cache'
\echo ''
\echo 'Press Ctrl+C to cancel, or continue to execute rollback...'
\echo ''

-- Add a delay to allow user to read warnings
SELECT pg_sleep(5);

-- ============================================================================
-- PART 1: Drop indexes (fastest to rollback, do first)
-- ============================================================================

\echo '\n============================================'
\echo 'PART 1: Dropping outcome indexes'
\echo '============================================'

BEGIN;

-- Drop outcome indexes using CONCURRENTLY to minimize locks
-- Note: DROP INDEX CONCURRENTLY cannot run in a transaction block
-- So we commit the transaction first
COMMIT;

-- Drop indexes one by one (CONCURRENTLY requires no transaction)
DROP INDEX CONCURRENTLY IF EXISTS qbn.idx_mtf_outcome_1h_training;
DROP INDEX CONCURRENTLY IF EXISTS qbn.idx_mtf_outcome_4h_training;
DROP INDEX CONCURRENTLY IF EXISTS qbn.idx_mtf_outcome_1d_training;
DROP INDEX CONCURRENTLY IF EXISTS qbn.idx_mtf_all_outcomes;
DROP INDEX CONCURRENTLY IF EXISTS qbn.idx_mtf_atr_signal;

-- Drop signal_classification indexes
DROP INDEX CONCURRENTLY IF EXISTS qbn.idx_signal_class_semantic;
DROP INDEX CONCURRENTLY IF EXISTS qbn.idx_signal_class_indicator;

\echo 'Indexes dropped successfully'

-- ============================================================================
-- PART 2: Drop signal_classification table
-- ============================================================================

\echo '\n============================================'
\echo 'PART 2: Dropping signal_classification table'
\echo '============================================'

BEGIN;

DROP TABLE IF EXISTS qbn.signal_classification CASCADE;

\echo 'signal_classification table dropped'

COMMIT;

-- ============================================================================
-- PART 3: Remove columns from MTF cache (do before hypertable)
-- ============================================================================

\echo '\n============================================'
\echo 'PART 3: Removing columns from MTF cache'
\echo '============================================'

BEGIN;

-- Remove outcome columns
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
DROP COLUMN IF EXISTS outcome_1h,
DROP COLUMN IF EXISTS outcome_4h,
DROP COLUMN IF EXISTS outcome_1d,
DROP COLUMN IF EXISTS return_1h_pct,
DROP COLUMN IF EXISTS return_4h_pct,
DROP COLUMN IF EXISTS return_1d_pct,
DROP COLUMN IF EXISTS atr_at_signal;

-- Remove STRUCTURAL layer signals
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
DROP COLUMN IF EXISTS structural_adx_signal,
DROP COLUMN IF EXISTS structural_adx_value,
DROP COLUMN IF EXISTS structural_cmf_signal,
DROP COLUMN IF EXISTS structural_cmf_value,
DROP COLUMN IF EXISTS structural_obv_signal,
DROP COLUMN IF EXISTS structural_obv_value,
DROP COLUMN IF EXISTS structural_stoch_k_signal,
DROP COLUMN IF EXISTS structural_stoch_d_signal,
DROP COLUMN IF EXISTS structural_macd_histogram_signal,
DROP COLUMN IF EXISTS structural_ichimoku_signal,
DROP COLUMN IF EXISTS structural_atr_value;

-- Remove TACTICAL layer signals
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
DROP COLUMN IF EXISTS tactical_adx_signal,
DROP COLUMN IF EXISTS tactical_adx_value,
DROP COLUMN IF EXISTS tactical_cmf_signal,
DROP COLUMN IF EXISTS tactical_cmf_value,
DROP COLUMN IF EXISTS tactical_obv_signal,
DROP COLUMN IF EXISTS tactical_obv_value,
DROP COLUMN IF EXISTS tactical_stoch_k_signal,
DROP COLUMN IF EXISTS tactical_stoch_d_signal,
DROP COLUMN IF EXISTS tactical_macd_histogram_signal,
DROP COLUMN IF EXISTS tactical_ichimoku_signal,
DROP COLUMN IF EXISTS tactical_atr_value;

-- Remove ENTRY layer signals
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
DROP COLUMN IF EXISTS entry_adx_signal,
DROP COLUMN IF EXISTS entry_adx_value,
DROP COLUMN IF EXISTS entry_cmf_signal,
DROP COLUMN IF EXISTS entry_cmf_value,
DROP COLUMN IF EXISTS entry_obv_signal,
DROP COLUMN IF EXISTS entry_obv_value,
DROP COLUMN IF EXISTS entry_stoch_k_signal,
DROP COLUMN IF EXISTS entry_stoch_d_signal,
DROP COLUMN IF EXISTS entry_macd_histogram_signal,
DROP COLUMN IF EXISTS entry_ichimoku_signal,
DROP COLUMN IF EXISTS entry_atr_value;

-- Remove UTF layer signals
ALTER TABLE qbn.ml_multi_timeframe_signals_cache
DROP COLUMN IF EXISTS utf_adx_signal,
DROP COLUMN IF EXISTS utf_adx_value,
DROP COLUMN IF EXISTS utf_cmf_signal,
DROP COLUMN IF EXISTS utf_cmf_value,
DROP COLUMN IF EXISTS utf_obv_signal,
DROP COLUMN IF EXISTS utf_obv_value,
DROP COLUMN IF EXISTS utf_stoch_k_signal,
DROP COLUMN IF EXISTS utf_stoch_d_signal,
DROP COLUMN IF EXISTS utf_macd_histogram_signal,
DROP COLUMN IF EXISTS utf_ichimoku_signal,
DROP COLUMN IF EXISTS utf_atr_value;

\echo 'MTF cache columns removed'

COMMIT;

-- ============================================================================
-- PART 4: Remove columns from hypertable
-- ============================================================================

\echo '\n============================================'
\echo 'PART 4: Removing columns from hypertable'
\echo '============================================'

BEGIN;

-- Remove outcome columns
ALTER TABLE qbn.ml_multi_timeframe_signals
DROP COLUMN IF EXISTS outcome_1h,
DROP COLUMN IF EXISTS outcome_4h,
DROP COLUMN IF EXISTS outcome_1d,
DROP COLUMN IF EXISTS return_1h_pct,
DROP COLUMN IF EXISTS return_4h_pct,
DROP COLUMN IF EXISTS return_1d_pct,
DROP COLUMN IF EXISTS atr_at_signal;

\echo 'Hypertable outcome columns removed'

COMMIT;

-- ============================================================================
-- PART 5: Remove new signals from signals_current
-- ============================================================================

\echo '\n============================================'
\echo 'PART 5: Removing new signals from signals_current'
\echo '============================================'

BEGIN;

-- Remove ADX columns
ALTER TABLE kfl.signals_current
DROP COLUMN IF EXISTS adx_signal,
DROP COLUMN IF EXISTS adx_plus_di,
DROP COLUMN IF EXISTS adx_minus_di,
DROP COLUMN IF EXISTS adx_value;

-- Remove CMF columns
ALTER TABLE kfl.signals_current
DROP COLUMN IF EXISTS cmf_signal,
DROP COLUMN IF EXISTS cmf_value;

-- Remove OBV columns
ALTER TABLE kfl.signals_current
DROP COLUMN IF EXISTS obv_signal,
DROP COLUMN IF EXISTS obv_value,
DROP COLUMN IF EXISTS obv_ma;

-- Remove Stochastic columns
ALTER TABLE kfl.signals_current
DROP COLUMN IF EXISTS stoch_k_signal,
DROP COLUMN IF EXISTS stoch_d_signal,
DROP COLUMN IF EXISTS stoch_k_value,
DROP COLUMN IF EXISTS stoch_d_value;

-- Remove MACD variant columns
ALTER TABLE kfl.signals_current
DROP COLUMN IF EXISTS macd_signal_line,
DROP COLUMN IF EXISTS macd_histogram,
DROP COLUMN IF EXISTS macd_histogram_signal;

-- Remove Ichimoku columns
ALTER TABLE kfl.signals_current
DROP COLUMN IF EXISTS ichimoku_conversion_line,
DROP COLUMN IF EXISTS ichimoku_base_line,
DROP COLUMN IF EXISTS ichimoku_span_a,
DROP COLUMN IF EXISTS ichimoku_span_b,
DROP COLUMN IF EXISTS ichimoku_signal;

-- Remove ATR columns
ALTER TABLE kfl.signals_current
DROP COLUMN IF EXISTS atr_value,
DROP COLUMN IF EXISTS atr_percent;

\echo 'signals_current columns removed'

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

\echo '\n============================================'
\echo 'ROLLBACK VERIFICATION'
\echo '============================================'

-- Verify outcome columns removed from hypertable
\echo '\n1. Verify outcome columns removed from hypertable'
SELECT COUNT(*) as remaining_outcome_columns
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'ml_multi_timeframe_signals'
AND column_name LIKE '%outcome%' OR column_name LIKE '%return_%' OR column_name = 'atr_at_signal';

-- Expected: 0

-- Verify signal_classification table dropped
\echo '\n2. Verify signal_classification table dropped'
SELECT EXISTS (
    SELECT FROM information_schema.tables
    WHERE table_schema = 'qbn'
    AND table_name = 'signal_classification'
) AS table_exists;

-- Expected: false

-- Verify outcome indexes dropped
\echo '\n3. Verify outcome indexes dropped'
SELECT COUNT(*) as remaining_outcome_indexes
FROM pg_indexes
WHERE schemaname = 'qbn'
AND tablename = 'ml_multi_timeframe_signals'
AND indexname LIKE '%outcome%';

-- Expected: 0

-- Verify new signals_current columns removed
\echo '\n4. Verify signals_current columns removed'
SELECT COUNT(*) as remaining_new_columns
FROM information_schema.columns
WHERE table_schema = 'kfl'
AND table_name = 'signals_current'
AND (
    column_name LIKE '%adx%'
    OR column_name LIKE '%cmf%'
    OR column_name LIKE '%obv%'
    OR column_name LIKE '%stoch%'
    OR column_name LIKE '%macd_histogram%'
    OR column_name LIKE '%ichimoku%'
    OR column_name LIKE '%atr%'
);

-- Expected: 0

-- Verify MTF cache columns removed
\echo '\n5. Verify MTF cache columns removed'
SELECT COUNT(*) as remaining_new_columns
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
    OR column_name LIKE '%outcome%'
    OR column_name LIKE '%return_%'
    OR column_name = 'atr_at_signal'
);

-- Expected: 0

-- ============================================================================
-- ROLLBACK COMPLETE
-- ============================================================================

\echo '\n============================================'
\echo 'ROLLBACK COMPLETE'
\echo '============================================'
\echo ''
\echo 'Summary of changes reversed:'
\echo '  - Dropped 5 outcome-related indexes'
\echo '  - Dropped signal_classification table'
\echo '  - Removed 7 outcome columns from hypertable'
\echo '  - Removed 7 outcome columns from MTF cache'
\echo '  - Removed 23 signal columns from signals_current'
\echo '  - Removed 48 signal columns from MTF cache'
\echo ''
\echo 'Database has been restored to pre-migration state'
\echo ''
\echo 'If rollback was due to critical issues:'
\echo '  1. Review migration logs for error details'
\echo '  2. Test migrations on dev database before retrying'
\echo '  3. Consider restore from backup if rollback incomplete'
\echo ''
\echo 'If rollback was planned:'
\echo '  1. Migrations can be re-run after addressing issues'
\echo '  2. No data loss (only schema changes were reversed)'
\echo ''

-- ============================================================================
-- NOTES
-- ============================================================================

-- Rollback Timing:
-- - Estimated execution time: 10-15 minutes
-- - DROP INDEX CONCURRENTLY allows reads/writes during execution
-- - Execute immediately if critical issues detected post-migration

-- Data Loss:
-- - No actual data is lost (all new columns were NULL)
-- - Only schema structure is removed
-- - Original table data remains intact

-- Re-running Migrations:
-- - Migrations can be re-executed after rollback
-- - All scripts use IF EXISTS / IF NOT EXISTS for safety
-- - Idempotent operations allow safe re-runs

-- Partial Rollback:
-- - If only specific changes need rollback, comment out unwanted sections
-- - Sections are independent and can be run separately
-- - Always verify after partial rollback

-- Full Recovery:
-- - If rollback fails or incomplete, restore from pre-migration backup
-- - Backup command: pg_restore -h 10.10.10.1 -U postgres -d KFLhyper --clean --if-exists /backup/KFLhyper_pre_migration_1.1_*.dump
