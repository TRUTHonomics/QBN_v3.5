-- Post-Migration Validation Queries: Phase 1.1
-- Purpose: Comprehensive verification of schema changes and data integrity
-- Date: 2025-12-10
-- Priority: HIGH
-- Usage: Copy and paste queries into pgAdmin Query Tool

-- ============================================================================
-- SECTION 1: Schema Verification
-- ============================================================================

-- SECTION 1: Schema Verification
-- =============================================

-- 1.1: Verify outcome columns in hypertable
-- Expected: 7 columns (3 outcomes, 3 returns, 1 ATR)
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'ml_multi_timeframe_signals'
AND column_name IN ('outcome_1h', 'outcome_4h', 'outcome_1d', 'return_1h_pct', 'return_4h_pct', 'return_1d_pct', 'atr_at_signal')
ORDER BY column_name;

-- 1.2: Verify outcome columns in cache table
-- Expected: 7 columns
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'ml_multi_timeframe_signals_cache'
AND column_name IN ('outcome_1h', 'outcome_4h', 'outcome_1d', 'return_1h_pct', 'return_4h_pct', 'return_1d_pct', 'atr_at_signal')
ORDER BY column_name;

-- 1.3: Verify signal_classification table exists
-- Table structure check:
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'signal_classification'
ORDER BY ordinal_position;

-- 1.4: Verify new signals_current columns
-- Expected: 23 new columns
SELECT COUNT(*) as new_columns_count
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

-- 1.5: Verify MTF cache expansion
-- Expected: 48 new columns
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

-- ============================================================================
-- SECTION 2: Constraint Verification
-- ============================================================================

-- SECTION 2: Constraint Verification
-- =============================================

-- 2.1: Verify outcome range constraints
-- Expected: 3 constraints (chk_outcome_1h, chk_outcome_4h, chk_outcome_1d)
SELECT
    conname AS constraint_name,
    pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'qbn.ml_multi_timeframe_signals'::regclass
AND conname LIKE '%outcome%'
ORDER BY conname;

-- 2.2: Verify ATR positive constraint
-- Expected: 2 constraints (one per table)
SELECT
    conname AS constraint_name,
    pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid IN ('qbn.ml_multi_timeframe_signals'::regclass, 'qbn.ml_multi_timeframe_signals_cache'::regclass)
AND conname LIKE '%atr%'
ORDER BY conname;

-- 2.3: Verify signal_classification constraints
-- Expected: Primary key + 2 CHECK constraints (semantic_class, polarity)
SELECT
    conname AS constraint_name,
    contype AS constraint_type,
    pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'qbn.signal_classification'::regclass
ORDER BY contype, conname;

-- 2.4: Test constraint enforcement (outcome range)
-- Attempting to insert invalid outcome value (should fail)
DO $$
BEGIN
    -- Insert with required NOT NULL columns plus outcome_1h to test constraint
    -- REASON: time_d is NOT NULL, so we need to provide it
    INSERT INTO qbn.ml_multi_timeframe_signals
    (asset_id, time, time_d, outcome_1h)
    VALUES (999999, NOW(), NOW(), 5); -- Invalid: outcome > 3

    RAISE EXCEPTION 'ERROR: Constraint validation failed - invalid value was accepted!';
EXCEPTION
    WHEN check_violation THEN
        RAISE NOTICE 'SUCCESS: Constraint correctly rejected invalid value';
    WHEN OTHERS THEN
        RAISE NOTICE 'WARNING: Unexpected error during constraint test: %', SQLERRM;
END$$;

-- ============================================================================
-- SECTION 3: Index Verification
-- ============================================================================

-- SECTION 3: Index Verification
-- =============================================

-- 3.1: List all outcome-related indexes
-- Expected: 4 indexes (3 single horizon + 1 all horizons)
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'qbn'
AND tablename = 'ml_multi_timeframe_signals'
AND indexname LIKE '%outcome%'
ORDER BY indexname;

-- 3.2: Verify indexes are valid and ready
-- Expected: All indexes should be valid=true and ready=true
SELECT
    schemaname || '.' || tablename AS table_name,
    indexname,
    indisvalid AS is_valid,
    indisready AS is_ready
FROM pg_index
JOIN pg_class ON pg_class.oid = indexrelid
JOIN pg_namespace ON pg_namespace.oid = relnamespace
JOIN pg_indexes ON pg_indexes.indexname = pg_class.relname
WHERE schemaname = 'qbn'
AND tablename = 'ml_multi_timeframe_signals'
AND indexname LIKE '%outcome%'
ORDER BY indexname;

-- 3.3: Index sizes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(schemaname || '.' || indexname)) AS index_size
FROM pg_indexes
WHERE schemaname = 'qbn'
AND tablename = 'ml_multi_timeframe_signals'
AND indexname LIKE '%outcome%'
ORDER BY pg_relation_size(schemaname || '.' || indexname) DESC;

-- 3.4: Test index usage with EXPLAIN
-- Expected: Should use Index Scan on idx_mtf_outcome_1h_training
EXPLAIN (ANALYZE, BUFFERS)
SELECT asset_id, time, outcome_1h
FROM qbn.ml_multi_timeframe_signals
WHERE asset_id = 1
AND outcome_1h IS NOT NULL
ORDER BY time DESC
LIMIT 100;

-- ============================================================================
-- SECTION 4: Data Integrity Checks
-- ============================================================================

-- SECTION 4: Data Integrity Checks
-- =============================================

-- 4.1: Verify all outcome values are NULL initially
-- Expected: All outcome/return/ATR counts should be 0
SELECT
    COUNT(*) as total_rows,
    COUNT(outcome_1h) as outcome_1h_count,
    COUNT(outcome_4h) as outcome_4h_count,
    COUNT(outcome_1d) as outcome_1d_count,
    COUNT(return_1h_pct) as return_1h_count,
    COUNT(return_4h_pct) as return_4h_count,
    COUNT(return_1d_pct) as return_1d_count,
    COUNT(atr_at_signal) as atr_count
FROM qbn.ml_multi_timeframe_signals;

-- 4.2: Verify signal_classification table is empty initially
-- Expected: 0 rows (will be populated in Phase 1.2)
SELECT COUNT(*) as row_count FROM qbn.signal_classification;

-- 4.3: Verify cache table has matching structure
-- Note: Cache may have fewer columns (no partition keys)
SELECT
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals') as hypertable_columns,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals_cache') as cache_columns;

-- 4.4: Table sizes after migration
SELECT
    schemaname || '.' || tablename AS table_name,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) AS table_size,
    pg_size_pretty(pg_indexes_size(schemaname || '.' || tablename)) AS indexes_size
FROM pg_tables
WHERE schemaname IN ('qbn', 'kfl')
AND tablename IN ('ml_multi_timeframe_signals', 'ml_multi_timeframe_signals_cache', 'signals_current', 'signal_classification')
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;

-- ============================================================================
-- SECTION 5: Performance Tests
-- ============================================================================

-- SECTION 5: Performance Tests
-- =============================================

-- 5.1: Query performance - single asset, single horizon
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT COUNT(*)
FROM qbn.ml_multi_timeframe_signals
WHERE asset_id = 1
AND outcome_1h IS NOT NULL;

-- 5.2: Query performance - cross-horizon analysis
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT asset_id, time, outcome_1h, outcome_4h, outcome_1d
FROM qbn.ml_multi_timeframe_signals
WHERE asset_id = 1
AND outcome_1h IS NOT NULL
AND outcome_4h IS NOT NULL
AND outcome_1d IS NOT NULL
ORDER BY time DESC
LIMIT 100;

-- 5.3: Test signals_current query performance
-- REASON: Kolom heet 'time', niet 'signal_date'
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT asset_id, time, adx_signal, cmf_signal, stoch_k_signal
FROM kfl.signals_current
WHERE asset_id = 1
ORDER BY time DESC
LIMIT 100;

-- ============================================================================
-- SECTION 6: Migration Summary
-- ============================================================================

-- SECTION 6: Migration Summary
-- =============================================

-- 6.1: Total column counts
SELECT
    'ml_multi_timeframe_signals' as table_name,
    COUNT(*) as total_columns
FROM information_schema.columns
WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals'
UNION ALL
SELECT
    'ml_multi_timeframe_signals_cache',
    COUNT(*)
FROM information_schema.columns
WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals_cache'
UNION ALL
SELECT
    'signals_current',
    COUNT(*)
FROM information_schema.columns
WHERE table_schema = 'kfl' AND table_name = 'signals_current'
UNION ALL
SELECT
    'signal_classification',
    COUNT(*)
FROM information_schema.columns
WHERE table_schema = 'qbn' AND table_name = 'signal_classification';

-- 6.2: Index counts per table
SELECT
    tablename,
    COUNT(*) as index_count
FROM pg_indexes
WHERE schemaname IN ('qbn', 'kfl')
AND tablename IN ('ml_multi_timeframe_signals', 'ml_multi_timeframe_signals_cache', 'signals_current', 'signal_classification')
GROUP BY tablename
ORDER BY tablename;

-- 6.3: Constraint counts
SELECT
    conrelid::regclass AS table_name,
    COUNT(*) as constraint_count
FROM pg_constraint
WHERE conrelid IN (
    'qbn.ml_multi_timeframe_signals'::regclass,
    'qbn.ml_multi_timeframe_signals_cache'::regclass,
    'kfl.signals_current'::regclass,
    'qbn.signal_classification'::regclass
)
GROUP BY conrelid
ORDER BY conrelid::text;

-- ============================================================================
-- VALIDATION COMPLETE
-- ============================================================================

-- VALIDATION COMPLETE
-- =============================================
--
-- Next steps:
-- 1. Review all validation results above
-- 2. Verify all indexes show is_valid=true and is_ready=true
-- 3. Confirm EXPLAIN plans show Index Scan (not Seq Scan)
-- 4. Monitor database performance for 24 hours
-- 5. Proceed with Phase 1.2 (Signal Classification Mapping)
