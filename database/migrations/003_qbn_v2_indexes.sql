-- Migration: 003_qbn_v2_indexes.sql
-- Purpose: Create performance indexes for outcome columns
-- Date: 2025-12-10
-- Priority: CRITICAL
-- Description: Proactive index creation for efficient CPT training queries

-- ============================================================================
-- IMPORTANT: TimescaleDB Hypertable Indexes
-- ============================================================================
-- Just use normal CREATE INDEX - TimescaleDB handles hypertables automatically

-- ============================================================================
-- PART 1: Partial indexes on outcome columns (WHERE outcome IS NOT NULL)
-- ============================================================================

-- Index for 1-hour outcome queries
-- Purpose: Efficient filtering for CPT training (only rows with outcomes)
CREATE INDEX IF NOT EXISTS idx_mtf_outcome_1h_training
ON qbn.ml_multi_timeframe_signals (asset_id, time, outcome_1h)
WHERE outcome_1h IS NOT NULL;

-- Index for 4-hour outcome queries
CREATE INDEX IF NOT EXISTS idx_mtf_outcome_4h_training
ON qbn.ml_multi_timeframe_signals (asset_id, time, outcome_4h)
WHERE outcome_4h IS NOT NULL;

-- Index for 1-day outcome queries
CREATE INDEX IF NOT EXISTS idx_mtf_outcome_1d_training
ON qbn.ml_multi_timeframe_signals (asset_id, time, outcome_1d)
WHERE outcome_1d IS NOT NULL;

-- ============================================================================
-- PART 2: Composite indexes for multi-horizon queries
-- ============================================================================

-- Index for queries that filter on all three outcome horizons
-- Useful for cross-horizon analysis
CREATE INDEX IF NOT EXISTS idx_mtf_all_outcomes
ON qbn.ml_multi_timeframe_signals (asset_id, time)
WHERE outcome_1h IS NOT NULL
  AND outcome_4h IS NOT NULL
  AND outcome_1d IS NOT NULL;

-- ============================================================================
-- PART 3: Index on ATR value for outcome calculation queries
-- ============================================================================

-- Index for ATR-related queries and outcome backfill validation
CREATE INDEX IF NOT EXISTS idx_mtf_atr_signal
ON qbn.ml_multi_timeframe_signals (asset_id, time, atr_at_signal)
WHERE atr_at_signal IS NOT NULL;

-- ============================================================================
-- VERIFICATION QUERIES (for immediate post-migration check)
-- ============================================================================

-- Verify all indexes were created successfully
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'qbn'
AND tablename = 'ml_multi_timeframe_signals'
AND indexname LIKE '%outcome%' OR indexname LIKE '%atr%'
ORDER BY indexname;

-- Check index sizes
SELECT
    schemaname || '.' || tablename AS table_name,
    indexname,
    pg_size_pretty(pg_relation_size(schemaname || '.' || indexname)) AS index_size
FROM pg_indexes
WHERE schemaname = 'qbn'
AND tablename = 'ml_multi_timeframe_signals'
AND (indexname LIKE '%outcome%' OR indexname LIKE '%atr%')
ORDER BY pg_relation_size(schemaname || '.' || indexname) DESC;

-- Verify indexes are valid (not broken)
SELECT
    schemaname || '.' || tablename AS table_name,
    indexname,
    indexrelid::regclass AS index,
    indisvalid AS is_valid,
    indisready AS is_ready
FROM pg_index
JOIN pg_class ON pg_class.oid = indexrelid
JOIN pg_namespace ON pg_namespace.oid = relnamespace
JOIN pg_indexes ON pg_indexes.indexname = pg_class.relname
WHERE schemaname = 'qbn'
AND tablename = 'ml_multi_timeframe_signals'
AND (indexname LIKE '%outcome%' OR indexname LIKE '%atr%');

-- Test query to verify index usage
-- Expected: Should use Index Scan (not Seq Scan)
EXPLAIN (ANALYZE, BUFFERS)
SELECT asset_id, time, outcome_1h
FROM qbn.ml_multi_timeframe_signals
WHERE asset_id = 1
AND outcome_1h IS NOT NULL
ORDER BY time DESC
LIMIT 100;

-- ============================================================================
-- NOTES
-- ============================================================================

-- Index Strategy Rationale:
--
-- 1. Partial Indexes (WHERE outcome IS NOT NULL):
--    - Smaller index size (only rows with outcomes)
--    - Faster queries for CPT training (no NULL filtering needed)
--    - Essential for Phase 1.3 outcome backfill validation
--
-- 2. Composite Index (all_outcomes):
--    - Supports cross-horizon analysis queries
--    - Useful for model validation and backtesting
--    - Smaller than three separate indexes
--
-- 3. ATR Index:
--    - Supports outcome calculation validation
--    - Used in backfill quality checks
--    - Helpful for debugging outcome binning issues

-- Timing:
-- - Estimated execution time: 5-15 minutes per index (total: ~15-45 min)
-- - TimescaleDB handles hypertable partitioning automatically
-- - Can execute during any time since pipeline is not running

-- Performance Impact:
-- - Index scans vs Sequential scans: ~10-100x faster for filtered queries
-- - Disk space: ~10-20% of table size per index
-- - Write performance: Minimal impact (~2-5% overhead per insert)

-- Maintenance:
-- - Indexes are automatically maintained by TimescaleDB
-- - Run ANALYZE after backfill to update statistics
-- - Monitor pg_stat_user_indexes for unused indexes

-- Next Steps:
-- 1. Verify all indexes created successfully (is_valid = true)
-- 2. Test EXPLAIN plans show Index Scan (not Seq Scan)
-- 3. Monitor index usage after 24 hours (pg_stat_user_indexes)
-- 4. Run ANALYZE qbn.ml_multi_timeframe_signals after backfill