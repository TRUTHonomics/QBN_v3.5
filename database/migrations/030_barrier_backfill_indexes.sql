-- 030_barrier_backfill_indexes.sql
-- REASON: Indexes voor snellere barrier backfill queries
-- 
-- De get_pending_timestamps() query doet een JOIN over:
-- - kfl.mtf_signals_lead (time_1, asset_id)
-- - kfl.indicators (asset_id, time, interval_min)
-- - kfl.klines_raw (asset_id, time, interval_min)
-- - qbn.barrier_outcomes (asset_id, time_1)
--
-- Deze indexes versnellen de JOINs en filters aanzienlijk.

-- =============================================================================
-- 1. Index voor mtf_signals_lead (voor barrier backfill filtering)
-- =============================================================================
-- REASON: Filter op asset_id en time_1 waar minuut = 0 (hourly signals)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_signals_lead_barrier_backfill
ON kfl.mtf_signals_lead (asset_id, time_1)
WHERE EXTRACT(MINUTE FROM time_1) = 0;

-- =============================================================================
-- 2. Index voor klines_raw lookup (interval_min = '1')
-- =============================================================================
-- REASON: Snelle lookup voor 1-minuut klines per asset in tijdbereik
-- Dit is de meest gebruikte query in fetch_klines_batch()
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_klines_raw_1min_asset_time
ON kfl.klines_raw (asset_id, time)
WHERE interval_min = '1';

-- =============================================================================
-- 3. Index voor indicators daily lookup
-- =============================================================================
-- REASON: JOIN met indicators voor ATR waarden (daily interval)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_indicators_daily_asset_time
ON kfl.indicators (asset_id, time)
WHERE interval_min = 'D';

-- =============================================================================
-- 4. Index voor barrier_outcomes lookup (LEFT JOIN voor pending check)
-- =============================================================================
-- REASON: Snelle check of outcome al bestaat
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_barrier_outcomes_asset_time
ON qbn.barrier_outcomes (asset_id, time_1);

-- =============================================================================
-- 5. Partial index voor barrier_outcomes zonder data (pending)
-- =============================================================================
-- REASON: Snelle identificatie van rijen die nog geen outcome hebben
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_barrier_outcomes_pending
ON qbn.barrier_outcomes (asset_id, time_1)
WHERE first_significant_barrier IS NULL;

-- =============================================================================
-- Verify indexes
-- =============================================================================
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_indexes
WHERE indexname LIKE 'idx_%barrier%' 
   OR indexname LIKE 'idx_klines_raw_1min%'
   OR indexname LIKE 'idx_indicators_daily%'
   OR indexname LIKE 'idx_mtf_signals_lead_barrier%'
ORDER BY tablename, indexname;
