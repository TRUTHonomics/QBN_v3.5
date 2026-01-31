-- =============================================================================
-- 003_add_outcome_indexes.sql
-- =============================================================================
-- Voeg performance indexen toe voor outcome backfill queries.
-- Deze partial indexes versnellen queries die zoeken naar NULL outcomes
-- met gevulde atr_at_signal waarden.
--
-- USAGE:
--   psql -h 10.10.10.1 -U postgres -d KFLhyper -f 003_add_outcome_indexes.sql
--
-- NOTE: CREATE INDEX CONCURRENTLY kan niet binnen een transactie.
--       Dit script moet statement-by-statement worden uitgevoerd.
--
-- REASON: outcome_backfill.py queries zoeken naar records met:
--         - outcome_{horizon} IS NULL
--         - atr_at_signal IS NOT NULL AND atr_at_signal > 0
--         - time < NOW() - INTERVAL 'X minutes'
-- =============================================================================

-- =============================================================================
-- INDEXEN: kfl.mtf_signals_lead
-- =============================================================================
-- Partial index voor 1h outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_lead_outcome_1h_backfill
    ON kfl.mtf_signals_lead (asset_id, time)
    WHERE outcome_1h IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- Partial index voor 4h outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_lead_outcome_4h_backfill
    ON kfl.mtf_signals_lead (asset_id, time)
    WHERE outcome_4h IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- Partial index voor 1d outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_lead_outcome_1d_backfill
    ON kfl.mtf_signals_lead (asset_id, time)
    WHERE outcome_1d IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- =============================================================================
-- INDEXEN: kfl.mtf_signals_coin
-- =============================================================================
-- Partial index voor 1h outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_coin_outcome_1h_backfill
    ON kfl.mtf_signals_coin (asset_id, time)
    WHERE outcome_1h IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- Partial index voor 4h outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_coin_outcome_4h_backfill
    ON kfl.mtf_signals_coin (asset_id, time)
    WHERE outcome_4h IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- Partial index voor 1d outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_coin_outcome_1d_backfill
    ON kfl.mtf_signals_coin (asset_id, time)
    WHERE outcome_1d IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- =============================================================================
-- INDEXEN: kfl.mtf_signals_conf
-- =============================================================================
-- Partial index voor 1h outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_conf_outcome_1h_backfill
    ON kfl.mtf_signals_conf (asset_id, time)
    WHERE outcome_1h IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- Partial index voor 4h outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_conf_outcome_4h_backfill
    ON kfl.mtf_signals_conf (asset_id, time)
    WHERE outcome_4h IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- Partial index voor 1d outcome backfill
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtf_conf_outcome_1d_backfill
    ON kfl.mtf_signals_conf (asset_id, time)
    WHERE outcome_1d IS NULL AND atr_at_signal IS NOT NULL AND atr_at_signal > 0;

-- =============================================================================
-- VERIFICATIE
-- =============================================================================
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'kfl'
  AND indexname LIKE 'idx_mtf_%_outcome_%_backfill'
ORDER BY tablename, indexname;

-- Toon index sizes
SELECT
    i.relname AS index_name,
    pg_size_pretty(pg_relation_size(i.oid)) AS index_size
FROM pg_class t
JOIN pg_index ix ON t.oid = ix.indrelid
JOIN pg_class i ON i.oid = ix.indexrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE n.nspname = 'kfl'
  AND i.relname LIKE 'idx_mtf_%_outcome_%_backfill'
ORDER BY pg_relation_size(i.oid) DESC;
