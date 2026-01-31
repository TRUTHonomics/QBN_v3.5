-- Migration: 010_normalize_signal_outcomes.sql
-- Purpose: Normalize outcome columns to separate qbn.signal_outcomes table
-- Date: 2025-12-16
-- Priority: CRITICAL
-- Description: Create qbn.signal_outcomes table and migrate outcome data from kfl.mtf_signals_* tables
--
-- CONTEXT:
-- - qbn.ml_multi_timeframe_signals does NOT exist (verouderd)
-- - Actual tables are kfl.mtf_signals_lead/coin/conf (87M rows, 0 outcomes)
-- - Backfill script is broken (tries to read from non-existent QBN table)
-- - This migration normalizes outcomes to reduce duplication
--
-- ARCHITECTURE:
-- - Outcomes-only table: qbn.signal_outcomes
-- - Foreign key to kfl.mtf_signals_lead (primary source table)
-- - TimescaleDB hypertable for time-series optimization
-- - Partial indexes for training queries (WHERE outcome IS NOT NULL)

BEGIN;

-- =============================================================================
-- STEP 1: CREATE qbn.signal_outcomes TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS qbn.signal_outcomes (
    -- Primary key (matches kfl.mtf_signals_lead)
    asset_id INTEGER NOT NULL,
    time_1 TIMESTAMPTZ NOT NULL,

    -- Outcome bins (-3 to +3, ATR-relative)
    outcome_1h SMALLINT CHECK (outcome_1h BETWEEN -3 AND 3),
    outcome_4h SMALLINT CHECK (outcome_4h BETWEEN -3 AND 3),
    outcome_1d SMALLINT CHECK (outcome_1d BETWEEN -3 AND 3),

    -- Raw return percentages (for analysis and re-binning)
    return_1h_pct REAL,
    return_4h_pct REAL,
    return_1d_pct REAL,

    -- ATR value at signal time (for normalization)
    atr_at_signal REAL CHECK (atr_at_signal IS NULL OR atr_at_signal > 0),

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Primary key constraint
    PRIMARY KEY (asset_id, time_1)
);

-- Comments for documentation
COMMENT ON TABLE qbn.signal_outcomes IS
'Normalized outcome data for multi-horizon price predictions. Separated from signal data to eliminate duplication.';

COMMENT ON COLUMN qbn.signal_outcomes.outcome_1h IS
'1h price outcome, discretized to 7 ATR-bins (-3 to +3). NULL = not yet calculated.';

COMMENT ON COLUMN qbn.signal_outcomes.outcome_4h IS
'4h price outcome, discretized to 7 ATR-bins (-3 to +3). NULL = not yet calculated.';

COMMENT ON COLUMN qbn.signal_outcomes.outcome_1d IS
'1d price outcome, discretized to 7 ATR-bins (-3 to +3). NULL = not yet calculated.';

COMMENT ON COLUMN qbn.signal_outcomes.return_1h_pct IS
'Raw 1h return percentage: (close_future - close_now) / close_now * 100';

COMMENT ON COLUMN qbn.signal_outcomes.return_4h_pct IS
'Raw 4h return percentage for analysis';

COMMENT ON COLUMN qbn.signal_outcomes.return_1d_pct IS
'Raw 1d return percentage for analysis';

COMMENT ON COLUMN qbn.signal_outcomes.atr_at_signal IS
'ATR value (%) at signal time, used for outcome discretization';

-- =============================================================================
-- STEP 2: CONVERT TO TIMESCALEDB HYPERTABLE
-- =============================================================================

SELECT create_hypertable(
    'qbn.signal_outcomes',
    'time_1',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

COMMENT ON COLUMN qbn.signal_outcomes.time_1 IS
'Partition column (1-minute timestamp). Matches kfl.mtf_signals_lead.time_1.';

-- =============================================================================
-- STEP 3: CREATE PARTIAL INDEXES FOR TRAINING QUERIES
-- =============================================================================

-- Index for 1h outcome training data (only where outcome exists)
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_1h_filled
    ON qbn.signal_outcomes (asset_id, time_1)
    WHERE outcome_1h IS NOT NULL;

-- Index for 4h outcome training data
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_4h_filled
    ON qbn.signal_outcomes (asset_id, time_1)
    WHERE outcome_4h IS NOT NULL;

-- Index for 1d outcome training data
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_1d_filled
    ON qbn.signal_outcomes (asset_id, time_1)
    WHERE outcome_1d IS NOT NULL;

-- Index for backfill queries (find records WITHOUT outcomes)
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_1h_null
    ON qbn.signal_outcomes (asset_id, time_1)
    WHERE outcome_1h IS NULL;

COMMENT ON INDEX qbn.idx_signal_outcomes_1h_filled IS
'Partial index for CPT training queries (only rows with 1h outcomes)';

COMMENT ON INDEX qbn.idx_signal_outcomes_4h_filled IS
'Partial index for CPT training queries (only rows with 4h outcomes)';

COMMENT ON INDEX qbn.idx_signal_outcomes_1d_filled IS
'Partial index for CPT training queries (only rows with 1d outcomes)';

COMMENT ON INDEX qbn.idx_signal_outcomes_1h_null IS
'Partial index for backfill queries (find rows needing outcome calculation)';

-- =============================================================================
-- STEP 4: REFERENTIAL INTEGRITY (NO FOREIGN KEY - TIMESCALEDB LIMITATION)
-- =============================================================================

-- NOTE: Cannot add foreign key constraint between two hypertables
-- TimescaleDB limitation: "hypertables cannot be used as foreign key references of hypertables"
--
-- REFERENTIAL INTEGRITY ENFORCEMENT:
-- - Application-level: backfill script only writes outcomes for existing signals
-- - No DELETE operations on kfl.mtf_signals_lead in normal operations
-- - Orphaned outcomes are prevented by application logic, not database constraints
--
-- ALTERNATIVE: Create validation trigger if strict enforcement is required

COMMENT ON TABLE qbn.signal_outcomes IS
'Normalized outcome data for multi-horizon price predictions. Separated from signal data to eliminate duplication.
REFERENTIAL INTEGRITY: Logically references kfl.mtf_signals_lead(asset_id, time_1) but no FK constraint due to TimescaleDB limitation.
Application ensures only valid signal references exist.';

-- =============================================================================
-- STEP 5: DATA MIGRATION (SKIP - ALL OUTCOMES ARE NULL)
-- =============================================================================

-- Verification query (should return 0 outcomes):
-- SELECT COUNT(*), COUNT(outcome_1h), COUNT(outcome_4h), COUNT(outcome_1d)
-- FROM kfl.mtf_signals_lead;
-- Result: 87,140,762 rows, 0 outcomes

-- No data migration needed - all outcome columns in kfl.mtf_signals_* are NULL

-- =============================================================================
-- STEP 6: VERIFICATION QUERIES
-- =============================================================================

-- Verify table exists
SELECT
    schemaname,
    tablename,
    tableowner
FROM pg_tables
WHERE schemaname = 'qbn' AND tablename = 'signal_outcomes';

-- Verify hypertable conversion
SELECT
    hypertable_schema,
    hypertable_name,
    num_dimensions,
    num_chunks
FROM timescaledb_information.hypertables
WHERE hypertable_schema = 'qbn' AND hypertable_name = 'signal_outcomes';

-- Verify indexes
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'qbn' AND tablename = 'signal_outcomes'
ORDER BY indexname;

-- Verify check constraints
SELECT
    conname,
    contype,
    pg_get_constraintdef(oid) as definition
FROM pg_constraint
WHERE conrelid = 'qbn.signal_outcomes'::regclass
    AND contype IN ('c', 'p')  -- c=check, p=primary key
ORDER BY conname;

-- Verify row count (should be 0 initially)
SELECT COUNT(*) as total_rows FROM qbn.signal_outcomes;

COMMIT;

-- =============================================================================
-- POST-MIGRATION NOTES
-- =============================================================================

-- NEXT STEPS:
-- 1. Update outcome_backfill.py to read from kfl.mtf_signals_lead and write to qbn.signal_outcomes
-- 2. Update cpt_generator.py to JOIN kfl.mtf_signals_lead with qbn.signal_outcomes
-- 3. Update validate_outcome_backfill.py to use new schema
-- 4. Test backfill script with small batch
-- 5. Monitor performance (JOIN overhead should be <100ms)

-- ROLLBACK:
-- DROP TABLE qbn.signal_outcomes CASCADE;

-- CLEANUP (after 30+ days):
-- ALTER TABLE kfl.mtf_signals_lead DROP COLUMN outcome_1h, outcome_4h, outcome_1d, return_1h_pct, return_4h_pct, return_1d_pct, atr_at_signal;
-- ALTER TABLE kfl.mtf_signals_coin DROP COLUMN outcome_1h, outcome_4h, outcome_1d, return_1h_pct, return_4h_pct, return_1d_pct, atr_at_signal;
-- ALTER TABLE kfl.mtf_signals_conf DROP COLUMN outcome_1h, outcome_4h, outcome_1d, return_1h_pct, return_4h_pct, return_1d_pct, atr_at_signal;
