-- Rollback: rollback_010_signal_outcomes.sql
-- Purpose: Rollback migration 010 (normalize signal outcomes)
-- Date: 2025-12-16
-- Description: Drop qbn.signal_outcomes table and restore to pre-migration state
--
-- CONTEXT:
-- - No data migration was performed (all outcomes were NULL)
-- - Rollback is simple: just DROP TABLE CASCADE
-- - Python code must also be reverted from git

BEGIN;

-- =============================================================================
-- STEP 1: VERIFY NO CRITICAL DATA EXISTS
-- =============================================================================

DO $$
DECLARE
    row_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO row_count FROM qbn.signal_outcomes;

    IF row_count > 0 THEN
        RAISE WARNING 'qbn.signal_outcomes contains % rows. Continuing with rollback...', row_count;
    ELSE
        RAISE NOTICE 'qbn.signal_outcomes is empty. Safe to drop.';
    END IF;
END $$;

-- =============================================================================
-- STEP 2: DROP qbn.signal_outcomes TABLE
-- =============================================================================

-- Will automatically drop:
-- - All indexes on the table
-- - Hypertable metadata
-- - Check constraints
-- NOTE: No foreign key constraints exist (TimescaleDB limitation)

DROP TABLE IF EXISTS qbn.signal_outcomes CASCADE;

-- Verify table is dropped
SELECT
    schemaname,
    tablename
FROM pg_tables
WHERE schemaname = 'qbn' AND tablename = 'signal_outcomes';
-- Should return 0 rows

COMMIT;

-- =============================================================================
-- POST-ROLLBACK NOTES
-- =============================================================================

-- MANUAL STEPS REQUIRED:
-- 1. Revert Python code changes from git:
--    - outcome_backfill.py (lines 133-152, 284-339)
--    - cpt_generator.py (lines 408-434)
--    - target_generator.py (lines 184-240, 267-274)
--    - validate_outcome_backfill.py (multiple queries)
--
-- 2. Restart services:
--    - Outcome backfill cron job
--    - Training pipeline
--
-- 3. Verify no errors in logs

-- WARNING:
-- After rollback, outcome columns in kfl.mtf_signals_* tables remain empty.
-- The backfill script will still be broken (tries to use non-existent qbn.ml_multi_timeframe_signals).
-- If you want outcomes to work, you need to either:
--   a) Re-run migration 010 and fix Python code
--   b) Fix Python code to write outcomes directly to kfl.mtf_signals_* tables (not recommended - violates normalization)
