-- Migration: Deprecate bayesian_predictions tables
-- Date: 2026-01-09
-- Purpose: Mark bayesian_predictions tables as deprecated
--
-- All inference output is now consolidated to qbn.output_entry.
-- These tables are kept temporarily for:
-- - Backward compatibility during transition
-- - Historical data access
-- - Rollback capability
--
-- DO NOT DROP these tables yet - they will be removed in a future migration
-- after confirming output_entry contains all necessary data.

-- Mark tables as deprecated via comments
COMMENT ON TABLE qbn.bayesian_predictions IS
'DEPRECATED (2026-01-09): This table is deprecated. Use qbn.output_entry instead.
All new inference output goes to output_entry. This table is kept for historical data only.
Will be dropped in a future migration after transition period.';

COMMENT ON TABLE qbn.bayesian_predictions_current IS
'DEPRECATED (2026-01-09): This table is deprecated. Use qbn.output_entry instead.
Real-time cache functionality is no longer needed - output_entry provides all data.
Will be dropped in a future migration after transition period.';

-- Reduce retention policy to speed up cleanup (from 365 to 30 days)
-- This will gradually remove old data
SELECT remove_retention_policy('qbn.bayesian_predictions', if_exists => TRUE);
SELECT add_retention_policy('qbn.bayesian_predictions', INTERVAL '30 days', if_not_exists => TRUE);

-- Log deprecation event
DO $$
BEGIN
    RAISE NOTICE 'bayesian_predictions tables marked as DEPRECATED';
    RAISE NOTICE 'All new inference output should use qbn.output_entry';
    RAISE NOTICE 'Retention policy reduced to 30 days for gradual cleanup';
END $$;
