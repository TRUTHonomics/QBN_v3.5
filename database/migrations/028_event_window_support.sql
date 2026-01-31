-- Migration 028: Event Window Support for QBN v3.1
-- Date: 2026-01-11
-- Description: Adds event_id column to barrier_outcomes and creates event_windows cache table

-- ============================================================================
-- 1. Add event_id column to barrier_outcomes
-- ============================================================================

ALTER TABLE qbn.barrier_outcomes
ADD COLUMN IF NOT EXISTS event_id VARCHAR(32);

-- Index for event-based queries
CREATE INDEX IF NOT EXISTS idx_barrier_outcomes_event
ON qbn.barrier_outcomes(asset_id, event_id);

COMMENT ON COLUMN qbn.barrier_outcomes.event_id IS
    'v3.1: Event window identifier for contextual training data';

-- ============================================================================
-- 2. Create event_windows cache table
-- ============================================================================

CREATE TABLE IF NOT EXISTS qbn.event_windows (
    event_id VARCHAR(32) PRIMARY KEY,
    asset_id INTEGER NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    trigger_score REAL NOT NULL,
    trigger_delta REAL DEFAULT 0,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('long', 'short')),
    outcome VARCHAR(20) NOT NULL CHECK (outcome IN (
        'up_strong', 'up_weak', 'neutral', 'down_weak', 'down_strong', 'timeout'
    )),
    duration_minutes INTEGER NOT NULL,
    n_rows INTEGER NOT NULL,
    run_id VARCHAR(32),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE qbn.event_windows OWNER TO qbn;

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_event_windows_asset
ON qbn.event_windows(asset_id, start_time DESC);

CREATE INDEX IF NOT EXISTS idx_event_windows_outcome
ON qbn.event_windows(asset_id, outcome);

CREATE INDEX IF NOT EXISTS idx_event_windows_direction
ON qbn.event_windows(asset_id, direction);

COMMENT ON TABLE qbn.event_windows IS
    'v3.1: Cache for detected event windows (Leading spike to barrier hit)';

-- ============================================================================
-- 3. Create output_position table for Position_Prediction
-- ============================================================================

CREATE TABLE IF NOT EXISTS qbn.output_position (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,

    -- Position context
    position_confidence VARCHAR(10) NOT NULL CHECK (position_confidence IN ('low', 'medium', 'high')),
    time_since_entry_min INTEGER NOT NULL,
    current_pnl_atr REAL NOT NULL,

    -- Prediction output
    p_target_hit REAL NOT NULL CHECK (p_target_hit >= 0 AND p_target_hit <= 1),
    p_stoploss_hit REAL NOT NULL CHECK (p_stoploss_hit >= 0 AND p_stoploss_hit <= 1),
    p_timeout REAL NOT NULL CHECK (p_timeout >= 0 AND p_timeout <= 1),
    dominant_outcome VARCHAR(20) NOT NULL,
    prediction_confidence REAL NOT NULL CHECK (prediction_confidence >= 0 AND prediction_confidence <= 1),

    -- Context (for debugging)
    coincident_composite VARCHAR(20),
    confirming_composite VARCHAR(20),

    -- Metadata
    inference_time_ms REAL,
    model_version VARCHAR(10) DEFAULT '3.1',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE qbn.output_position OWNER TO qbn;

-- Primary lookup index
CREATE INDEX IF NOT EXISTS idx_output_position_asset_time
ON qbn.output_position(asset_id, timestamp DESC);

-- Model version index for migrations
CREATE INDEX IF NOT EXISTS idx_output_position_version
ON qbn.output_position(model_version);

COMMENT ON TABLE qbn.output_position IS
    'v3.1: Position_Prediction output for active position management';

-- ============================================================================
-- 4. pg_notify trigger for real-time position updates
-- ============================================================================

CREATE OR REPLACE FUNCTION notify_position_update()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('qbn_position_update', row_to_json(NEW)::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_position_update ON qbn.output_position;

CREATE TRIGGER trg_position_update
AFTER INSERT ON qbn.output_position
FOR EACH ROW EXECUTE FUNCTION notify_position_update();

-- ============================================================================
-- 5. Grant permissions
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON qbn.event_windows TO qbn;
GRANT SELECT, INSERT, UPDATE, DELETE ON qbn.output_position TO qbn;
GRANT USAGE, SELECT ON SEQUENCE qbn.output_position_id_seq TO qbn;

-- ============================================================================
-- Verification
-- ============================================================================

DO $$
BEGIN
    -- Verify event_id column exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'qbn' AND table_name = 'barrier_outcomes' AND column_name = 'event_id'
    ) THEN
        RAISE EXCEPTION 'event_id column not created in barrier_outcomes';
    END IF;

    -- Verify event_windows table exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'qbn' AND table_name = 'event_windows'
    ) THEN
        RAISE EXCEPTION 'event_windows table not created';
    END IF;

    -- Verify output_position table exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'qbn' AND table_name = 'output_position'
    ) THEN
        RAISE EXCEPTION 'output_position table not created';
    END IF;

    RAISE NOTICE 'Migration 028 completed successfully';
END $$;
