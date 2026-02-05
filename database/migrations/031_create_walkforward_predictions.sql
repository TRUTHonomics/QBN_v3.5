-- =============================================================================
-- MIGRATIE: Walk-Forward Predictions (retrospectief, lookahead-safe)
-- Versie: 1.0
-- Datum: 2026-02-01
--
-- REASON: Deze tabel bevat retrospectief berekende predictions uit walk-forward
--         simulaties (niet live productie inference). Dit maakt offline
--         accuracy evaluatie mogelijk tegen qbn.barrier_outcomes zonder
--         lookahead bias.
-- =============================================================================

BEGIN;

-- Controleer of schema bestaat
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'qbn') THEN
        RAISE EXCEPTION 'Schema qbn bestaat niet. Creëer eerst het schema.';
    END IF;
END $$;

-- =============================================================================
-- TABEL: qbn.walkforward_predictions
-- =============================================================================

CREATE TABLE IF NOT EXISTS qbn.walkforward_predictions (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,  -- Prediction timestamp (candle close)
    asset_id INTEGER NOT NULL,
    run_id VARCHAR(50) NOT NULL,  -- Training run gebruikt voor deze prediction

    -- Entry-side predictions (categorical)
    trade_hypothesis VARCHAR(20),
    entry_confidence VARCHAR(10),
    leading_composite VARCHAR(20),
    coincident_composite VARCHAR(20),
    confirming_composite VARCHAR(20),

    -- Directional predictions per horizon (legacy compat)
    prediction_1h VARCHAR(20),
    prediction_4h VARCHAR(20),
    prediction_1d VARCHAR(20),

    -- Full distributions (JSONB)
    distribution_1h JSONB,
    distribution_4h JSONB,
    distribution_1d JSONB,

    -- Barrier predictions (JSONB: o.a. expected_direction)
    barrier_prediction_1h JSONB,
    barrier_prediction_4h JSONB,
    barrier_prediction_1d JSONB,

    -- Walk-forward metadata (train window used for this prediction)
    train_window_start TIMESTAMPTZ,
    train_window_end TIMESTAMPTZ,
    backtest_run_id VARCHAR(100),  -- Identifier voor deze walk-forward sessie

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, time)
);

ALTER TABLE qbn.walkforward_predictions OWNER TO postgres;

COMMENT ON TABLE qbn.walkforward_predictions IS 'Retrospectieve walk-forward predictions (offline) voor accuracy evaluatie vs barrier_outcomes; lookahead-safe.';
COMMENT ON COLUMN qbn.walkforward_predictions.backtest_run_id IS 'Groepering van 1 walk-forward run/sessie (reporting/traceability).';

-- =============================================================================
-- TIMESCALE: hypertable
-- =============================================================================

-- REASON: Time-partitioned queries per asset/horizon.
SELECT create_hypertable('qbn.walkforward_predictions', 'time', if_not_exists => TRUE);

-- =============================================================================
-- INDICES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_wf_pred_asset_time ON qbn.walkforward_predictions(asset_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_wf_pred_backtest_run ON qbn.walkforward_predictions(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_wf_pred_run_id ON qbn.walkforward_predictions(run_id);

-- =============================================================================
-- VALIDATIE
-- =============================================================================

DO $$
DECLARE
    tbl_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'qbn' AND table_name = 'walkforward_predictions'
    ) INTO tbl_exists;
    IF NOT tbl_exists THEN
        RAISE EXCEPTION 'walkforward_predictions tabel niet aangemaakt';
    END IF;
    RAISE NOTICE 'Migratie succesvol: qbn.walkforward_predictions aangemaakt';
END $$;

COMMIT;

