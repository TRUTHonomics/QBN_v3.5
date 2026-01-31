-- Migration: Create qbn.output_entry table with pg_notify trigger for TSEM
-- Date: 2026-01-09
-- Purpose: QBN entry output voor TSEM integratie via pg_notify
-- NOTE: QBN levert alleen inference output. TSEM bepaalt sizing/leverage/SL.

-- QBN Entry Output tabel
CREATE TABLE IF NOT EXISTS qbn.output_entry (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Asset identificatie
    asset_id INTEGER NOT NULL,
    symbol VARCHAR(32) NOT NULL,

    -- Trade Decision (uit Trade_Hypothesis)
    trade_hypothesis VARCHAR(20) NOT NULL,  -- no_setup/weak_long/strong_long/weak_short/strong_short

    -- Entry Confidence (TSEM bepaalt sizing op basis hiervan)
    entry_confidence VARCHAR(10) NOT NULL,     -- low/medium/high
    entry_confidence_score REAL,               -- -1.0 to +1.0 (alignment score)

    -- Context
    regime VARCHAR(25),
    leading_composite VARCHAR(20),
    coincident_composite VARCHAR(20),
    confirming_composite VARCHAR(20),

    -- Predictions (optional, for logging)
    prediction_1h VARCHAR(20),
    prediction_4h VARCHAR(20),
    prediction_1d VARCHAR(20),

    -- Metadata
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMPTZ,

    PRIMARY KEY (id, time)
);

-- TimescaleDB hypertable (7 dagen chunks)
SELECT create_hypertable('qbn.output_entry', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Index voor TSEM queries
CREATE INDEX IF NOT EXISTS idx_output_entry_asset_time
ON qbn.output_entry (asset_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_output_entry_unprocessed
ON qbn.output_entry (processed, time DESC)
WHERE processed = FALSE;

-- Retention policy (90 dagen)
SELECT add_retention_policy('qbn.output_entry', INTERVAL '90 days', if_not_exists => TRUE);

-- Trigger functie voor pg_notify
CREATE OR REPLACE FUNCTION qbn.notify_signal()
RETURNS TRIGGER AS $$
DECLARE
    payload JSON;
BEGIN
    -- Bouw compacte JSON payload (< 8000 bytes)
    -- NOTE: Alleen inference output, geen sizing parameters
    payload := json_build_object(
        'id', NEW.id,
        'time', NEW.time,
        'asset_id', NEW.asset_id,
        'symbol', NEW.symbol,
        'hypothesis', NEW.trade_hypothesis,
        'entry_conf', NEW.entry_confidence,
        'entry_score', NEW.entry_confidence_score,
        'regime', NEW.regime,
        'leading', NEW.leading_composite,
        'coincident', NEW.coincident_composite,
        'confirming', NEW.confirming_composite,
        'pred_1h', NEW.prediction_1h,
        'pred_4h', NEW.prediction_4h,
        'pred_1d', NEW.prediction_1d
    );

    -- Notify TSEM
    PERFORM pg_notify('qbn_signal', payload::text);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger op INSERT
DROP TRIGGER IF EXISTS trg_output_entry_notify ON qbn.output_entry;
CREATE TRIGGER trg_output_entry_notify
    AFTER INSERT ON qbn.output_entry
    FOR EACH ROW
    EXECUTE FUNCTION qbn.notify_signal();

-- Comments
COMMENT ON TABLE qbn.output_entry IS 'QBN entry signals voor TSEM. Alleen entry-gerelateerde data (geen position management).';
COMMENT ON COLUMN qbn.output_entry.entry_confidence IS 'Entry confidence state (low/medium/high) - TSEM vertaalt naar sizing/leverage';
COMMENT ON COLUMN qbn.output_entry.entry_confidence_score IS 'Alignment score (-1.0 to +1.0) uit Coincident+Confirming alignment';
