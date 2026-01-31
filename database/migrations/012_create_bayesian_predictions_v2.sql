-- ============================================================================
-- MIGRATION: 012_create_bayesian_predictions_v2
-- DESCRIPTION: Creates the v2 schema for Bayesian Predictions with 7-state
--              distributions, multi-horizon support and TimescaleDB policies.
-- DATE: 2026-01-02
-- ============================================================================

-- 1. DROP OLD TABLE IF EXISTS (Cleaning up potential legacy v1)
DROP TABLE IF EXISTS qbn.bayesian_predictions CASCADE;

-- 2. CREATE HISTORICAL HYPERTABLE
CREATE TABLE qbn.bayesian_predictions (
    asset_id INTEGER NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    
    -- Multi-horizon predictions (-3=Strong_Bearish tot +3=Strong_Bullish)
    prediction_1h SMALLINT,
    prediction_4h SMALLINT,
    prediction_1d SMALLINT,
    
    -- Confidence per horizon (max probability)
    confidence_1h REAL,
    confidence_4h REAL,
    confidence_1d REAL,
    
    -- Full 7-state distributions (JSONB)
    distribution_1h JSONB,
    distribution_4h JSONB,
    distribution_1d JSONB,
    
    -- Expected ATR moves (weighted average)
    expected_atr_1h REAL,
    expected_atr_4h REAL,
    expected_atr_1d REAL,
    
    -- Semantic breakdown (Intermediate nodes)
    regime VARCHAR(25),
    leading_composite VARCHAR(20),
    coincident_composite VARCHAR(20),
    confirming_composite VARCHAR(20),
    entry_timing_distribution JSONB,
    
    -- Metadata
    inference_time_ms REAL,
    model_version VARCHAR(10) DEFAULT '2.0',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (asset_id, time)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('qbn.bayesian_predictions', 'time', 
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- 3. CREATE CURRENT CACHE TABLE
CREATE TABLE qbn.bayesian_predictions_current (
    asset_id INTEGER PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    
    prediction_1h SMALLINT,
    prediction_4h SMALLINT,
    prediction_1d SMALLINT,
    
    confidence_1h REAL,
    confidence_4h REAL,
    confidence_1d REAL,
    
    distribution_1h JSONB,
    distribution_4h JSONB,
    distribution_1d JSONB,
    
    expected_atr_1h REAL,
    expected_atr_4h REAL,
    expected_atr_1d REAL,
    
    regime VARCHAR(25),
    leading_composite VARCHAR(20),
    coincident_composite VARCHAR(20),
    confirming_composite VARCHAR(20),
    entry_timing_distribution JSONB,
    
    inference_time_ms REAL,
    model_version VARCHAR(10) DEFAULT '2.0',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. STORAGE POLICIES

-- Compression policy: Compress chunks after 30 days
ALTER TABLE qbn.bayesian_predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset_id',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('qbn.bayesian_predictions', INTERVAL '30 days');

-- Retention policy: Keep data for 1 year
SELECT add_retention_policy('qbn.bayesian_predictions', INTERVAL '365 days');

-- 5. INDEXES
CREATE INDEX idx_bayesian_predictions_asset_time ON qbn.bayesian_predictions (asset_id, time DESC);
CREATE INDEX idx_bayesian_predictions_regime ON qbn.bayesian_predictions (regime, time DESC);

-- 6. COMMENTS
COMMENT ON TABLE qbn.bayesian_predictions IS 'Historical Bayesian Network predictions (v2)';
COMMENT ON TABLE qbn.bayesian_predictions_current IS 'Most recent Bayesian Network predictions per asset (v2)';

