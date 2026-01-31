-- Migration: Position Delta Threshold Configuration Table
-- Version: 3.2
-- Date: 2026-01-22
-- Description: Tabel voor opslag van geoptimaliseerde delta thresholds voor Position_Confidence/Prediction

-- =============================================================================
-- TABEL: qbn.position_delta_threshold_config
-- =============================================================================
-- Slaat optimale thresholds op voor delta scores in position management.
-- Delta scores meten verandering in coincident/confirming composites sinds entry.
--
-- delta_type: 'cumulative' (sinds entry) of 'instantaneous' (per candle)
-- score_type: 'coincident' of 'confirming'
-- threshold: waarde voor discretisatie naar deteriorating/stable/improving

CREATE TABLE IF NOT EXISTS qbn.position_delta_threshold_config (
    id SERIAL PRIMARY KEY,
    
    -- Scope
    asset_id INTEGER NOT NULL,
    
    -- Delta configuratie
    delta_type VARCHAR(20) NOT NULL CHECK (delta_type IN ('cumulative', 'instantaneous')),
    score_type VARCHAR(20) NOT NULL CHECK (score_type IN ('coincident', 'confirming')),
    
    -- Threshold waarde
    threshold DECIMAL(6,4) NOT NULL,
    
    -- Optimalisatie metadata
    mi_score DECIMAL(8,6),           -- Mutual Information score
    distribution JSONB,               -- State distributie na discretisatie
    source_method VARCHAR(50),        -- 'MI Grid Search', 'Manual', etc.
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(asset_id, delta_type, score_type)
);

-- Index voor snelle lookups
CREATE INDEX IF NOT EXISTS idx_position_delta_threshold_asset 
    ON qbn.position_delta_threshold_config(asset_id);

-- Trigger voor updated_at
CREATE OR REPLACE FUNCTION qbn.update_position_delta_threshold_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_position_delta_threshold_timestamp 
    ON qbn.position_delta_threshold_config;

CREATE TRIGGER trigger_update_position_delta_threshold_timestamp
    BEFORE UPDATE ON qbn.position_delta_threshold_config
    FOR EACH ROW
    EXECUTE FUNCTION qbn.update_position_delta_threshold_timestamp();

-- =============================================================================
-- COMMENT
-- =============================================================================
COMMENT ON TABLE qbn.position_delta_threshold_config IS 
    'v3.2: Geoptimaliseerde delta thresholds voor Position_Confidence en Position_Prediction training. '
    'Delta scores meten verandering in coincident/confirming composites sinds trade entry.';

COMMENT ON COLUMN qbn.position_delta_threshold_config.delta_type IS 
    'Type delta: cumulative (sinds entry) of instantaneous (per candle)';

COMMENT ON COLUMN qbn.position_delta_threshold_config.score_type IS 
    'Welke composite: coincident of confirming';

COMMENT ON COLUMN qbn.position_delta_threshold_config.threshold IS 
    'Threshold voor discretisatie: delta < -threshold = deteriorating, delta > +threshold = improving, anders stable';

COMMENT ON COLUMN qbn.position_delta_threshold_config.mi_score IS 
    'Mutual Information score van de optimalisatie';

-- =============================================================================
-- DEFAULT DATA (optioneel)
-- =============================================================================
-- Voeg defaults toe voor asset 9889 (test asset) als deze nog niet bestaan
INSERT INTO qbn.position_delta_threshold_config 
    (asset_id, delta_type, score_type, threshold, source_method, distribution)
VALUES 
    (9889, 'cumulative', 'coincident', 0.08, 'default', '{"deteriorating": 0.33, "stable": 0.34, "improving": 0.33}'),
    (9889, 'cumulative', 'confirming', 0.10, 'default', '{"deteriorating": 0.33, "stable": 0.34, "improving": 0.33}')
ON CONFLICT (asset_id, delta_type, score_type) DO NOTHING;
