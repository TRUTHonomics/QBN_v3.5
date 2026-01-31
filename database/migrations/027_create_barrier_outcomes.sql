-- f:/Containers/QBN_v3/database/migrations/027_create_barrier_outcomes.sql
-- =============================================================================
-- MIGRATIE: First-Touch Barrier Tables
-- Versie: 1.0
-- Datum: 2026-01-10
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
-- BARRIER OUTCOMES TABEL
-- =============================================================================

CREATE TABLE IF NOT EXISTS qbn.barrier_outcomes (
    asset_id INTEGER NOT NULL,
    time_1 TIMESTAMPTZ NOT NULL,
    atr_at_signal REAL NOT NULL,
    reference_price REAL NOT NULL,
    max_observation_min SMALLINT NOT NULL DEFAULT 2880,
    time_to_up_025_atr SMALLINT,
    time_to_up_050_atr SMALLINT,
    time_to_up_075_atr SMALLINT,
    time_to_up_100_atr SMALLINT,
    time_to_up_125_atr SMALLINT,
    time_to_up_150_atr SMALLINT,
    time_to_down_025_atr SMALLINT,
    time_to_down_050_atr SMALLINT,
    time_to_down_075_atr SMALLINT,
    time_to_down_100_atr SMALLINT,
    time_to_down_125_atr SMALLINT,
    time_to_down_150_atr SMALLINT,
    max_up_atr REAL,
    max_down_atr REAL,
    time_to_max_up_min SMALLINT,
    time_to_max_down_min SMALLINT,
    first_significant_barrier VARCHAR(10),
    first_significant_time_min SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (asset_id, time_1),
    CONSTRAINT barrier_outcomes_atr_positive CHECK (atr_at_signal > 0),
    CONSTRAINT barrier_outcomes_price_positive CHECK (reference_price > 0),
    CONSTRAINT barrier_outcomes_window_range CHECK (max_observation_min BETWEEN 1 AND 10080),
    CONSTRAINT barrier_outcomes_significant_valid CHECK (first_significant_barrier IN (
        'up_025', 'up_050', 'up_075', 'up_100', 'up_125', 'up_150',
        'down_025', 'down_050', 'down_075', 'down_100', 'down_125', 'down_150',
        'none'
    ) OR first_significant_barrier IS NULL)
);

ALTER TABLE qbn.barrier_outcomes OWNER TO postgres;

COMMENT ON TABLE qbn.barrier_outcomes IS 'First-touch barrier outcomes voor trading-gerichte voorspellingen';
COMMENT ON COLUMN qbn.barrier_outcomes.time_to_up_075_atr IS 'Minuten tot prijs +0.75 ATR bereikt (NULL = niet bereikt binnen window)';
COMMENT ON COLUMN qbn.barrier_outcomes.max_up_atr IS 'Maximale upside in ATR-eenheden bereikt binnen observation window';
COMMENT ON COLUMN qbn.barrier_outcomes.first_significant_barrier IS 'Welke "significant" barrier (default 0.75 ATR) het eerst werd geraakt';

-- =============================================================================
-- INDICES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_barrier_outcomes_time ON qbn.barrier_outcomes(time_1 DESC);
CREATE INDEX IF NOT EXISTS idx_barrier_outcomes_significant ON qbn.barrier_outcomes(asset_id, first_significant_barrier, first_significant_time_min) WHERE first_significant_barrier IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_barrier_outcomes_asset_time ON qbn.barrier_outcomes(asset_id, time_1);
CREATE INDEX IF NOT EXISTS idx_barrier_outcomes_up_first ON qbn.barrier_outcomes(asset_id, time_1, first_significant_time_min) WHERE first_significant_barrier LIKE 'up_%';
CREATE INDEX IF NOT EXISTS idx_barrier_outcomes_down_first ON qbn.barrier_outcomes(asset_id, time_1, first_significant_time_min) WHERE first_significant_barrier LIKE 'down_%';

-- =============================================================================
-- BARRIER CONFIG TABEL
-- =============================================================================

CREATE TABLE IF NOT EXISTS qbn.barrier_config (
    config_id SERIAL PRIMARY KEY,
    config_name VARCHAR(50) NOT NULL UNIQUE,
    up_barriers REAL[] NOT NULL DEFAULT '{0.25, 0.50, 0.75, 1.00, 1.25, 1.50}',
    down_barriers REAL[] NOT NULL DEFAULT '{0.25, 0.50, 0.75, 1.00, 1.25, 1.50}',
    significant_threshold REAL NOT NULL DEFAULT 0.75,
    max_observation_min SMALLINT NOT NULL DEFAULT 2880,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    CONSTRAINT barrier_config_threshold_range CHECK (significant_threshold BETWEEN 0.1 AND 5.0),
    CONSTRAINT barrier_config_window_range CHECK (max_observation_min BETWEEN 60 AND 10080),
    CONSTRAINT barrier_config_arrays_not_empty CHECK (array_length(up_barriers, 1) > 0 AND array_length(down_barriers, 1) > 0)
);

ALTER TABLE qbn.barrier_config OWNER TO postgres;

COMMENT ON TABLE qbn.barrier_config IS 'Configuratie voor barrier levels en analyse parameters';
COMMENT ON COLUMN qbn.barrier_config.significant_threshold IS 'ATR threshold voor "significant" barrier (gebruikt voor first_significant_*)';
COMMENT ON COLUMN qbn.barrier_config.up_barriers IS 'Array van up barrier levels in ATR units (bijv. {0.25, 0.50, 0.75})';

-- =============================================================================
-- DEFAULT CONFIGURATIES
-- =============================================================================

INSERT INTO qbn.barrier_config (config_name, up_barriers, down_barriers, significant_threshold, max_observation_min, notes) 
VALUES ('default', '{0.25, 0.50, 0.75, 1.00, 1.25, 1.50}', '{0.25, 0.50, 0.75, 1.00, 1.25, 1.50}', 0.75, 2880, 'Standaard symmetrische barriers op 0.25 ATR intervals, 48h window') 
ON CONFLICT (config_name) DO NOTHING;

INSERT INTO qbn.barrier_config (config_name, up_barriers, down_barriers, significant_threshold, max_observation_min, notes) 
VALUES ('trend_following', '{0.50, 0.75, 1.00, 1.50, 2.00}', '{0.25, 0.50, 0.75}', 0.50, 1440, 'Asymmetrisch: grotere up targets, tightere down stops, 24h window') 
ON CONFLICT (config_name) DO NOTHING;

INSERT INTO qbn.barrier_config (config_name, up_barriers, down_barriers, significant_threshold, max_observation_min, notes) 
VALUES ('scalping', '{0.15, 0.25, 0.35, 0.50}', '{0.15, 0.25, 0.35, 0.50}', 0.25, 240, 'Kleine barriers voor scalping, 4h window') 
ON CONFLICT (config_name) DO NOTHING;

-- =============================================================================
-- VALIDATIE
-- =============================================================================

DO $$
DECLARE
    outcomes_exists BOOLEAN;
    config_exists BOOLEAN;
BEGIN
    SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'qbn' AND table_name = 'barrier_outcomes') INTO outcomes_exists;
    SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'qbn' AND table_name = 'barrier_config') INTO config_exists;
    IF NOT outcomes_exists THEN RAISE EXCEPTION 'barrier_outcomes tabel niet aangemaakt'; END IF;
    IF NOT config_exists THEN RAISE EXCEPTION 'barrier_config tabel niet aangemaakt'; END IF;
    RAISE NOTICE 'Migratie succesvol: barrier_outcomes en barrier_config aangemaakt';
END $$;

COMMIT;
