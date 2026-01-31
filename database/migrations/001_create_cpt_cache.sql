-- Migration: 001_create_cpt_cache.sql
-- Beschrijving: Creëer qbn.cpt_cache tabel voor persistente CPT opslag
-- Datum: 2025-12-05
--
-- REASON: CPT's on-the-fly genereren kost ~200-500ms per asset.
-- Met database cache: ~5-10ms laden → geschikt voor live trading.

-- Ensure qbn schema exists
CREATE SCHEMA IF NOT EXISTS qbn;

-- CPT Cache tabel
CREATE TABLE IF NOT EXISTS qbn.cpt_cache (
    asset_id        INTEGER NOT NULL,
    node_name       VARCHAR(64) NOT NULL,
    cpt_data        JSONB NOT NULL,
    lookback_days   INTEGER DEFAULT NULL,  -- NULL = alle beschikbare data
    observations    INTEGER,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version_hash    VARCHAR(32),
    
    PRIMARY KEY (asset_id, node_name)
);

-- Indexes voor queries
CREATE INDEX IF NOT EXISTS idx_cpt_cache_generated ON qbn.cpt_cache (generated_at);
CREATE INDEX IF NOT EXISTS idx_cpt_cache_asset ON qbn.cpt_cache (asset_id);

-- Foreign key naar kfl.symbols (optioneel, alleen als tabel bestaat)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kfl' AND table_name = 'symbols') THEN
        ALTER TABLE qbn.cpt_cache
            ADD CONSTRAINT fk_cpt_cache_asset
            FOREIGN KEY (asset_id)
            REFERENCES kfl.symbols(asset_id)
            ON DELETE CASCADE;
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Comment
COMMENT ON TABLE qbn.cpt_cache IS 
'Persistente cache voor Conditional Probability Tables (CPT).
Gegenereerd uit qbn.ml_multi_timeframe_signals hypertable.
Update strategie: dagelijks herberekenen of bij significante data wijzigingen.
lookback_days = NULL betekent alle beschikbare data gebruikt.';

COMMENT ON COLUMN qbn.cpt_cache.cpt_data IS 'CPT als JSONB: {node, parents, states, probabilities/conditional_probabilities, type, ...}';
COMMENT ON COLUMN qbn.cpt_cache.lookback_days IS 'Aantal dagen historische data gebruikt (NULL = alle data)';
COMMENT ON COLUMN qbn.cpt_cache.observations IS 'Aantal observaties gebruikt voor CPT generatie';
COMMENT ON COLUMN qbn.cpt_cache.version_hash IS 'Hash voor change detection en cache invalidatie';
