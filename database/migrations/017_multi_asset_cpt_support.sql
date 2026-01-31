-- Migration: 017_multi_asset_cpt_support.sql
-- Beschrijving: Uitbreiding qbn.cpt_cache voor multi-asset/composite CPT ondersteuning
-- Datum: 2026-01-09
--
-- REASON: CPTs kunnen nu gegenereerd worden voor:
--   - Enkelvoudig asset (scope_type='single')
--   - Composite van meerdere assets (scope_type='composite', bijv. 'top_10')
--   - Globaal over alle data (scope_type='global')
--
-- De source_assets kolom trackt welke asset IDs zijn gebruikt voor training.

-- =============================================================================
-- 1. EXTEND qbn.cpt_cache met scope kolommen
-- =============================================================================

-- Scope type: single (1 asset), composite (meerdere), global (alle)
ALTER TABLE qbn.cpt_cache
ADD COLUMN IF NOT EXISTS scope_type VARCHAR(20) DEFAULT 'single';

-- Scope key: unieke identifier zoals 'asset_1', 'top_10', 'all_assets'
ALTER TABLE qbn.cpt_cache
ADD COLUMN IF NOT EXISTS scope_key VARCHAR(64);

-- Source assets: array van asset IDs gebruikt voor deze CPT
ALTER TABLE qbn.cpt_cache
ADD COLUMN IF NOT EXISTS source_assets INTEGER[];

-- =============================================================================
-- 2. CONSTRAINT voor geldige scope types
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'valid_scope_type'
    ) THEN
        ALTER TABLE qbn.cpt_cache
        ADD CONSTRAINT valid_scope_type
        CHECK (scope_type IN ('single', 'composite', 'global'));
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- 3. BACKFILL bestaande records
-- =============================================================================

-- Bestaande single-asset CPTs krijgen correcte scope waarden
UPDATE qbn.cpt_cache
SET
    scope_type = 'single',
    scope_key = 'asset_' || asset_id::text,
    source_assets = ARRAY[asset_id]
WHERE scope_key IS NULL
  AND asset_id > 0;

-- =============================================================================
-- 4. INDEXES voor scope lookups
-- =============================================================================

-- Index voor scope_key queries (meest gebruikte lookup)
CREATE INDEX IF NOT EXISTS idx_cpt_cache_scope_key
ON qbn.cpt_cache(scope_key, model_version);

-- GIN index voor source_assets queries ("welke CPTs bevatten asset X?")
CREATE INDEX IF NOT EXISTS idx_cpt_cache_source_assets
ON qbn.cpt_cache USING GIN(source_assets);

-- Composite index voor scope_type filtering
CREATE INDEX IF NOT EXISTS idx_cpt_cache_scope_type
ON qbn.cpt_cache(scope_type, generated_at DESC);

-- =============================================================================
-- 5. ASSET CPT MAPPING tabel
-- =============================================================================

-- Configuratie welke CPT scope een asset moet gebruiken voor inference
CREATE TABLE IF NOT EXISTS qbn.asset_cpt_mapping (
    asset_id        INTEGER PRIMARY KEY,
    preferred_scope VARCHAR(64),           -- NULL = gebruik eigen asset CPT indien beschikbaar
    fallback_scope  VARCHAR(64) DEFAULT 'global',
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Foreign key naar kfl.symbols (optioneel)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kfl' AND table_name = 'symbols') THEN
        ALTER TABLE qbn.asset_cpt_mapping
            ADD CONSTRAINT fk_asset_cpt_mapping_asset
            FOREIGN KEY (asset_id)
            REFERENCES kfl.symbols(asset_id)
            ON DELETE CASCADE;
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- 6. COMMENTS
-- =============================================================================

COMMENT ON COLUMN qbn.cpt_cache.scope_type IS
'Type CPT: single (1 asset), composite (meerdere assets), global (alle data)';

COMMENT ON COLUMN qbn.cpt_cache.scope_key IS
'Unieke scope identifier: asset_1, top_10, all_assets, etc.';

COMMENT ON COLUMN qbn.cpt_cache.source_assets IS
'Array van asset IDs die zijn gebruikt voor CPT training. Traceerbaarheid.';

COMMENT ON TABLE qbn.asset_cpt_mapping IS
'Configuratie welke CPT scope een asset moet gebruiken voor inference.
preferred_scope = NULL betekent asset-specifieke CPT indien beschikbaar.
fallback_scope wordt gebruikt als preferred niet beschikbaar is.';

COMMENT ON COLUMN qbn.asset_cpt_mapping.preferred_scope IS
'Gewenste CPT scope (bijv. top_10). NULL = gebruik asset-specifieke CPT.';

COMMENT ON COLUMN qbn.asset_cpt_mapping.fallback_scope IS
'Fallback CPT scope als preferred niet beschikbaar. Default: global.';
