-- Migration: 002_add_outcome_targets.sql
-- Beschrijving: Voeg outcome target kolommen toe voor CPT training op echte koersuitkomsten
-- Datum: 2025-12-05
--
-- REASON: CPT's moeten getraind worden op echte koersuitkomsten, niet op signal-signal correlaties.
-- Deze migratie voegt 7 ATR-relatieve outcome states toe voor 3 prediction horizons.
--
-- TARGET STATES (-3 tot +3):
--   -3: Strong_Bearish  (return < -2.0 * ATR)
--   -2: Bearish         (-2.0*ATR <= return < -1.0*ATR)
--   -1: Slight_Bearish  (-1.0*ATR <= return < -0.5*ATR)
--    0: Neutral         (-0.5*ATR <= return < +0.5*ATR)
--   +1: Slight_Bullish  (+0.5*ATR <= return < +1.0*ATR)
--   +2: Bullish         (+1.0*ATR <= return < +2.0*ATR)
--   +3: Strong_Bullish  (return >= +2.0*ATR)

-- =============================================================================
-- 1. EXTEND qbn.ml_multi_timeframe_signals (Hypertable)
-- =============================================================================

-- Outcome kolommen (discretized naar 7 ATR-bins)
ALTER TABLE qbn.ml_multi_timeframe_signals 
    ADD COLUMN IF NOT EXISTS outcome_1h SMALLINT,
    ADD COLUMN IF NOT EXISTS outcome_4h SMALLINT,
    ADD COLUMN IF NOT EXISTS outcome_1d SMALLINT;

-- Ruwe return percentages (voor analyse en herberekening bij threshold wijzigingen)
ALTER TABLE qbn.ml_multi_timeframe_signals 
    ADD COLUMN IF NOT EXISTS return_1h_pct REAL,
    ADD COLUMN IF NOT EXISTS return_4h_pct REAL,
    ADD COLUMN IF NOT EXISTS return_1d_pct REAL;

-- ATR op moment van signal (nodig voor discretisatie)
ALTER TABLE qbn.ml_multi_timeframe_signals 
    ADD COLUMN IF NOT EXISTS atr_at_signal REAL;

-- Constraints voor outcome values (-3 tot +3)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'valid_outcome_1h'
    ) THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals 
            ADD CONSTRAINT valid_outcome_1h CHECK (outcome_1h IS NULL OR (outcome_1h >= -3 AND outcome_1h <= 3));
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'valid_outcome_4h'
    ) THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals 
            ADD CONSTRAINT valid_outcome_4h CHECK (outcome_4h IS NULL OR (outcome_4h >= -3 AND outcome_4h <= 3));
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'valid_outcome_1d'
    ) THEN
        ALTER TABLE qbn.ml_multi_timeframe_signals 
            ADD CONSTRAINT valid_outcome_1d CHECK (outcome_1d IS NULL OR (outcome_1d >= -3 AND outcome_1d <= 3));
    END IF;
END $$;

-- =============================================================================
-- 2. EXTEND qbn.ml_multi_timeframe_signals_cache (Real-time cache)
-- =============================================================================

-- ATR kolom voor real-time (outcomes blijven NULL in cache - worden later berekend)
ALTER TABLE qbn.ml_multi_timeframe_signals_cache 
    ADD COLUMN IF NOT EXISTS atr_at_signal REAL;

-- Outcome kolommen (blijven NULL in real-time, alleen voor schema compatibiliteit)
ALTER TABLE qbn.ml_multi_timeframe_signals_cache 
    ADD COLUMN IF NOT EXISTS outcome_1h SMALLINT,
    ADD COLUMN IF NOT EXISTS outcome_4h SMALLINT,
    ADD COLUMN IF NOT EXISTS outcome_1d SMALLINT,
    ADD COLUMN IF NOT EXISTS return_1h_pct REAL,
    ADD COLUMN IF NOT EXISTS return_4h_pct REAL,
    ADD COLUMN IF NOT EXISTS return_1d_pct REAL;

-- =============================================================================
-- 3. EXTEND staging.mtf_signals (Staging tabel voor bulk loads)
-- =============================================================================

DO $$
BEGIN
    -- Check of staging schema bestaat
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'staging') THEN
        -- Check of tabel bestaat
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'staging' AND table_name = 'mtf_signals') THEN
            ALTER TABLE staging.mtf_signals 
                ADD COLUMN IF NOT EXISTS outcome_1h SMALLINT,
                ADD COLUMN IF NOT EXISTS outcome_4h SMALLINT,
                ADD COLUMN IF NOT EXISTS outcome_1d SMALLINT,
                ADD COLUMN IF NOT EXISTS return_1h_pct REAL,
                ADD COLUMN IF NOT EXISTS return_4h_pct REAL,
                ADD COLUMN IF NOT EXISTS return_1d_pct REAL,
                ADD COLUMN IF NOT EXISTS atr_at_signal REAL;
        END IF;
    END IF;
END $$;

-- =============================================================================
-- 4. EXTEND qbn.cpt_cache (CPT cache met horizon support)
-- =============================================================================

-- Target horizon kolom voor aparte CPT's per prediction horizon
ALTER TABLE qbn.cpt_cache 
    ADD COLUMN IF NOT EXISTS target_horizon VARCHAR(8);

-- REASON: We willen aparte CPT's per horizon (1h, 4h, 1d) kunnen cachen
-- Update bestaande entries naar 'legacy' of NULL
COMMENT ON COLUMN qbn.cpt_cache.target_horizon IS 
'Prediction horizon: 1h, 4h, 1d, of NULL voor legacy/non-prediction nodes';

-- =============================================================================
-- 5. INDEXES voor outcome queries
-- =============================================================================

-- Index voor outcome backfill queries (vind records zonder outcomes)
CREATE INDEX IF NOT EXISTS idx_mtf_signals_outcome_null 
    ON qbn.ml_multi_timeframe_signals (asset_id, time) 
    WHERE outcome_1h IS NULL;

-- Index voor training queries (vind records met outcomes)
CREATE INDEX IF NOT EXISTS idx_mtf_signals_outcome_filled 
    ON qbn.ml_multi_timeframe_signals (asset_id, time) 
    WHERE outcome_1h IS NOT NULL;

-- =============================================================================
-- 6. OUTCOME STATE MAPPING TABEL
-- =============================================================================

CREATE TABLE IF NOT EXISTS qbn.outcome_state_mapping (
    state_id    SMALLINT PRIMARY KEY,
    state_name  VARCHAR(32) NOT NULL,
    atr_lower   REAL,           -- Lower bound in ATR units (NULL = no lower bound)
    atr_upper   REAL,           -- Upper bound in ATR units (NULL = no upper bound)
    description TEXT
);

-- Insert state definitions
INSERT INTO qbn.outcome_state_mapping (state_id, state_name, atr_lower, atr_upper, description)
VALUES 
    (-3, 'Strong_Bearish', NULL, -2.0, 'return < -2.0 * ATR'),
    (-2, 'Bearish', -2.0, -1.0, '-2.0*ATR <= return < -1.0*ATR'),
    (-1, 'Slight_Bearish', -1.0, -0.5, '-1.0*ATR <= return < -0.5*ATR'),
    (0, 'Neutral', -0.5, 0.5, '-0.5*ATR <= return < +0.5*ATR'),
    (1, 'Slight_Bullish', 0.5, 1.0, '+0.5*ATR <= return < +1.0*ATR'),
    (2, 'Bullish', 1.0, 2.0, '+1.0*ATR <= return < +2.0*ATR'),
    (3, 'Strong_Bullish', 2.0, NULL, 'return >= +2.0*ATR')
ON CONFLICT (state_id) DO UPDATE SET
    state_name = EXCLUDED.state_name,
    atr_lower = EXCLUDED.atr_lower,
    atr_upper = EXCLUDED.atr_upper,
    description = EXCLUDED.description;

-- =============================================================================
-- 7. COMMENTS
-- =============================================================================

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.outcome_1h IS 
'Koersuitkomst over 1 uur, gediscretiseerd naar 7 ATR-bins (-3 tot +3). NULL = nog niet berekend.';

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.outcome_4h IS 
'Koersuitkomst over 4 uur, gediscretiseerd naar 7 ATR-bins (-3 tot +3). NULL = nog niet berekend.';

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.outcome_1d IS 
'Koersuitkomst over 24 uur, gediscretiseerd naar 7 ATR-bins (-3 tot +3). NULL = nog niet berekend.';

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.return_1h_pct IS 
'Ruwe return percentage over 1 uur: (close_future - close_now) / close_now * 100';

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.return_4h_pct IS 
'Ruwe return percentage over 4 uur';

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.return_1d_pct IS 
'Ruwe return percentage over 24 uur';

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.atr_at_signal IS 
'ATR waarde (in %) op moment van signal, gebruikt voor outcome discretisatie';

COMMENT ON TABLE qbn.outcome_state_mapping IS 
'Mapping van outcome state IDs naar namen en ATR bounds voor 7-state discretisatie';
