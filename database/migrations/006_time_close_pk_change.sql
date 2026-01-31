-- ============================================================================
-- Migratie: time_close PRIMARY KEY Wijziging
-- Datum: 2025-12-12
-- Doel: Wijzig PK van (asset_id, time) naar (asset_id, time_1) en verwijder time kolom
-- ============================================================================
--
-- WAARSCHUWING: Dit script vereist een MAINTENANCE WINDOW!
-- 
-- INSTRUCTIES:
-- 1. Zorg dat 005_time_close_columns.sql succesvol is uitgevoerd
-- 2. Stop alle applicaties die naar de MTF tabellen schrijven
-- 3. Voer dit script uit
-- 4. Valideer resultaten
-- 5. Herstart applicaties
--
-- RISICO's:
-- - Lange lock tijd op grote tabellen
-- - TimescaleDB hypertables kunnen extra handling vereisen
-- - Alle queries die 'time' gebruiken zullen falen na dit script
--
-- ============================================================================

-- ============================================================================
-- PRE-CHECK: Valideer dat voorwaarden zijn voldaan
-- ============================================================================

DO $$
DECLARE
  row_count INTEGER;
BEGIN
  -- Check dat time_close kolommen bestaan
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals' AND column_name = 'time_close_1'
  ) THEN
    RAISE EXCEPTION 'FOUT: Voer eerst 005_time_close_columns.sql uit';
  END IF;
  
  -- Check hoeveel rijen er zijn
  SELECT COUNT(*) INTO row_count FROM qbn.ml_multi_timeframe_signals;
  
  IF row_count = 0 THEN
    RAISE NOTICE 'INFO: Tabel is leeg - backfill checks overgeslagen';
  ELSE
    -- Check dat alle time_close_1 gevuld zijn (alleen als er data is)
    IF EXISTS (
      SELECT 1 FROM qbn.ml_multi_timeframe_signals WHERE time_close_1 IS NULL LIMIT 1
    ) THEN
      RAISE EXCEPTION 'FOUT: Er zijn nog rijen zonder time_close_1. Voer backfill uit.';
    END IF;
    
    -- Check dat alle time_1 gevuld zijn
    IF EXISTS (
      SELECT 1 FROM qbn.ml_multi_timeframe_signals WHERE time_1 IS NULL LIMIT 1
    ) THEN
      RAISE EXCEPTION 'FOUT: Er zijn nog rijen zonder time_1. Deze kunnen niet in nieuwe PK.';
    END IF;
  END IF;
  
  RAISE NOTICE 'OK: Pre-checks geslaagd (% rijen in tabel)', row_count;
END $$;

-- ============================================================================
-- FASE 1.3: Hypertable her-partitioneren (tabel is leeg)
-- ============================================================================
-- 
-- PROBLEEM: ml_multi_timeframe_signals is een TimescaleDB hypertable
-- gepartitioneerd op 'time'. PK moet de partitie-kolom bevatten.
-- 
-- OPLOSSING: Aangezien de tabel leeg is, droppen we de hypertable en
-- maken we een nieuwe aan met time_1 als partitie-kolom.
--

-- Stap 1: Drop de bestaande hypertable (ALLEEN ALS LEEG!)
DO $$
DECLARE
  row_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO row_count FROM qbn.ml_multi_timeframe_signals;
  IF row_count > 0 THEN
    RAISE EXCEPTION 'FOUT: Tabel bevat % rijen. Kan niet droppen. Gebruik backup/restore procedure.', row_count;
  END IF;
END $$;

DROP TABLE IF EXISTS qbn.ml_multi_timeframe_signals;

-- Stap 2: Maak nieuwe tabel aan met correcte structuur
CREATE TABLE qbn.ml_multi_timeframe_signals (
    asset_id INTEGER NOT NULL,
    -- time kolom VERWIJDERD - was de oude PK
    
    -- Daily (D) timeframe
    time_d TIMESTAMPTZ NOT NULL,
    time_close_d TIMESTAMPTZ,
    rsi_signal_d SMALLINT,
    macd_signal_d SMALLINT,
    bb_signal_d SMALLINT,
    keltner_signal_d SMALLINT,
    atr_signal_d SMALLINT,
    
    -- 4-hour (240) timeframe
    time_240 TIMESTAMPTZ NOT NULL,
    time_close_240 TIMESTAMPTZ,
    rsi_signal_240 SMALLINT,
    macd_signal_240 SMALLINT,
    bb_signal_240 SMALLINT,
    keltner_signal_240 SMALLINT,
    atr_signal_240 SMALLINT,
    
    -- 1-hour (60) timeframe
    time_60 TIMESTAMPTZ NOT NULL,
    time_close_60 TIMESTAMPTZ,
    rsi_signal_60 SMALLINT,
    macd_signal_60 SMALLINT,
    bb_signal_60 SMALLINT,
    keltner_signal_60 SMALLINT,
    atr_signal_60 SMALLINT,
    
    -- 1-minute (1) timeframe - NIEUWE PARTITIE KOLOM
    time_1 TIMESTAMPTZ NOT NULL,
    time_close_1 TIMESTAMPTZ,
    rsi_signal_1 SMALLINT,
    macd_signal_1 SMALLINT,
    bb_signal_1 SMALLINT,
    keltner_signal_1 SMALLINT,
    atr_signal_1 SMALLINT,
    
    -- Concordance metrics
    concordance_raw INTEGER NOT NULL DEFAULT 0,
    concordance_score REAL,
    signal_strength REAL,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Outcome kolommen (toegevoegd in eerdere migratie)
    outcome_1h SMALLINT,
    outcome_4h SMALLINT,
    outcome_1d SMALLINT,
    return_1h_pct REAL,
    return_4h_pct REAL,
    return_1d_pct REAL,
    atr_at_signal REAL,
    
    -- PRIMARY KEY op (asset_id, time_1)
    PRIMARY KEY (asset_id, time_1)
);

-- Stap 3: Constraints toevoegen
ALTER TABLE qbn.ml_multi_timeframe_signals
  ADD CONSTRAINT chk_outcome_1h CHECK (outcome_1h BETWEEN -3 AND 3),
  ADD CONSTRAINT chk_outcome_4h CHECK (outcome_4h BETWEEN -3 AND 3),
  ADD CONSTRAINT chk_outcome_1d CHECK (outcome_1d BETWEEN -3 AND 3),
  ADD CONSTRAINT chk_atr_positive CHECK (atr_at_signal IS NULL OR atr_at_signal >= 0);

-- Stap 4: Maak hypertable met time_1 als partitie-kolom
SELECT create_hypertable(
  'qbn.ml_multi_timeframe_signals',
  'time_1',
  chunk_time_interval => INTERVAL '7 days',
  if_not_exists => TRUE
);

-- Stap 5: Comments toevoegen
COMMENT ON TABLE qbn.ml_multi_timeframe_signals IS 
'Multi-timeframe signals voor QBN inference. Gepartitioneerd op time_1 (1-minute candle open time).';

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.time_1 IS 
'Open time van 1-minute candle. Dit is de PRIMARY KEY samen met asset_id en de partitie-kolom.';

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.time_close_1 IS 
'Close time van 1-minute candle: time_1 + 1 minute. Expliciet opgeslagen voor lookahead bias preventie.';

-- ============================================================================
-- FASE 1.4: Indexes aanmaken
-- ============================================================================

-- Index op time_1 (hypertable auto-indexeert PK al)
CREATE INDEX IF NOT EXISTS idx_mtf_signals_asset_time1 
  ON qbn.ml_multi_timeframe_signals (asset_id, time_1 DESC);

-- Outcome training indexes
CREATE INDEX IF NOT EXISTS idx_mtf_outcome_1h_training 
  ON qbn.ml_multi_timeframe_signals (asset_id, time_1, outcome_1h) 
  WHERE outcome_1h IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mtf_outcome_4h_training 
  ON qbn.ml_multi_timeframe_signals (asset_id, time_1, outcome_4h) 
  WHERE outcome_4h IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mtf_outcome_1d_training 
  ON qbn.ml_multi_timeframe_signals (asset_id, time_1, outcome_1d) 
  WHERE outcome_1d IS NOT NULL;

-- All outcomes index
CREATE INDEX IF NOT EXISTS idx_mtf_all_outcomes 
  ON qbn.ml_multi_timeframe_signals (asset_id, time_1) 
  WHERE outcome_1h IS NOT NULL AND outcome_4h IS NOT NULL AND outcome_1d IS NOT NULL;

-- ATR signal index
CREATE INDEX IF NOT EXISTS idx_mtf_atr_signal 
  ON qbn.ml_multi_timeframe_signals (asset_id, time_1, atr_at_signal) 
  WHERE atr_at_signal IS NOT NULL;

-- Update staging index
DROP INDEX IF EXISTS staging.idx_staging_mtf_signals_pk;
CREATE INDEX IF NOT EXISTS idx_staging_mtf_signals_pk 
  ON staging.mtf_signals (asset_id, time_1);

-- ============================================================================
-- FASE 1.5: time kolom verwijderen uit andere tabellen
-- ============================================================================

-- MTF cache tabel
ALTER TABLE qbn.ml_multi_timeframe_signals_cache DROP COLUMN IF EXISTS time;

-- Staging MTF tabel
ALTER TABLE staging.mtf_signals DROP COLUMN IF EXISTS time;

-- ============================================================================
-- POST-CHECK: Valideer structuur
-- ============================================================================

DO $$
BEGIN
  -- Check dat time kolom NIET bestaat
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals' AND column_name = 'time'
  ) THEN
    RAISE EXCEPTION 'FOUT: time kolom nog aanwezig in ml_multi_timeframe_signals';
  END IF;
  
  -- Check dat time_1 WEL bestaat
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals' AND column_name = 'time_1'
  ) THEN
    RAISE EXCEPTION 'FOUT: time_1 kolom ontbreekt in ml_multi_timeframe_signals';
  END IF;
  
  -- Check dat het een hypertable is
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables
    WHERE hypertable_schema = 'qbn' AND hypertable_name = 'ml_multi_timeframe_signals'
  ) THEN
    RAISE EXCEPTION 'FOUT: Tabel is geen hypertable';
  END IF;
  
  -- Check time_close kolommen bestaan
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals' AND column_name = 'time_close_1'
  ) THEN
    RAISE EXCEPTION 'FOUT: time_close_1 kolom ontbreekt';
  END IF;
  
  RAISE NOTICE 'OK: Post-checks geslaagd - hypertable opnieuw aangemaakt met time_1 partitionering';
END $$;

-- ============================================================================
-- STATISTIEKEN
-- ============================================================================

SELECT 
  'qbn.ml_multi_timeframe_signals' as tabel,
  COUNT(*) as rijen,
  MIN(time_1) as oudste,
  MAX(time_1) as nieuwste
FROM qbn.ml_multi_timeframe_signals;

