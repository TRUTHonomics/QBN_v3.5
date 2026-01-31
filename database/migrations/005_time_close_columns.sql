-- ============================================================================
-- Migratie: time_close Kolommen Implementatie
-- Datum: 2025-12-12
-- Doel: Toevoegen van time_close kolommen aan signals en MTF tabellen
-- ============================================================================
--
-- INSTRUCTIES:
-- 1. Voer dit script uit in een maintenance window
-- 2. Fase 1.1 en 1.2 kunnen veilig worden uitgevoerd (non-breaking)
-- 3. Fase 1.3-1.5 vereisen downtime vanwege PK wijziging
-- 4. Voer na succes 006_time_close_pk_change.sql uit
--
-- ============================================================================

-- ============================================================================
-- FASE 1.1: Kolommen toevoegen
-- ============================================================================

-- signals_current: time_close kolom
ALTER TABLE kfl.signals_current ADD COLUMN IF NOT EXISTS time_close TIMESTAMPTZ;

COMMENT ON COLUMN kfl.signals_current.time_close IS 
'Close time van de candle: time + interval_duration. Automatisch berekend door trigger.';

-- MTF cache tabel: time_close_X kolommen
ALTER TABLE qbn.ml_multi_timeframe_signals_cache 
  ADD COLUMN IF NOT EXISTS time_close_d TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_240 TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_60 TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_1 TIMESTAMPTZ;

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals_cache.time_close_d IS 'Close time Daily candle: time_d + 1 day';
COMMENT ON COLUMN qbn.ml_multi_timeframe_signals_cache.time_close_240 IS 'Close time 4-hour candle: time_240 + 240 minutes';
COMMENT ON COLUMN qbn.ml_multi_timeframe_signals_cache.time_close_60 IS 'Close time 1-hour candle: time_60 + 60 minutes';
COMMENT ON COLUMN qbn.ml_multi_timeframe_signals_cache.time_close_1 IS 'Close time 1-minute candle: time_1 + 1 minute';

-- MTF historische tabel: time_close_X kolommen
ALTER TABLE qbn.ml_multi_timeframe_signals
  ADD COLUMN IF NOT EXISTS time_close_d TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_240 TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_60 TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_1 TIMESTAMPTZ;

COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.time_close_d IS 'Close time Daily candle: time_d + 1 day';
COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.time_close_240 IS 'Close time 4-hour candle: time_240 + 240 minutes';
COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.time_close_60 IS 'Close time 1-hour candle: time_60 + 60 minutes';
COMMENT ON COLUMN qbn.ml_multi_timeframe_signals.time_close_1 IS 'Close time 1-minute candle: time_1 + 1 minute';

-- Staging tabellen: time_close kolommen
ALTER TABLE staging.signals ADD COLUMN IF NOT EXISTS time_close TIMESTAMPTZ;

-- REASON: staging.mtf_signals krijgt time_close_X kolommen
-- time kolom blijft voorlopig bestaan voor backwards compatibility
ALTER TABLE staging.mtf_signals 
  ADD COLUMN IF NOT EXISTS time_close_d TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_240 TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_60 TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS time_close_1 TIMESTAMPTZ;

-- ============================================================================
-- FASE 1.2: Bestaande data backfillen
-- ============================================================================

-- REASON: Vul time_close_X kolommen voor bestaande MTF data
-- Dit is een UPDATE op mogelijk grote tabellen - kan even duren

-- Backfill ml_multi_timeframe_signals (historische tabel)
UPDATE qbn.ml_multi_timeframe_signals SET
  time_close_d = time_d + INTERVAL '1 day',
  time_close_240 = time_240 + INTERVAL '240 minutes',
  time_close_60 = time_60 + INTERVAL '60 minutes',
  time_close_1 = time_1 + INTERVAL '1 minute'
WHERE time_close_1 IS NULL;  -- Alleen rijen zonder time_close

-- Backfill ml_multi_timeframe_signals_cache (cache tabel)
UPDATE qbn.ml_multi_timeframe_signals_cache SET
  time_close_d = time_d + INTERVAL '1 day',
  time_close_240 = time_240 + INTERVAL '240 minutes',
  time_close_60 = time_60 + INTERVAL '60 minutes',
  time_close_1 = time_1 + INTERVAL '1 minute'
WHERE time_close_1 IS NULL;

-- Backfill signals_current time_close
-- REASON: Bereken time_close op basis van interval_min
UPDATE kfl.signals_current SET
  time_close = CASE 
    WHEN interval_min = 'M' THEN time + INTERVAL '1 month'
    WHEN interval_min = 'W' THEN time + INTERVAL '1 week'
    WHEN interval_min = 'D' THEN time + INTERVAL '1 day'
    ELSE time + (interval_min::text::integer * INTERVAL '1 minute')
  END
WHERE time_close IS NULL;

-- ============================================================================
-- VERIFICATIE
-- ============================================================================

-- Check kolommen bestaan
DO $$
BEGIN
  -- signals_current
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'kfl' AND table_name = 'signals_current' AND column_name = 'time_close'
  ) THEN
    RAISE EXCEPTION 'FOUT: kfl.signals_current.time_close niet aangemaakt';
  END IF;
  
  -- MTF cache
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals_cache' AND column_name = 'time_close_1'
  ) THEN
    RAISE EXCEPTION 'FOUT: qbn.ml_multi_timeframe_signals_cache.time_close_1 niet aangemaakt';
  END IF;
  
  -- MTF historisch
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals' AND column_name = 'time_close_1'
  ) THEN
    RAISE EXCEPTION 'FOUT: qbn.ml_multi_timeframe_signals.time_close_1 niet aangemaakt';
  END IF;
  
  RAISE NOTICE 'OK: Alle time_close kolommen succesvol aangemaakt';
END $$;

-- Toon statistieken
SELECT 'kfl.signals_current' as tabel, 
       COUNT(*) as totaal, 
       COUNT(time_close) as met_time_close
FROM kfl.signals_current
UNION ALL
SELECT 'qbn.ml_multi_timeframe_signals_cache', 
       COUNT(*), 
       COUNT(time_close_1)
FROM qbn.ml_multi_timeframe_signals_cache
UNION ALL
SELECT 'qbn.ml_multi_timeframe_signals', 
       COUNT(*), 
       COUNT(time_close_1)
FROM qbn.ml_multi_timeframe_signals;

