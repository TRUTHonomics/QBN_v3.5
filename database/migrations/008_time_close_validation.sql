-- ============================================================================
-- Migratie: time_close Validatie en Cleanup
-- Datum: 2025-12-12
-- Doel: Validatie queries en index updates na time_close migratie
-- ============================================================================
--
-- INSTRUCTIES:
-- 1. Voer dit script uit NA 005, 006, en 007 migraties
-- 2. Dit script valideert dat alles correct is gemigreerd
-- 3. Updates indexes die 'time' gebruikten naar 'time_1'
--
-- ============================================================================

-- ============================================================================
-- FASE 5.1: Validatie queries
-- ============================================================================

-- Valideer dat time_close correct is gevuld in signals_current
SELECT 
  'kfl.signals_current - time_close NULL check' as check_name,
  COUNT(*) as totaal,
  COUNT(time_close) as met_time_close,
  COUNT(*) - COUNT(time_close) as ontbrekend
FROM kfl.signals_current;

-- Valideer dat time_close_X correct is gevuld in MTF tabellen
SELECT 
  'qbn.ml_multi_timeframe_signals - time_close_X NULL check' as check_name,
  COUNT(*) as totaal,
  COUNT(time_close_1) as met_time_close_1,
  COUNT(time_close_60) as met_time_close_60,
  COUNT(time_close_240) as met_time_close_240,
  COUNT(time_close_d) as met_time_close_d
FROM qbn.ml_multi_timeframe_signals;

-- Valideer dat time_close waarden correct zijn berekend
-- (steekproef - check eerste 10 afwijkingen)
SELECT 
  asset_id, 
  time_1, 
  time_close_1,
  time_1 + INTERVAL '1 minute' as expected_time_close_1,
  CASE WHEN time_close_1 = time_1 + INTERVAL '1 minute' THEN 'OK' ELSE 'FOUT' END as status
FROM qbn.ml_multi_timeframe_signals 
WHERE time_close_1 != time_1 + INTERVAL '1 minute'
LIMIT 10;

-- Check dat time kolom niet meer bestaat
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'qbn' AND table_name = 'ml_multi_timeframe_signals' AND column_name = 'time'
  ) THEN
    RAISE WARNING 'WAARSCHUWING: time kolom bestaat nog in qbn.ml_multi_timeframe_signals';
  ELSE
    RAISE NOTICE 'OK: time kolom succesvol verwijderd uit qbn.ml_multi_timeframe_signals';
  END IF;
END $$;

-- ============================================================================
-- FASE 5.2: Index updates (time → time_1)
-- ============================================================================

-- REASON: Deze indexes gebruikten (asset_id, time) en moeten nu (asset_id, time_1) gebruiken

-- Verwijder oude indexes (indien ze nog bestaan na PK wijziging)
DROP INDEX IF EXISTS qbn.idx_mtf_outcome_1h_training;
DROP INDEX IF EXISTS qbn.idx_mtf_outcome_4h_training;
DROP INDEX IF EXISTS qbn.idx_mtf_outcome_1d_training;
DROP INDEX IF EXISTS qbn.idx_mtf_all_outcomes;
DROP INDEX IF EXISTS qbn.idx_mtf_atr_signal;

-- Maak nieuwe indexes met time_1
CREATE INDEX IF NOT EXISTS idx_mtf_outcome_1h_training
ON qbn.ml_multi_timeframe_signals (asset_id, time_1, outcome_1h)
WHERE outcome_1h IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mtf_outcome_4h_training
ON qbn.ml_multi_timeframe_signals (asset_id, time_1, outcome_4h)
WHERE outcome_4h IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mtf_outcome_1d_training
ON qbn.ml_multi_timeframe_signals (asset_id, time_1, outcome_1d)
WHERE outcome_1d IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mtf_all_outcomes
ON qbn.ml_multi_timeframe_signals (asset_id, time_1)
WHERE outcome_1h IS NOT NULL
  AND outcome_4h IS NOT NULL
  AND outcome_1d IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mtf_atr_signal
ON qbn.ml_multi_timeframe_signals (asset_id, time_1, atr_at_signal)
WHERE atr_at_signal IS NOT NULL;

-- ============================================================================
-- FASE 5.3: Toon overzicht van alle indexes
-- ============================================================================

SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'qbn'
AND tablename = 'ml_multi_timeframe_signals'
ORDER BY indexname;

-- ============================================================================
-- FASE 5.4: Test query voor validatie
-- ============================================================================

-- Deze query moet Index Scan gebruiken (niet Seq Scan)
EXPLAIN (ANALYZE, COSTS OFF)
SELECT asset_id, time_1, outcome_1h
FROM qbn.ml_multi_timeframe_signals
WHERE asset_id = 1
AND outcome_1h IS NOT NULL
ORDER BY time_1 DESC
LIMIT 100;

-- ============================================================================
-- SAMENVATTING
-- ============================================================================

SELECT 'MIGRATIE VALIDATIE VOLTOOID' as status;

-- Check kolom structuur
SELECT 
  table_schema || '.' || table_name as tabel,
  column_name,
  data_type
FROM information_schema.columns
WHERE table_schema IN ('kfl', 'qbn')
  AND table_name IN ('signals_current', 'ml_multi_timeframe_signals', 'ml_multi_timeframe_signals_cache')
  AND column_name LIKE 'time%'
ORDER BY table_schema, table_name, column_name;

