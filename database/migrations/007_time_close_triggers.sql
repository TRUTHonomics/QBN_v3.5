-- ============================================================================
-- Migratie: time_close Triggers
-- Datum: 2025-12-12
-- Doel: Trigger aanmaken voor automatische time_close berekening op signals_current
-- ============================================================================
--
-- INSTRUCTIES:
-- 1. Voer dit script uit NA 005_time_close_columns.sql
-- 2. Dit script is non-breaking en kan tijdens productie worden uitgevoerd
--
-- ============================================================================

-- ============================================================================
-- FASE 1.6: Trigger voor signals_current
-- ============================================================================

-- REASON: De functie kfl.calculate_signals_time_close() bestaat al 
-- (aangemaakt in 004_signal_current_triggers.sql voor kfl.signals).
-- We hergebruiken deze functie voor kfl.signals_current.

-- Verwijder eventueel bestaande trigger
DROP TRIGGER IF EXISTS trg_signals_current_calculate_time_close ON kfl.signals_current;

-- Maak trigger aan
CREATE TRIGGER trg_signals_current_calculate_time_close
  BEFORE INSERT OR UPDATE ON kfl.signals_current
  FOR EACH ROW
  EXECUTE FUNCTION kfl.calculate_signals_time_close();

COMMENT ON TRIGGER trg_signals_current_calculate_time_close ON kfl.signals_current IS
'Berekent automatisch time_close kolom bij elke insert/update.
Hergebruikt kfl.calculate_signals_time_close() functie.';

-- ============================================================================
-- VERIFICATIE
-- ============================================================================

DO $$
BEGIN
  -- Check trigger bestaat
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.triggers 
    WHERE trigger_schema = 'kfl' 
      AND event_object_table = 'signals_current' 
      AND trigger_name = 'trg_signals_current_calculate_time_close'
  ) THEN
    RAISE EXCEPTION 'FOUT: Trigger trg_signals_current_calculate_time_close niet aangemaakt';
  END IF;
  
  RAISE NOTICE 'OK: Trigger trg_signals_current_calculate_time_close aangemaakt';
END $$;

-- Toon alle triggers op signals_current
SELECT 
  trigger_name, 
  event_manipulation, 
  action_timing,
  action_statement
FROM information_schema.triggers 
WHERE event_object_table = 'signals_current' 
  AND event_object_schema = 'kfl'
ORDER BY trigger_name;

