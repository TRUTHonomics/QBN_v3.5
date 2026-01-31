-- ============================================================================
-- 011_drop_mtf_outcome_columns.sql
-- 
-- DOEL: Verwijder redundante outcome kolommen uit MTF tabellen.
-- Outcomes worden nu centraal opgeslagen in qbn.signal_outcomes.
--
-- UITVOEREN:
--   psql -h 10.10.10.3 -U postgres -d kflhyper -f 011_drop_mtf_outcome_columns.sql
--
-- VOLGORDE:
--   1. Eerst triggers updaten (008_mtf_triggers_complete.sql)
--   2. Dan dit script uitvoeren
--   3. Dan outcome_backfill.py draaien om qbn.signal_outcomes te vullen
-- ============================================================================

-- REASON: Outcome data wordt 3x redundant opgeslagen in lead/coin/conf tabellen.
-- Na normalisatie naar qbn.signal_outcomes zijn deze kolommen niet meer nodig.

BEGIN;

-- ============================================================================
-- STAP 1: Drop kolommen uit hypertables (archief)
-- ============================================================================

-- kfl.mtf_signals_lead
ALTER TABLE kfl.mtf_signals_lead
    DROP COLUMN IF EXISTS outcome_1h,
    DROP COLUMN IF EXISTS outcome_4h,
    DROP COLUMN IF EXISTS outcome_1d,
    DROP COLUMN IF EXISTS return_1h_pct,
    DROP COLUMN IF EXISTS return_4h_pct,
    DROP COLUMN IF EXISTS return_1d_pct,
    DROP COLUMN IF EXISTS atr_at_signal;

\echo 'Dropped outcome columns from kfl.mtf_signals_lead'

-- kfl.mtf_signals_coin
ALTER TABLE kfl.mtf_signals_coin
    DROP COLUMN IF EXISTS outcome_1h,
    DROP COLUMN IF EXISTS outcome_4h,
    DROP COLUMN IF EXISTS outcome_1d,
    DROP COLUMN IF EXISTS return_1h_pct,
    DROP COLUMN IF EXISTS return_4h_pct,
    DROP COLUMN IF EXISTS return_1d_pct,
    DROP COLUMN IF EXISTS atr_at_signal;

\echo 'Dropped outcome columns from kfl.mtf_signals_coin'

-- kfl.mtf_signals_conf
ALTER TABLE kfl.mtf_signals_conf
    DROP COLUMN IF EXISTS outcome_1h,
    DROP COLUMN IF EXISTS outcome_4h,
    DROP COLUMN IF EXISTS outcome_1d,
    DROP COLUMN IF EXISTS return_1h_pct,
    DROP COLUMN IF EXISTS return_4h_pct,
    DROP COLUMN IF EXISTS return_1d_pct,
    DROP COLUMN IF EXISTS atr_at_signal;

\echo 'Dropped outcome columns from kfl.mtf_signals_conf'

-- ============================================================================
-- STAP 2: Drop kolommen uit current tables (cache)
-- ============================================================================

-- kfl.mtf_signals_current_lead
ALTER TABLE kfl.mtf_signals_current_lead
    DROP COLUMN IF EXISTS outcome_1h,
    DROP COLUMN IF EXISTS outcome_4h,
    DROP COLUMN IF EXISTS outcome_1d,
    DROP COLUMN IF EXISTS return_1h_pct,
    DROP COLUMN IF EXISTS return_4h_pct,
    DROP COLUMN IF EXISTS return_1d_pct,
    DROP COLUMN IF EXISTS atr_at_signal;

\echo 'Dropped outcome columns from kfl.mtf_signals_current_lead'

-- kfl.mtf_signals_current_coin
ALTER TABLE kfl.mtf_signals_current_coin
    DROP COLUMN IF EXISTS outcome_1h,
    DROP COLUMN IF EXISTS outcome_4h,
    DROP COLUMN IF EXISTS outcome_1d,
    DROP COLUMN IF EXISTS return_1h_pct,
    DROP COLUMN IF EXISTS return_4h_pct,
    DROP COLUMN IF EXISTS return_1d_pct,
    DROP COLUMN IF EXISTS atr_at_signal;

\echo 'Dropped outcome columns from kfl.mtf_signals_current_coin'

-- kfl.mtf_signals_current_conf
ALTER TABLE kfl.mtf_signals_current_conf
    DROP COLUMN IF EXISTS outcome_1h,
    DROP COLUMN IF EXISTS outcome_4h,
    DROP COLUMN IF EXISTS outcome_1d,
    DROP COLUMN IF EXISTS return_1h_pct,
    DROP COLUMN IF EXISTS return_4h_pct,
    DROP COLUMN IF EXISTS return_1d_pct,
    DROP COLUMN IF EXISTS atr_at_signal;

\echo 'Dropped outcome columns from kfl.mtf_signals_current_conf'

-- ============================================================================
-- STAP 3: Drop oude partial indexes (indien aanwezig)
-- ============================================================================

DROP INDEX IF EXISTS kfl.idx_mtf_lead_outcome_1h_backfill;
DROP INDEX IF EXISTS kfl.idx_mtf_lead_outcome_4h_backfill;
DROP INDEX IF EXISTS kfl.idx_mtf_lead_outcome_1d_backfill;
DROP INDEX IF EXISTS kfl.idx_mtf_coin_outcome_1h_backfill;
DROP INDEX IF EXISTS kfl.idx_mtf_coin_outcome_4h_backfill;
DROP INDEX IF EXISTS kfl.idx_mtf_coin_outcome_1d_backfill;
DROP INDEX IF EXISTS kfl.idx_mtf_conf_outcome_1h_backfill;
DROP INDEX IF EXISTS kfl.idx_mtf_conf_outcome_4h_backfill;
DROP INDEX IF EXISTS kfl.idx_mtf_conf_outcome_1d_backfill;

\echo 'Dropped old partial indexes'

-- ============================================================================
-- STAP 4: Verificatie
-- ============================================================================

DO $$
DECLARE
    col_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_schema = 'kfl'
      AND table_name LIKE 'mtf_signals%'
      AND column_name IN ('outcome_1h', 'outcome_4h', 'outcome_1d', 
                          'return_1h_pct', 'return_4h_pct', 'return_1d_pct', 
                          'atr_at_signal');
    
    IF col_count = 0 THEN
        RAISE NOTICE 'SUCCESS: Alle outcome kolommen zijn verwijderd uit MTF tabellen';
    ELSE
        RAISE WARNING 'WARNING: Er zijn nog % outcome kolommen aanwezig', col_count;
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- VERIFICATIE QUERY (handmatig uitvoeren)
-- ============================================================================
-- SELECT table_name, column_name
-- FROM information_schema.columns
-- WHERE table_schema = 'kfl'
--   AND table_name LIKE 'mtf_signals%'
--   AND column_name IN ('outcome_1h', 'outcome_4h', 'outcome_1d', 
--                       'return_1h_pct', 'return_4h_pct', 'return_1d_pct', 
--                       'atr_at_signal')
-- ORDER BY table_name, column_name;
-- Expected: 0 rows

