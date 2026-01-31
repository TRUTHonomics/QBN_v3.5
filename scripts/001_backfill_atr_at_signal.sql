-- =============================================================================
-- 001_backfill_atr_at_signal.sql
-- =============================================================================
-- Backfill script om NULL atr_at_signal waarden te vullen vanuit
-- kfl.indicators_unified_cache.atr_14 voor alle drie MTF tabellen.
--
-- USAGE:
--   psql -h 10.10.10.1 -U postgres -d KFLhyper -f 001_backfill_atr_at_signal.sql
--
-- OF via MCP execute_sql tool (in batches voor grote tabellen)
--
-- REASON: De MTF triggers kopieerden voorheen OLD.atr_at_signal die altijd NULL was.
--         Dit script vult de historische records met de correcte ATR waarde.
--
-- WAARSCHUWING: Dit script kan lang duren bij ~87M records per tabel.
--               Overweeg om in batches te draaien voor productie.
-- =============================================================================

-- Toon huidige status voor backfill
DO $$
DECLARE
    lead_null_count BIGINT;
    coin_null_count BIGINT;
    conf_null_count BIGINT;
BEGIN
    SELECT COUNT(*) INTO lead_null_count FROM kfl.mtf_signals_lead WHERE atr_at_signal IS NULL;
    SELECT COUNT(*) INTO coin_null_count FROM kfl.mtf_signals_coin WHERE atr_at_signal IS NULL;
    SELECT COUNT(*) INTO conf_null_count FROM kfl.mtf_signals_conf WHERE atr_at_signal IS NULL;
    
    RAISE NOTICE '=== ATR_AT_SIGNAL BACKFILL STATUS (VOOR) ===';
    RAISE NOTICE 'kfl.mtf_signals_lead: % records met NULL atr_at_signal', lead_null_count;
    RAISE NOTICE 'kfl.mtf_signals_coin: % records met NULL atr_at_signal', coin_null_count;
    RAISE NOTICE 'kfl.mtf_signals_conf: % records met NULL atr_at_signal', conf_null_count;
    RAISE NOTICE 'Totaal: % records', lead_null_count + coin_null_count + conf_null_count;
END $$;

-- =============================================================================
-- BACKFILL: kfl.mtf_signals_lead
-- =============================================================================
DO $$ BEGIN RAISE NOTICE 'Start backfill kfl.mtf_signals_lead...'; END $$;

UPDATE kfl.mtf_signals_lead mtf
SET atr_at_signal = iuc.atr_14
FROM kfl.indicators_unified_cache iuc
WHERE mtf.atr_at_signal IS NULL
  AND mtf.asset_id = iuc.asset_id
  AND mtf.time = iuc.time
  AND iuc.atr_14 IS NOT NULL
  AND iuc.atr_14 > 0;

-- =============================================================================
-- BACKFILL: kfl.mtf_signals_coin
-- =============================================================================
DO $$ BEGIN RAISE NOTICE 'Start backfill kfl.mtf_signals_coin...'; END $$;

UPDATE kfl.mtf_signals_coin mtf
SET atr_at_signal = iuc.atr_14
FROM kfl.indicators_unified_cache iuc
WHERE mtf.atr_at_signal IS NULL
  AND mtf.asset_id = iuc.asset_id
  AND mtf.time = iuc.time
  AND iuc.atr_14 IS NOT NULL
  AND iuc.atr_14 > 0;

-- =============================================================================
-- BACKFILL: kfl.mtf_signals_conf
-- =============================================================================
DO $$ BEGIN RAISE NOTICE 'Start backfill kfl.mtf_signals_conf...'; END $$;

UPDATE kfl.mtf_signals_conf mtf
SET atr_at_signal = iuc.atr_14
FROM kfl.indicators_unified_cache iuc
WHERE mtf.atr_at_signal IS NULL
  AND mtf.asset_id = iuc.asset_id
  AND mtf.time = iuc.time
  AND iuc.atr_14 IS NOT NULL
  AND iuc.atr_14 > 0;

-- =============================================================================
-- VERIFICATIE
-- =============================================================================
DO $$
DECLARE
    lead_null_count BIGINT;
    lead_filled_count BIGINT;
    coin_null_count BIGINT;
    coin_filled_count BIGINT;
    conf_null_count BIGINT;
    conf_filled_count BIGINT;
BEGIN
    SELECT COUNT(*) INTO lead_null_count FROM kfl.mtf_signals_lead WHERE atr_at_signal IS NULL;
    SELECT COUNT(*) INTO lead_filled_count FROM kfl.mtf_signals_lead WHERE atr_at_signal IS NOT NULL;
    SELECT COUNT(*) INTO coin_null_count FROM kfl.mtf_signals_coin WHERE atr_at_signal IS NULL;
    SELECT COUNT(*) INTO coin_filled_count FROM kfl.mtf_signals_coin WHERE atr_at_signal IS NOT NULL;
    SELECT COUNT(*) INTO conf_null_count FROM kfl.mtf_signals_conf WHERE atr_at_signal IS NULL;
    SELECT COUNT(*) INTO conf_filled_count FROM kfl.mtf_signals_conf WHERE atr_at_signal IS NOT NULL;
    
    RAISE NOTICE '=== ATR_AT_SIGNAL BACKFILL STATUS (NA) ===';
    RAISE NOTICE 'kfl.mtf_signals_lead: % NULL, % gevuld', lead_null_count, lead_filled_count;
    RAISE NOTICE 'kfl.mtf_signals_coin: % NULL, % gevuld', coin_null_count, coin_filled_count;
    RAISE NOTICE 'kfl.mtf_signals_conf: % NULL, % gevuld', conf_null_count, conf_filled_count;
    
    IF lead_null_count + coin_null_count + conf_null_count > 0 THEN
        RAISE NOTICE 'WAARSCHUWING: Sommige records hebben nog steeds NULL atr_at_signal.';
        RAISE NOTICE 'Dit kan betekenen dat er geen corresponderende indicators_unified_cache record is.';
    ELSE
        RAISE NOTICE 'SUCCESS: Alle atr_at_signal waarden zijn gevuld!';
    END IF;
END $$;

-- =============================================================================
-- BATCH VERSIE (voor zeer grote tabellen)
-- =============================================================================
-- Uncomment onderstaande code als je in batches wilt werken voor productie:
--
-- DO $$
-- DECLARE
--     batch_size INT := 100000;
--     updated_count INT := 1;
--     total_updated BIGINT := 0;
-- BEGIN
--     WHILE updated_count > 0 LOOP
--         UPDATE kfl.mtf_signals_lead mtf
--         SET atr_at_signal = iuc.atr_14
--         FROM kfl.indicators_unified_cache iuc
--         WHERE mtf.atr_at_signal IS NULL
--           AND mtf.asset_id = iuc.asset_id
--           AND mtf.time = iuc.time
--           AND iuc.atr_14 IS NOT NULL
--           AND iuc.atr_14 > 0
--           AND mtf.ctid IN (
--               SELECT mtf2.ctid FROM kfl.mtf_signals_lead mtf2
--               WHERE mtf2.atr_at_signal IS NULL
--               LIMIT batch_size
--           );
--         
--         GET DIAGNOSTICS updated_count = ROW_COUNT;
--         total_updated := total_updated + updated_count;
--         RAISE NOTICE 'Batch update: % records (totaal: %)', updated_count, total_updated;
--         COMMIT;
--     END LOOP;
-- END $$;
