-- =============================================================================
-- 002_add_outcome_constraints.sql
-- =============================================================================
-- Voeg CHECK constraints toe aan outcome kolommen in alle drie MTF tabellen.
-- Dit zorgt ervoor dat outcome waarden alleen -3, -2, -1, 0, +1, +2, +3 kunnen zijn.
--
-- USAGE:
--   psql -h 10.10.10.1 -U postgres -d KFLhyper -f 002_add_outcome_constraints.sql
--
-- REASON: Data integriteit - voorkom ongeldige outcome waarden bij backfill.
-- =============================================================================

-- =============================================================================
-- CONSTRAINTS: kfl.mtf_signals_lead
-- =============================================================================
DO $$ BEGIN RAISE NOTICE 'Adding constraints to kfl.mtf_signals_lead...'; END $$;

ALTER TABLE kfl.mtf_signals_lead
    DROP CONSTRAINT IF EXISTS chk_lead_outcome_1h_range,
    DROP CONSTRAINT IF EXISTS chk_lead_outcome_4h_range,
    DROP CONSTRAINT IF EXISTS chk_lead_outcome_1d_range;

ALTER TABLE kfl.mtf_signals_lead
    ADD CONSTRAINT chk_lead_outcome_1h_range 
        CHECK (outcome_1h IS NULL OR (outcome_1h >= -3 AND outcome_1h <= 3)),
    ADD CONSTRAINT chk_lead_outcome_4h_range 
        CHECK (outcome_4h IS NULL OR (outcome_4h >= -3 AND outcome_4h <= 3)),
    ADD CONSTRAINT chk_lead_outcome_1d_range 
        CHECK (outcome_1d IS NULL OR (outcome_1d >= -3 AND outcome_1d <= 3));

-- =============================================================================
-- CONSTRAINTS: kfl.mtf_signals_coin
-- =============================================================================
DO $$ BEGIN RAISE NOTICE 'Adding constraints to kfl.mtf_signals_coin...'; END $$;

ALTER TABLE kfl.mtf_signals_coin
    DROP CONSTRAINT IF EXISTS chk_coin_outcome_1h_range,
    DROP CONSTRAINT IF EXISTS chk_coin_outcome_4h_range,
    DROP CONSTRAINT IF EXISTS chk_coin_outcome_1d_range;

ALTER TABLE kfl.mtf_signals_coin
    ADD CONSTRAINT chk_coin_outcome_1h_range 
        CHECK (outcome_1h IS NULL OR (outcome_1h >= -3 AND outcome_1h <= 3)),
    ADD CONSTRAINT chk_coin_outcome_4h_range 
        CHECK (outcome_4h IS NULL OR (outcome_4h >= -3 AND outcome_4h <= 3)),
    ADD CONSTRAINT chk_coin_outcome_1d_range 
        CHECK (outcome_1d IS NULL OR (outcome_1d >= -3 AND outcome_1d <= 3));

-- =============================================================================
-- CONSTRAINTS: kfl.mtf_signals_conf
-- =============================================================================
DO $$ BEGIN RAISE NOTICE 'Adding constraints to kfl.mtf_signals_conf...'; END $$;

ALTER TABLE kfl.mtf_signals_conf
    DROP CONSTRAINT IF EXISTS chk_conf_outcome_1h_range,
    DROP CONSTRAINT IF EXISTS chk_conf_outcome_4h_range,
    DROP CONSTRAINT IF EXISTS chk_conf_outcome_1d_range;

ALTER TABLE kfl.mtf_signals_conf
    ADD CONSTRAINT chk_conf_outcome_1h_range 
        CHECK (outcome_1h IS NULL OR (outcome_1h >= -3 AND outcome_1h <= 3)),
    ADD CONSTRAINT chk_conf_outcome_4h_range 
        CHECK (outcome_4h IS NULL OR (outcome_4h >= -3 AND outcome_4h <= 3)),
    ADD CONSTRAINT chk_conf_outcome_1d_range 
        CHECK (outcome_1d IS NULL OR (outcome_1d >= -3 AND outcome_1d <= 3));

-- =============================================================================
-- VERIFICATIE
-- =============================================================================
DO $$
DECLARE
    constraint_count INT;
BEGIN
    SELECT COUNT(*) INTO constraint_count
    FROM information_schema.table_constraints
    WHERE constraint_type = 'CHECK'
      AND constraint_name LIKE 'chk_%_outcome_%_range'
      AND table_schema = 'kfl';
    
    RAISE NOTICE '=== CONSTRAINT VERIFICATIE ===';
    RAISE NOTICE 'Aantal outcome CHECK constraints: % (verwacht: 9)', constraint_count;
    
    IF constraint_count = 9 THEN
        RAISE NOTICE 'SUCCESS: Alle constraints zijn toegevoegd!';
    ELSE
        RAISE WARNING 'WAARSCHUWING: Niet alle constraints zijn toegevoegd.';
    END IF;
END $$;

-- Toon alle constraints
SELECT 
    table_name,
    constraint_name,
    check_clause
FROM information_schema.check_constraints cc
JOIN information_schema.table_constraints tc 
    ON cc.constraint_name = tc.constraint_name
WHERE tc.table_schema = 'kfl'
  AND tc.constraint_name LIKE 'chk_%_outcome_%_range'
ORDER BY table_name, constraint_name;
