-- f:/Containers/QBN_v3/database/migrations/rollback_027_create_barrier_outcomes.sql
-- =============================================================================
-- ROLLBACK: First-Touch Barrier Tables
-- =============================================================================

BEGIN;

DROP TABLE IF EXISTS qbn.barrier_outcomes CASCADE;
DROP TABLE IF EXISTS qbn.barrier_config CASCADE;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'qbn' AND table_name = 'barrier_outcomes') THEN
        RAISE EXCEPTION 'Rollback gefaald: barrier_outcomes bestaat nog';
    END IF;
    RAISE NOTICE 'Rollback succesvol';
END $$;

COMMIT;
