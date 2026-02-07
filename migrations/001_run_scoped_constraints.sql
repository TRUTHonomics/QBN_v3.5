-- Migration: Run-Scoped Data Isolatie
-- Doel: Voeg run_id toe aan primary/unique constraints voor signal_weights en position_delta_threshold_config
-- Datum: 2026-02-06

-- 1a. signal_weights: run_id toevoegen aan PK
ALTER TABLE qbn.signal_weights DROP CONSTRAINT signal_weights_pkey;
ALTER TABLE qbn.signal_weights ADD PRIMARY KEY (asset_id, signal_name, horizon, layer, run_id);

-- 1b. position_delta_threshold_config: run_id toevoegen aan unique constraint
ALTER TABLE qbn.position_delta_threshold_config 
  DROP CONSTRAINT position_delta_threshold_conf_asset_id_delta_type_score_typ_key;
ALTER TABLE qbn.position_delta_threshold_config 
  ADD CONSTRAINT position_delta_threshold_config_asset_delta_score_run_key 
  UNIQUE (asset_id, delta_type, score_type, run_id);

-- 1c. combination_alpha: verwijder verouderde analyzed_at constraint
DROP INDEX IF EXISTS qbn.idx_combination_alpha_unique;

-- Verificatie queries:
-- SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='qbn' AND tablename='signal_weights';
-- SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='qbn' AND tablename='position_delta_threshold_config';
-- SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='qbn' AND tablename='combination_alpha';
