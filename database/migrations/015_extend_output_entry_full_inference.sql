-- Migration: Extend qbn.output_entry for full inference output
-- Date: 2026-01-09
-- Purpose: Consolidate all inference output into output_entry, replacing bayesian_predictions
--
-- This migration adds columns for:
-- - Full probability distributions (JSONB)
-- - Per-horizon confidence values
-- - Expected ATR moves
-- - Entry timing distribution
-- - Inference metadata (timing, version)
-- - Position confidence

-- Add distribution columns (full 7-state probability distributions)
ALTER TABLE qbn.output_entry
ADD COLUMN IF NOT EXISTS distribution_1h JSONB,
ADD COLUMN IF NOT EXISTS distribution_4h JSONB,
ADD COLUMN IF NOT EXISTS distribution_1d JSONB;

-- Add per-horizon confidence (max probability from distribution)
ALTER TABLE qbn.output_entry
ADD COLUMN IF NOT EXISTS confidence_1h REAL,
ADD COLUMN IF NOT EXISTS confidence_4h REAL,
ADD COLUMN IF NOT EXISTS confidence_1d REAL;

-- Add expected ATR moves per horizon
ALTER TABLE qbn.output_entry
ADD COLUMN IF NOT EXISTS expected_atr_1h REAL,
ADD COLUMN IF NOT EXISTS expected_atr_4h REAL,
ADD COLUMN IF NOT EXISTS expected_atr_1d REAL;

-- Add entry timing distribution
ALTER TABLE qbn.output_entry
ADD COLUMN IF NOT EXISTS entry_timing_distribution JSONB;

-- Add inference metadata
ALTER TABLE qbn.output_entry
ADD COLUMN IF NOT EXISTS inference_time_ms REAL,
ADD COLUMN IF NOT EXISTS model_version VARCHAR(10) DEFAULT '3.2';

SELECT remove_retention_policy('qbn.output_entry', if_exists => TRUE);

-- Add index for model version queries (monitoring)
CREATE INDEX IF NOT EXISTS idx_output_entry_model_version
ON qbn.output_entry (model_version, time DESC);

-- Add index for inference time monitoring
CREATE INDEX IF NOT EXISTS idx_output_entry_inference_time
ON qbn.output_entry (time DESC)
WHERE inference_time_ms > 50;  -- Slow query detection

-- Update table comment
COMMENT ON TABLE qbn.output_entry IS 'Complete QBN inference output. Replaces bayesian_predictions. Contains all predictions, distributions, and metadata. TSEM receives entry signals via pg_notify.';

-- Column comments
COMMENT ON COLUMN qbn.output_entry.distribution_1h IS 'Full 7-state probability distribution for 1h horizon (JSONB)';
COMMENT ON COLUMN qbn.output_entry.distribution_4h IS 'Full 7-state probability distribution for 4h horizon (JSONB)';
COMMENT ON COLUMN qbn.output_entry.distribution_1d IS 'Full 7-state probability distribution for 1d horizon (JSONB)';
COMMENT ON COLUMN qbn.output_entry.confidence_1h IS 'Max probability from distribution_1h (0.0-1.0)';
COMMENT ON COLUMN qbn.output_entry.confidence_4h IS 'Max probability from distribution_4h (0.0-1.0)';
COMMENT ON COLUMN qbn.output_entry.confidence_1d IS 'Max probability from distribution_1d (0.0-1.0)';
COMMENT ON COLUMN qbn.output_entry.expected_atr_1h IS 'Expected ATR move for 1h horizon';
COMMENT ON COLUMN qbn.output_entry.expected_atr_4h IS 'Expected ATR move for 4h horizon';
COMMENT ON COLUMN qbn.output_entry.expected_atr_1d IS 'Expected ATR move for 1d horizon';
COMMENT ON COLUMN qbn.output_entry.entry_timing_distribution IS 'Entry timing window distribution (JSONB)';
COMMENT ON COLUMN qbn.output_entry.inference_time_ms IS 'Inference latency in milliseconds';
COMMENT ON COLUMN qbn.output_entry.model_version IS 'QBN model version used for inference';

