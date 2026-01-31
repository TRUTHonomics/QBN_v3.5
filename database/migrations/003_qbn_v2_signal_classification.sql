-- Migration: 003_qbn_v2_signal_classification.sql
-- Purpose: Create signal classification table for semantic grouping
-- Date: 2025-12-10
-- Priority: HIGH
-- Description: New table to classify signals into LEADING, COINCIDENT, and CONFIRMING categories

-- ============================================================================
-- PART 1: Create signal_classification table
-- ============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS qbn.signal_classification (
    signal_name VARCHAR(100) PRIMARY KEY,
    semantic_class VARCHAR(20) NOT NULL,
    indicator_base VARCHAR(50),          -- e.g., 'RSI', 'MACD', 'BB', 'ADX'
    indicator_variant VARCHAR(20),        -- e.g., 'overbought', 'crossover', 'squeeze'
    polarity VARCHAR(10),                 -- 'bullish', 'bearish', 'neutral'
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraint: semantic_class must be one of three categories
    CONSTRAINT chk_semantic_class CHECK (
        semantic_class IN ('LEADING', 'COINCIDENT', 'CONFIRMING')
    ),

    -- Constraint: polarity must be one of three values
    CONSTRAINT chk_polarity CHECK (
        polarity IS NULL OR polarity IN ('bullish', 'bearish', 'neutral')
    )
);

-- Add comment to table
COMMENT ON TABLE qbn.signal_classification IS 'Classification of trading signals into semantic categories for Bayesian Network structure';

-- Add comments to columns
COMMENT ON COLUMN qbn.signal_classification.signal_name IS 'Unique signal identifier (e.g., structural_rsi_signal, tactical_bb_signal)';
COMMENT ON COLUMN qbn.signal_classification.semantic_class IS 'Signal temporal category: LEADING (predictive), COINCIDENT (confirmatory), CONFIRMING (validation)';
COMMENT ON COLUMN qbn.signal_classification.indicator_base IS 'Base technical indicator (RSI, MACD, BB, ADX, etc.)';
COMMENT ON COLUMN qbn.signal_classification.indicator_variant IS 'Specific variant or condition (overbought, crossover, squeeze, etc.)';
COMMENT ON COLUMN qbn.signal_classification.polarity IS 'Directional bias: bullish, bearish, or neutral';
COMMENT ON COLUMN qbn.signal_classification.description IS 'Human-readable description of signal meaning and interpretation';

COMMIT;

-- ============================================================================
-- PART 2: Create indexes for efficient queries
-- ============================================================================

BEGIN;

-- Index on semantic_class for filtering by category
CREATE INDEX IF NOT EXISTS idx_signal_class_semantic
ON qbn.signal_classification (semantic_class);

-- Index on indicator_base for grouping by indicator type
CREATE INDEX IF NOT EXISTS idx_signal_class_indicator
ON qbn.signal_classification (indicator_base);

COMMIT;

-- ============================================================================
-- PART 3: Insert example signal classifications (TEMPLATE)
-- ============================================================================
-- Note: Actual signal classification data will be populated in Phase 1.2
-- This section provides examples for reference

/*
BEGIN;

-- LEADING signals (predictive, ~35 expected)
INSERT INTO qbn.signal_classification (signal_name, semantic_class, indicator_base, indicator_variant, polarity, description) VALUES
('structural_rsi_signal', 'LEADING', 'RSI', 'overbought_oversold', 'neutral', 'Daily RSI indicating overbought/oversold conditions'),
('structural_macd_signal', 'LEADING', 'MACD', 'crossover', 'neutral', 'Daily MACD histogram showing momentum shifts'),
('structural_bb_signal', 'LEADING', 'BB', 'squeeze_breakout', 'neutral', 'Daily Bollinger Bands indicating volatility regime'),
-- ... ~32 more LEADING signals

-- COINCIDENT signals (confirmatory, ~30 expected)
('tactical_rsi_signal', 'COINCIDENT', 'RSI', 'overbought_oversold', 'neutral', '4H RSI confirming trend continuation'),
('tactical_macd_signal', 'COINCIDENT', 'MACD', 'crossover', 'neutral', '4H MACD confirming momentum'),
-- ... ~28 more COINCIDENT signals

-- CONFIRMING signals (validation, ~30 expected)
('entry_rsi_signal', 'CONFIRMING', 'RSI', 'overbought_oversold', 'neutral', '1H RSI validating entry timing'),
('entry_macd_signal', 'CONFIRMING', 'MACD', 'crossover', 'neutral', '1H MACD validating entry signal'),
-- ... ~28 more CONFIRMING signals

ON CONFLICT (signal_name) DO NOTHING;

COMMIT;
*/

-- ============================================================================
-- VERIFICATION QUERIES (for immediate post-migration check)
-- ============================================================================

-- Verify table exists
SELECT EXISTS (
    SELECT FROM information_schema.tables
    WHERE table_schema = 'qbn'
    AND table_name = 'signal_classification'
) AS table_exists;

-- Verify table structure (columns)
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'qbn'
AND table_name = 'signal_classification'
ORDER BY ordinal_position;

-- Verify constraints
SELECT conname, contype, pg_get_constraintdef(oid) as definition
FROM pg_constraint
WHERE conrelid = 'qbn.signal_classification'::regclass
ORDER BY conname;

-- Verify indexes
SELECT indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'qbn'
AND tablename = 'signal_classification'
ORDER BY indexname;

-- Count rows (should be 0 immediately after migration)
SELECT COUNT(*) as row_count FROM qbn.signal_classification;

-- ============================================================================
-- NOTES
-- ============================================================================

-- Signal Classification Categories:
--
-- LEADING (~35 signals):
--   - Structural layer (Daily timeframe): ~12-15 signals
--   - Predictive indicators: RSI, MACD, BB, ADX, Stochastic
--   - Purpose: Early warning of potential trend changes
--
-- COINCIDENT (~30 signals):
--   - Tactical layer (4H timeframe): ~12-15 signals
--   - Confirmatory indicators: Same as LEADING but shorter timeframe
--   - Purpose: Confirm trend established by LEADING signals
--
-- CONFIRMING (~30 signals):
--   - Entry layer (1H timeframe): ~12-15 signals
--   - UTF layer (1m timeframe): ~3-5 signals
--   - Validation indicators: Same indicators, entry timing
--   - Purpose: Validate entry/exit timing

-- Timing:
-- - Estimated execution time: ~1 minute
-- - No dependencies on other migrations
-- - Fast operation (simple CREATE TABLE)

-- Next Steps:
-- 1. Phase 1.2: Populate table with ~95 signal classifications
-- 2. Domain expert review required for classification accuracy
-- 3. Validate distribution: ~35 LEADING / ~30 COINCIDENT / ~30 CONFIRMING
