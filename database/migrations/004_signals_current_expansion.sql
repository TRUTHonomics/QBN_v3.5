-- Migration: 004_signals_current_expansion.sql
-- Purpose: Expand kfl.signals_current with additional technical indicators
-- Date: 2025-12-10
-- Priority: MEDIUM
-- Description: Add ADX, CMF, OBV, Stochastic, MACD variants, Ichimoku components, and ATR

-- ============================================================================
-- PART 1: Add ADX (Average Directional Index) signals
-- ============================================================================

BEGIN;

-- ADX strength and direction signals
ALTER TABLE kfl.signals_current
ADD COLUMN IF NOT EXISTS adx_signal SMALLINT,           -- ADX strength: -1 (weak), 0 (neutral), +1 (strong trend)
ADD COLUMN IF NOT EXISTS adx_plus_di REAL,               -- Plus Directional Indicator
ADD COLUMN IF NOT EXISTS adx_minus_di REAL,              -- Minus Directional Indicator
ADD COLUMN IF NOT EXISTS adx_value REAL;                 -- ADX value (0-100)

COMMENT ON COLUMN kfl.signals_current.adx_signal IS 'ADX trend strength: -1 (weak/ranging), 0 (neutral), +1 (strong trend)';
COMMENT ON COLUMN kfl.signals_current.adx_plus_di IS '+DI value for bullish pressure measurement';
COMMENT ON COLUMN kfl.signals_current.adx_minus_di IS '-DI value for bearish pressure measurement';
COMMENT ON COLUMN kfl.signals_current.adx_value IS 'ADX value (0-100): <20 weak, 20-40 moderate, >40 strong trend';

COMMIT;

-- ============================================================================
-- PART 2: Add CMF (Chaikin Money Flow) signals
-- ============================================================================

BEGIN;

ALTER TABLE kfl.signals_current
ADD COLUMN IF NOT EXISTS cmf_signal SMALLINT,           -- CMF: -1 (selling pressure), 0 (neutral), +1 (buying pressure)
ADD COLUMN IF NOT EXISTS cmf_value REAL;                 -- CMF value (-1 to +1)

COMMENT ON COLUMN kfl.signals_current.cmf_signal IS 'Chaikin Money Flow: -1 (distribution), 0 (neutral), +1 (accumulation)';
COMMENT ON COLUMN kfl.signals_current.cmf_value IS 'CMF value (-1 to +1): measures money flow volume';

COMMIT;

-- ============================================================================
-- PART 3: Add OBV (On-Balance Volume) signals
-- ============================================================================

BEGIN;

ALTER TABLE kfl.signals_current
ADD COLUMN IF NOT EXISTS obv_signal SMALLINT,           -- OBV trend: -1 (down), 0 (flat), +1 (up)
ADD COLUMN IF NOT EXISTS obv_value REAL,                 -- OBV cumulative value
ADD COLUMN IF NOT EXISTS obv_ma REAL;                    -- OBV moving average for trend detection

COMMENT ON COLUMN kfl.signals_current.obv_signal IS 'On-Balance Volume trend: -1 (declining), 0 (flat), +1 (rising)';
COMMENT ON COLUMN kfl.signals_current.obv_value IS 'OBV cumulative value: running total of volume direction';
COMMENT ON COLUMN kfl.signals_current.obv_ma IS 'OBV moving average for trend comparison';

COMMIT;

-- ============================================================================
-- PART 4: Add Stochastic Oscillator signals
-- ============================================================================

BEGIN;

ALTER TABLE kfl.signals_current
ADD COLUMN IF NOT EXISTS stoch_k_signal SMALLINT,       -- Stoch %K: -1 (oversold), 0 (neutral), +1 (overbought)
ADD COLUMN IF NOT EXISTS stoch_d_signal SMALLINT,       -- Stoch %D: -1 (oversold), 0 (neutral), +1 (overbought)
ADD COLUMN IF NOT EXISTS stoch_k_value REAL,             -- %K value (0-100)
ADD COLUMN IF NOT EXISTS stoch_d_value REAL;             -- %D value (0-100)

COMMENT ON COLUMN kfl.signals_current.stoch_k_signal IS 'Stochastic %K signal: -1 (<20 oversold), 0 (20-80), +1 (>80 overbought)';
COMMENT ON COLUMN kfl.signals_current.stoch_d_signal IS 'Stochastic %D signal: -1 (<20 oversold), 0 (20-80), +1 (>80 overbought)';
COMMENT ON COLUMN kfl.signals_current.stoch_k_value IS 'Stochastic %K value (0-100): fast oscillator';
COMMENT ON COLUMN kfl.signals_current.stoch_d_value IS 'Stochastic %D value (0-100): slow oscillator (smoothed %K)';

COMMIT;

-- ============================================================================
-- PART 5: Add MACD variants (signal line, histogram)
-- ============================================================================

BEGIN;

ALTER TABLE kfl.signals_current
ADD COLUMN IF NOT EXISTS macd_signal_line REAL,         -- MACD signal line (9-period EMA of MACD)
ADD COLUMN IF NOT EXISTS macd_histogram REAL,            -- MACD - Signal Line
ADD COLUMN IF NOT EXISTS macd_histogram_signal SMALLINT; -- Histogram trend: -1 (bearish), 0 (neutral), +1 (bullish)

COMMENT ON COLUMN kfl.signals_current.macd_signal_line IS 'MACD signal line: 9-period EMA of MACD line';
COMMENT ON COLUMN kfl.signals_current.macd_histogram IS 'MACD histogram: difference between MACD and signal line';
COMMENT ON COLUMN kfl.signals_current.macd_histogram_signal IS 'Histogram trend: -1 (declining), 0 (flat), +1 (rising)';

COMMIT;

-- ============================================================================
-- PART 6: Add Ichimoku Cloud components
-- ============================================================================

BEGIN;

ALTER TABLE kfl.signals_current
ADD COLUMN IF NOT EXISTS ichimoku_conversion_line REAL,  -- Tenkan-sen (9-period)
ADD COLUMN IF NOT EXISTS ichimoku_base_line REAL,        -- Kijun-sen (26-period)
ADD COLUMN IF NOT EXISTS ichimoku_span_a REAL,           -- Senkou Span A (leading)
ADD COLUMN IF NOT EXISTS ichimoku_span_b REAL,           -- Senkou Span B (lagging)
ADD COLUMN IF NOT EXISTS ichimoku_signal SMALLINT;       -- Cloud position: -1 (below), 0 (in cloud), +1 (above)

COMMENT ON COLUMN kfl.signals_current.ichimoku_conversion_line IS 'Tenkan-sen: (9-period high + 9-period low) / 2';
COMMENT ON COLUMN kfl.signals_current.ichimoku_base_line IS 'Kijun-sen: (26-period high + 26-period low) / 2';
COMMENT ON COLUMN kfl.signals_current.ichimoku_span_a IS 'Senkou Span A: (Conversion + Base) / 2, plotted 26 periods ahead';
COMMENT ON COLUMN kfl.signals_current.ichimoku_span_b IS 'Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead';
COMMENT ON COLUMN kfl.signals_current.ichimoku_signal IS 'Price vs Cloud: -1 (below cloud, bearish), 0 (in cloud), +1 (above cloud, bullish)';

COMMIT;

-- ============================================================================
-- PART 7: Add ATR (Average True Range) value
-- ============================================================================

BEGIN;

ALTER TABLE kfl.signals_current
ADD COLUMN IF NOT EXISTS atr_value REAL,                 -- ATR value (for volatility normalization)
ADD COLUMN IF NOT EXISTS atr_percent REAL;               -- ATR as percentage of price

COMMENT ON COLUMN kfl.signals_current.atr_value IS 'Average True Range: 14-period ATR for volatility measurement';
COMMENT ON COLUMN kfl.signals_current.atr_percent IS 'ATR as percentage of current price (ATR/price * 100)';

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES (for immediate post-migration check)
-- ============================================================================

-- Verify all new columns exist
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'kfl'
AND table_name = 'signals_current'
AND column_name IN (
    'adx_signal', 'adx_plus_di', 'adx_minus_di', 'adx_value',
    'cmf_signal', 'cmf_value',
    'obv_signal', 'obv_value', 'obv_ma',
    'stoch_k_signal', 'stoch_d_signal', 'stoch_k_value', 'stoch_d_value',
    'macd_signal_line', 'macd_histogram', 'macd_histogram_signal',
    'ichimoku_conversion_line', 'ichimoku_base_line', 'ichimoku_span_a', 'ichimoku_span_b', 'ichimoku_signal',
    'atr_value', 'atr_percent'
)
ORDER BY column_name;

-- Count new columns
SELECT COUNT(*) as new_columns_count
FROM information_schema.columns
WHERE table_schema = 'kfl'
AND table_name = 'signals_current'
AND column_name IN (
    'adx_signal', 'adx_plus_di', 'adx_minus_di', 'adx_value',
    'cmf_signal', 'cmf_value',
    'obv_signal', 'obv_value', 'obv_ma',
    'stoch_k_signal', 'stoch_d_signal', 'stoch_k_value', 'stoch_d_value',
    'macd_signal_line', 'macd_histogram', 'macd_histogram_signal',
    'ichimoku_conversion_line', 'ichimoku_base_line', 'ichimoku_span_a', 'ichimoku_span_b', 'ichimoku_signal',
    'atr_value', 'atr_percent'
);

-- Expected: 23 new columns

-- Verify all values are NULL initially
SELECT
    COUNT(*) as total_rows,
    COUNT(adx_signal) as adx_count,
    COUNT(cmf_signal) as cmf_count,
    COUNT(obv_signal) as obv_count,
    COUNT(stoch_k_signal) as stoch_count,
    COUNT(ichimoku_signal) as ichimoku_count,
    COUNT(atr_value) as atr_count
FROM kfl.signals_current;

-- Expected: all counts should be 0 initially

-- ============================================================================
-- NOTES
-- ============================================================================

-- New Indicators Summary:
-- 1. ADX (4 columns): Trend strength and directional indicators
-- 2. CMF (2 columns): Money flow and volume analysis
-- 3. OBV (3 columns): Volume-based trend confirmation
-- 4. Stochastic (4 columns): Momentum oscillator for overbought/oversold
-- 5. MACD variants (3 columns): Enhanced MACD analysis
-- 6. Ichimoku (5 columns): Cloud-based trend and support/resistance
-- 7. ATR (2 columns): Volatility measurement for normalization

-- Total: 23 new columns

-- Timing:
-- - Estimated execution time: 5-10 minutes
-- - ALTER operations on kfl.signals_current table
-- - Execute during off-hours to minimize lock contention

-- Integration:
-- - These signals will be calculated and populated by the signal generator
-- - Will be propagated to MTF cache via existing triggers
-- - Available for BN model as additional evidence nodes

-- Next Steps:
-- 1. Run 004_mtf_cache_expansion.sql to add corresponding columns to MTF cache
-- 2. Update signal generator to calculate these new indicators
-- 3. Update MTF triggers to propagate new signals to cache
