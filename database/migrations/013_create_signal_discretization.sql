-- f:/Containers/QBN_v2/database/migrations/013_create_signal_discretization.sql
-- REASON: Centrale opslag voor indicator drempelwaarden om hardcoded waarden te elimineren.

BEGIN;

CREATE TABLE IF NOT EXISTS qbn.signal_discretization (
    indicator_base      TEXT NOT NULL,
    threshold_name      TEXT NOT NULL,
    threshold_value     DOUBLE PRECISION NOT NULL,
    description         TEXT,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (indicator_base, threshold_name)
);

COMMENT ON TABLE qbn.signal_discretization IS 'Centrale drempelwaarden voor indicator discretisatie (-2 tot +2)';

-- Seed met huidige (voorheen hardcoded) waarden voor RSI
INSERT INTO qbn.signal_discretization (indicator_base, threshold_name, threshold_value, description)
VALUES 
    ('RSI', 'extreme_oversold', 20.0, 'RSI < 20 -> +2'),
    ('RSI', 'oversold', 30.0, 'RSI < 30 -> +1'),
    ('RSI', 'overbought', 70.0, 'RSI > 70 -> -1'),
    ('RSI', 'extreme_overbought', 80.0, 'RSI > 80 -> -2')
ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
    threshold_value = EXCLUDED.threshold_value,
    updated_at = NOW();

-- Seed MACD thresholds
INSERT INTO qbn.signal_discretization (indicator_base, threshold_name, threshold_value, description)
VALUES 
    ('MACD', 'histogram_strong_threshold', 0.001, 'Minimale histogram afstand voor strong signal'),
    ('MACD', 'signal_ratio_strong', 1.1, 'MACD / Signal ratio voor strong signal')
ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
    threshold_value = EXCLUDED.threshold_value,
    updated_at = NOW();

-- Seed Bollinger thresholds
INSERT INTO qbn.signal_discretization (indicator_base, threshold_name, threshold_value, description)
VALUES 
    ('BOLLINGER', 'lower_zone', 0.25, 'Positie < 0.25 (bullish zone)'),
    ('BOLLINGER', 'upper_zone', 0.75, 'Positie > 0.75 (bearish zone)')
ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
    threshold_value = EXCLUDED.threshold_value,
    updated_at = NOW();

-- Seed Keltner thresholds
INSERT INTO qbn.signal_discretization (indicator_base, threshold_name, threshold_value, description)
VALUES 
    ('KELTNER', 'lower_zone', 0.25, 'Positie < 0.25 (bullish zone)'),
    ('KELTNER', 'upper_zone', 0.75, 'Positie > 0.75 (bearish zone)')
ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
    threshold_value = EXCLUDED.threshold_value,
    updated_at = NOW();

-- Seed ATR thresholds
INSERT INTO qbn.signal_discretization (indicator_base, threshold_name, threshold_value, description)
VALUES 
    ('ATR', 'extreme_expansion', 2.0, 'Ratio > 2.0 (extreme expansion)'),
    ('ATR', 'expansion', 1.5, 'Ratio > 1.5 (expansion)'),
    ('ATR', 'contraction', 0.75, 'Ratio < 0.75 (contraction)'),
    ('ATR', 'extreme_squeeze', 0.5, 'Ratio < 0.5 (extreme squeeze)')
ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
    threshold_value = EXCLUDED.threshold_value,
    updated_at = NOW();

-- Seed ADX thresholds
INSERT INTO qbn.signal_discretization (indicator_base, threshold_name, threshold_value, description)
VALUES 
    ('ADX', 'non_trending', 25.0, 'ADX < 25 (ranging regime)'),
    ('ADX', 'strong_trend', 40.0, 'ADX > 40 (strong trend)')
ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
    threshold_value = EXCLUDED.threshold_value,
    updated_at = NOW();

-- Seed Stochastic thresholds
INSERT INTO qbn.signal_discretization (indicator_base, threshold_name, threshold_value, description)
VALUES 
    ('STOCHASTIC', 'extreme_oversold', 10.0, 'K < 10 -> +2'),
    ('STOCHASTIC', 'oversold', 20.0, 'K < 20 -> +1'),
    ('STOCHASTIC', 'overbought', 80.0, 'K > 80 -> -1'),
    ('STOCHASTIC', 'extreme_overbought', 90.0, 'K > 90 -> -2')
ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
    threshold_value = EXCLUDED.threshold_value,
    updated_at = NOW();

-- Seed CMF thresholds
INSERT INTO qbn.signal_discretization (indicator_base, threshold_name, threshold_value, description)
VALUES 
    ('CMF', 'strong_buying', 0.15, 'CMF > 0.15 -> +2'),
    ('CMF', 'bullish_bias', 0.05, 'CMF > 0.05 -> +1'),
    ('CMF', 'bearish_bias', -0.05, 'CMF < -0.05 -> -1'),
    ('CMF', 'strong_selling', -0.15, 'CMF < -0.15 -> -2')
ON CONFLICT (indicator_base, threshold_name) DO UPDATE SET
    threshold_value = EXCLUDED.threshold_value,
    updated_at = NOW();

-- REASON: Pipeline user permissies verlenen voor sync en loading
GRANT SELECT, INSERT, UPDATE, DELETE ON qbn.signal_discretization TO pipeline;

COMMIT;

