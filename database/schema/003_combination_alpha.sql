-- =============================================================================
-- Combination Alpha Analysis Schema
-- =============================================================================
-- Slaat resultaten op van de Combination Alpha Analysis (Phase 2.5).
-- Bevat Odds Ratios, CI's, Sens/Spec en classificaties per signaalcombinatie.
--
-- ARCHITECTUUR NOOT:
-- - Tabel bevindt zich in schema qbn (niet kfl)
-- - Index op (asset_id, horizon, classification) voor snelle lookups
-- - Partitioning niet nodig (typisch <10K rows per asset)
--
-- Gebruik:
--   Vanuit Python via CombinationAlphaAnalyzer.save_to_database()
--   Of handmatig voor queries op Golden Rules
-- =============================================================================

-- Ensure qbn schema exists
CREATE SCHEMA IF NOT EXISTS qbn;

-- Main table for combination alpha results
CREATE TABLE IF NOT EXISTS qbn.combination_alpha (
    -- Primary key
    id SERIAL PRIMARY KEY,
    
    -- Identification
    asset_id INTEGER NOT NULL,
    horizon VARCHAR(5) NOT NULL,  -- '1h', '4h', '1d'
    combination_key VARCHAR(100) NOT NULL,  -- 'bullish|neutral|bearish' etc
    target_type VARCHAR(20) NOT NULL,  -- 'bullish', 'bearish', 'significant'
    
    -- Sample info
    n_with_combination INTEGER NOT NULL,
    n_total INTEGER NOT NULL,
    
    -- Odds Ratio
    odds_ratio NUMERIC(10, 4) NOT NULL,
    or_ci_lower NUMERIC(10, 4) NOT NULL,
    or_ci_upper NUMERIC(10, 4) NOT NULL,
    
    -- Bootstrap CI (nullable - may not always be computed)
    bootstrap_ci_lower NUMERIC(10, 4),
    bootstrap_ci_upper NUMERIC(10, 4),
    
    -- Sensitivity / Specificity
    sensitivity NUMERIC(6, 4) NOT NULL,
    specificity NUMERIC(6, 4) NOT NULL,
    ppv NUMERIC(6, 4) NOT NULL,  -- Positive Predictive Value
    npv NUMERIC(6, 4) NOT NULL,  -- Negative Predictive Value
    lr_positive NUMERIC(10, 4),  -- Likelihood Ratio+ (can be NULL if infinity)
    lr_negative NUMERIC(10, 4),  -- Likelihood Ratio- (can be NULL if infinity)
    
    -- Effect sizes
    mcc NUMERIC(6, 4) NOT NULL,  -- Matthews Correlation Coefficient
    cramers_v NUMERIC(6, 4) NOT NULL,
    information_gain NUMERIC(10, 6) NOT NULL,
    
    -- Chi-square / Fisher
    chi_statistic NUMERIC(12, 4) NOT NULL,
    chi_p_value NUMERIC(12, 10) NOT NULL,
    test_type VARCHAR(20) NOT NULL,  -- 'chi_square' or 'fisher_exact'
    
    -- Corrected p-value
    p_value_corrected NUMERIC(12, 10),
    
    -- Classification
    classification VARCHAR(20) NOT NULL,  -- 'golden_rule', 'promising', 'noise'
    
    -- Metadata
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT combination_alpha_horizon_check 
        CHECK (horizon IN ('1h', '4h', '1d')),
    CONSTRAINT combination_alpha_target_check 
        CHECK (target_type IN ('bullish', 'bearish', 'significant')),
    CONSTRAINT combination_alpha_classification_check 
        CHECK (classification IN ('golden_rule', 'promising', 'noise')),
    CONSTRAINT combination_alpha_or_positive 
        CHECK (odds_ratio > 0),
    CONSTRAINT combination_alpha_sens_spec_range 
        CHECK (sensitivity >= 0 AND sensitivity <= 1 AND specificity >= 0 AND specificity <= 1)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_combination_alpha_asset_horizon 
    ON qbn.combination_alpha(asset_id, horizon);

CREATE INDEX IF NOT EXISTS idx_combination_alpha_classification 
    ON qbn.combination_alpha(classification);

CREATE INDEX IF NOT EXISTS idx_combination_alpha_golden_rules 
    ON qbn.combination_alpha(asset_id, horizon, classification) 
    WHERE classification = 'golden_rule';

CREATE INDEX IF NOT EXISTS idx_combination_alpha_target_type 
    ON qbn.combination_alpha(target_type);

-- Unique constraint to prevent duplicate analyses
CREATE UNIQUE INDEX IF NOT EXISTS idx_combination_alpha_unique 
    ON qbn.combination_alpha(asset_id, horizon, combination_key, target_type, analyzed_at);

-- Comments
COMMENT ON TABLE qbn.combination_alpha IS 
    'Resultaten van Combination Alpha Analysis (Phase 2.5). Bevat OR, CI, Sens/Spec per signaalcombinatie.';

COMMENT ON COLUMN qbn.combination_alpha.combination_key IS 
    'Combinatie van leading|coincident|confirming states, bijv. "bullish|neutral|bearish"';

COMMENT ON COLUMN qbn.combination_alpha.classification IS 
    'golden_rule: OR>2, CI_lower>1, n>=30, MCC>0.1 | promising: OR>1.5, significant | noise: rest';

-- =============================================================================
-- Helper Views
-- =============================================================================

-- View for Golden Rules only
CREATE OR REPLACE VIEW qbn.v_golden_rules AS
SELECT 
    asset_id,
    horizon,
    combination_key,
    target_type,
    odds_ratio,
    or_ci_lower,
    or_ci_upper,
    sensitivity,
    specificity,
    mcc,
    n_with_combination,
    analyzed_at
FROM qbn.combination_alpha
WHERE classification = 'golden_rule'
ORDER BY odds_ratio DESC;

COMMENT ON VIEW qbn.v_golden_rules IS 
    'Gefilterde view met alleen Golden Rule combinaties';

-- View for latest analysis per asset
CREATE OR REPLACE VIEW qbn.v_combination_alpha_latest AS
SELECT DISTINCT ON (asset_id, horizon, combination_key, target_type)
    *
FROM qbn.combination_alpha
ORDER BY asset_id, horizon, combination_key, target_type, analyzed_at DESC;

COMMENT ON VIEW qbn.v_combination_alpha_latest IS 
    'Meest recente analyse resultaten per combinatie';

-- Summary view
CREATE OR REPLACE VIEW qbn.v_combination_alpha_summary AS
SELECT 
    asset_id,
    target_type,
    horizon,
    COUNT(*) AS n_combinations,
    SUM(CASE WHEN classification = 'golden_rule' THEN 1 ELSE 0 END) AS n_golden,
    SUM(CASE WHEN classification = 'promising' THEN 1 ELSE 0 END) AS n_promising,
    SUM(CASE WHEN classification = 'noise' THEN 1 ELSE 0 END) AS n_noise,
    MAX(odds_ratio) AS max_or,
    AVG(odds_ratio) AS avg_or,
    MAX(analyzed_at) AS latest_analysis
FROM qbn.combination_alpha
GROUP BY asset_id, target_type, horizon
ORDER BY asset_id, target_type, horizon;

COMMENT ON VIEW qbn.v_combination_alpha_summary IS 
    'Samenvatting van analyses per asset/target/horizon';

-- =============================================================================
-- Sample Queries
-- =============================================================================

-- Query: Top 10 Golden Rules voor een asset
-- SELECT * FROM qbn.v_golden_rules WHERE asset_id = 1 LIMIT 10;

-- Query: Summary per asset
-- SELECT * FROM qbn.v_combination_alpha_summary WHERE asset_id = 1;

-- Query: Combinaties met hoge sens EN spec
-- SELECT * FROM qbn.combination_alpha 
-- WHERE sensitivity > 0.6 AND specificity > 0.6 AND classification = 'golden_rule';

-- Query: Vergelijk horizons
-- SELECT horizon, AVG(odds_ratio), COUNT(*) FILTER (WHERE classification = 'golden_rule')
-- FROM qbn.combination_alpha WHERE asset_id = 1 GROUP BY horizon;

