-- Migration: Split qbn.cpt_cache into structural/entry/position tables
-- Purpose: Enable independent computation of structural, entry-side, and position-side CPTs
-- Aligns with QBN v3.4 dual-prediction architecture

-- Create structural CPT table (HTF_Regime only)
CREATE TABLE qbn.cpt_cache_structural (LIKE qbn.cpt_cache INCLUDING ALL);

-- Create entry-side CPT table (composites + trade hypothesis + predictions)
CREATE TABLE qbn.cpt_cache_entry (LIKE qbn.cpt_cache INCLUDING ALL);

-- Create position-side CPT table (momentum, volatility, exit timing, position prediction)
CREATE TABLE qbn.cpt_cache_position (LIKE qbn.cpt_cache INCLUDING ALL);

-- Migrate existing data from qbn.cpt_cache to structural table
INSERT INTO qbn.cpt_cache_structural 
SELECT * FROM qbn.cpt_cache 
WHERE node_name = 'HTF_Regime';

-- Migrate existing data from qbn.cpt_cache to entry table
INSERT INTO qbn.cpt_cache_entry 
SELECT * FROM qbn.cpt_cache
WHERE node_name IN (
    'Leading_Composite',
    'Coincident_Composite',
    'Confirming_Composite',
    'Trade_Hypothesis',
    'Prediction_1h',
    'Prediction_4h',
    'Prediction_1d'
);

-- Migrate existing data from qbn.cpt_cache to position table
INSERT INTO qbn.cpt_cache_position 
SELECT * FROM qbn.cpt_cache
WHERE node_name IN (
    'Momentum_Prediction',
    'Volatility_Regime',
    'Exit_Timing',
    'Position_Prediction'
);

-- Note: qbn.cpt_cache remains as fallback until all systems are verified
-- Position_Confidence CPTs (legacy) remain in qbn.cpt_cache but are deprecated in v3.4
