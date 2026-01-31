-- v3.3 Triple Composite Architecture: Output Position Columns
-- Voegt kolommen toe aan qbn.output_position voor v3.3 inference outputs

-- Momentum Prediction (Leading-based)
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    momentum_prediction VARCHAR(20);
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_momentum_bearish REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_momentum_neutral REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_momentum_bullish REAL;

-- Volatility Regime (Coincident-based)
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    volatility_regime VARCHAR(20);
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_vol_low REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_vol_normal REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_vol_high REAL;

-- Exit Timing (Confirming-based)
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    exit_timing VARCHAR(20);
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_exit_now REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_hold REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_extend REAL;

-- Risk Adjusted Confidence (Ensemble)
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    risk_adjusted_confidence VARCHAR(20);
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_rac_very_low REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_rac_low REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_rac_medium REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_rac_high REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    p_rac_very_high REAL;

-- Delta Scores (voor debugging/monitoring)
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    delta_leading REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    delta_coincident REAL;
ALTER TABLE qbn.output_position ADD COLUMN IF NOT EXISTS
    delta_confirming REAL;

-- Indexes voor nieuwe kolommen (voor analyse queries)
CREATE INDEX IF NOT EXISTS idx_output_position_momentum 
    ON qbn.output_position(momentum_prediction);
CREATE INDEX IF NOT EXISTS idx_output_position_volatility 
    ON qbn.output_position(volatility_regime);
CREATE INDEX IF NOT EXISTS idx_output_position_exit_timing 
    ON qbn.output_position(exit_timing);
CREATE INDEX IF NOT EXISTS idx_output_position_rac 
    ON qbn.output_position(risk_adjusted_confidence);

-- Comments voor documentatie
COMMENT ON COLUMN qbn.output_position.momentum_prediction IS 
    'v3.3: Leading-based momentum prediction (bearish/neutral/bullish)';
COMMENT ON COLUMN qbn.output_position.volatility_regime IS 
    'v3.3: Coincident-based volatility regime (low_vol/normal/high_vol)';
COMMENT ON COLUMN qbn.output_position.exit_timing IS 
    'v3.3: Confirming-based exit timing (exit_now/hold/extend)';
COMMENT ON COLUMN qbn.output_position.risk_adjusted_confidence IS 
    'v3.3: Ensemble confidence (very_low/low/medium/high/very_high)';
COMMENT ON COLUMN qbn.output_position.delta_leading IS 
    'v3.3: Delta Leading score since entry';
COMMENT ON COLUMN qbn.output_position.delta_coincident IS 
    'v3.3: Delta Coincident score since entry';
COMMENT ON COLUMN qbn.output_position.delta_confirming IS 
    'v3.3: Delta Confirming score since entry';
