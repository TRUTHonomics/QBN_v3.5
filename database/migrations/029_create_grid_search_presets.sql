-- Migration: 029 - Create Grid Search Presets Table
-- Purpose: Store reusable Grid Search parameter configurations
-- Author: QBN v3.4
-- Date: 2026-01-26

-- Create grid search presets table
CREATE TABLE IF NOT EXISTS qbn.grid_search_presets (
    preset_id SERIAL PRIMARY KEY,
    preset_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    parameters JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index voor snelle preset lookups
CREATE INDEX IF NOT EXISTS idx_grid_search_presets_name 
ON qbn.grid_search_presets(preset_name);

-- Trigger voor updated_at
CREATE OR REPLACE FUNCTION qbn.update_grid_search_presets_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_grid_search_presets_timestamp
    BEFORE UPDATE ON qbn.grid_search_presets
    FOR EACH ROW
    EXECUTE FUNCTION qbn.update_grid_search_presets_timestamp();

-- Seed met default preset (conservatieve strategie)
INSERT INTO qbn.grid_search_presets (preset_name, description, parameters)
VALUES (
    'default_conservative',
    'Conservatieve grid search configuratie met standaard ATR-based exits en momentum filtering',
    '{
        "stop_loss_atr_mult": {"enabled": true, "type": "numeric", "min": 1.0, "max": 2.0, "step": 0.5},
        "take_profit_atr_mult": {"enabled": true, "type": "numeric", "min": 2.0, "max": 4.0, "step": 1.0},
        "min_trade_hypothesis": {"enabled": true, "type": "categorical", "values": ["moderate_long", "strong_long"]},
        "min_momentum_prediction": {"enabled": false, "type": "categorical", "values": ["bullish", null]},
        "min_position_confidence": {"enabled": false, "type": "categorical", "values": ["low", "medium", "high"]},
        "regime_filter": {"enabled": false, "type": "categorical", "values": []},
        "use_qbn_exit_timing": {"enabled": false, "type": "boolean", "values": [true, false]},
        "exit_on_momentum_reversal": {"enabled": true, "type": "boolean", "values": [true, false]},
        "max_holding_time_hours": {"enabled": false, "type": "numeric", "min": 24, "max": 168, "step": 24},
        "leverage": {"enabled": true, "type": "numeric", "min": 1.0, "max": 5.0, "step": 2.0},
        "trailing_stop_enabled": {"enabled": false, "type": "boolean", "values": [true, false]},
        "trailing_activation_pct": {"enabled": false, "type": "numeric", "min": 0.5, "max": 2.0, "step": 0.5},
        "trailing_stop_pct": {"enabled": false, "type": "numeric", "min": 25.0, "max": 75.0, "step": 25.0}
    }'::jsonb
) ON CONFLICT (preset_name) DO NOTHING;

-- Seed met aggressive preset
INSERT INTO qbn.grid_search_presets (preset_name, description, parameters)
VALUES (
    'aggressive_scalper',
    'Agressieve scalping configuratie met strakke stops en hogere leverage',
    '{
        "stop_loss_atr_mult": {"enabled": true, "type": "numeric", "min": 0.5, "max": 1.5, "step": 0.5},
        "take_profit_atr_mult": {"enabled": true, "type": "numeric", "min": 1.0, "max": 2.5, "step": 0.5},
        "min_trade_hypothesis": {"enabled": true, "type": "categorical", "values": ["weak_long", "moderate_long", "strong_long"]},
        "min_momentum_prediction": {"enabled": true, "type": "categorical", "values": ["bullish"]},
        "min_position_confidence": {"enabled": true, "type": "categorical", "values": ["medium", "high"]},
        "regime_filter": {"enabled": false, "type": "categorical", "values": []},
        "use_qbn_exit_timing": {"enabled": true, "type": "boolean", "values": [true, false]},
        "exit_on_momentum_reversal": {"enabled": true, "type": "boolean", "values": [true]},
        "max_holding_time_hours": {"enabled": true, "type": "numeric", "min": 4, "max": 24, "step": 4},
        "leverage": {"enabled": true, "type": "numeric", "min": 3.0, "max": 10.0, "step": 2.0},
        "trailing_stop_enabled": {"enabled": true, "type": "boolean", "values": [true, false]},
        "trailing_activation_pct": {"enabled": true, "type": "numeric", "min": 0.3, "max": 1.0, "step": 0.3},
        "trailing_stop_pct": {"enabled": true, "type": "numeric", "min": 30.0, "max": 60.0, "step": 15.0}
    }'::jsonb
) ON CONFLICT (preset_name) DO NOTHING;

-- Comment op tabel
COMMENT ON TABLE qbn.grid_search_presets IS 'Reusable parameter configurations for Grid Search optimization';
COMMENT ON COLUMN qbn.grid_search_presets.parameters IS 'JSONB structure: {"param_name": {"enabled": bool, "type": "numeric"|"categorical"|"boolean", "min": float, "max": float, "step": float, "values": [...]}}';
