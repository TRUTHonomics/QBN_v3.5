-- Migration 006: Create Backtest Tables
-- Doel: Tabellen voor backtest runs en trades om historische simulaties op te slaan
-- Datum: 2026-01-25

-- ==========================================
-- 1. BACKTEST RUNS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS qbn.backtest_runs (
    backtest_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL DEFAULT 'default',
    asset_id INTEGER NOT NULL,
    
    -- Tijdsvenster
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    train_window_days INTEGER NOT NULL,
    retrain_interval_days INTEGER,
    
    -- Entry Parameters
    order_type VARCHAR(20) DEFAULT 'market',
    leverage DECIMAL(6,2) DEFAULT 1.0,
    position_size_pct DECIMAL(6,3),
    position_size_usd DECIMAL(12,2),
    slippage_pct DECIMAL(6,4) DEFAULT 0.05,
    
    -- Exit Parameters
    stop_loss_atr_mult DECIMAL(6,3),
    stop_loss_pct DECIMAL(6,3),
    take_profit_atr_mult DECIMAL(6,3),
    take_profit_pct DECIMAL(6,3),
    use_atr_based_exits BOOLEAN DEFAULT true,
    
    -- Position Management
    trailing_stop_enabled BOOLEAN DEFAULT false,
    trailing_stop_pct DECIMAL(6,3),
    trailing_activation_pct DECIMAL(6,3),
    max_holding_time_hours INTEGER,
    use_qbn_exit_timing BOOLEAN DEFAULT true,
    
    -- QBN Signal Filters (stored as JSONB)
    min_trade_hypothesis VARCHAR(30),
    min_momentum_prediction VARCHAR(30),
    min_position_confidence VARCHAR(30),
    regime_filter JSONB,
    exit_on_momentum_reversal BOOLEAN DEFAULT false,
    
    -- Fees & Costs
    maker_fee_pct DECIMAL(6,4) DEFAULT 0.02,
    taker_fee_pct DECIMAL(6,4) DEFAULT 0.05,
    funding_rate_enabled BOOLEAN DEFAULT false,
    
    -- Results Summary
    initial_capital_usd DECIMAL(12,2) NOT NULL,
    final_capital_usd DECIMAL(12,2),
    total_pnl_usd DECIMAL(12,2),
    total_pnl_pct DECIMAL(8,4),
    
    -- Trade Statistics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    breakeven_trades INTEGER DEFAULT 0,
    win_rate_pct DECIMAL(6,2),
    
    -- Performance Metrics
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    max_drawdown_pct DECIMAL(8,4),
    max_drawdown_usd DECIMAL(12,2),
    profit_factor DECIMAL(8,4),
    avg_win_usd DECIMAL(12,2),
    avg_loss_usd DECIMAL(12,2),
    avg_trade_duration_hours DECIMAL(8,2),
    
    -- Metadata
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    
    -- Foreign Keys
    FOREIGN KEY (asset_id) REFERENCES symbols.symbols(id) ON DELETE CASCADE
);

-- Indexes voor backtest_runs
CREATE INDEX IF NOT EXISTS idx_backtest_runs_user_id ON qbn.backtest_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_asset_id ON qbn.backtest_runs(asset_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON qbn.backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_created_at ON qbn.backtest_runs(created_at DESC);

COMMENT ON TABLE qbn.backtest_runs IS 'Backtest configuraties en resultaten voor QBN signaal simulaties';

-- ==========================================
-- 2. BACKTEST TRADES TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS qbn.backtest_trades (
    trade_id SERIAL PRIMARY KEY,
    backtest_id VARCHAR(64) NOT NULL,
    asset_id INTEGER NOT NULL,
    
    -- Trade Identificatie
    signal_timestamp TIMESTAMPTZ NOT NULL,
    direction VARCHAR(10) NOT NULL, -- long, short
    
    -- Entry Details
    entry_timestamp TIMESTAMPTZ NOT NULL,
    entry_price DECIMAL(18,8) NOT NULL,
    position_size_usd DECIMAL(12,2) NOT NULL,
    position_size_units DECIMAL(18,8),
    entry_fees_usd DECIMAL(12,4),
    entry_slippage_pct DECIMAL(6,4),
    
    -- QBN Signal Context (at entry)
    htf_regime VARCHAR(40),
    trade_hypothesis VARCHAR(40),
    momentum_prediction VARCHAR(30),
    volatility_regime VARCHAR(30),
    exit_timing VARCHAR(30),
    position_confidence VARCHAR(30),
    leading_composite VARCHAR(30),
    coincident_composite VARCHAR(30),
    confirming_composite VARCHAR(30),
    
    -- Planned Exits
    planned_stop_loss DECIMAL(18,8),
    planned_take_profit DECIMAL(18,8),
    atr_at_entry DECIMAL(18,8),
    
    -- Exit Details
    exit_timestamp TIMESTAMPTZ,
    exit_price DECIMAL(18,8),
    exit_reason VARCHAR(40), -- stop_loss, take_profit, timeout, trailing_stop, qbn_exit_signal, manual
    exit_fees_usd DECIMAL(12,4),
    
    -- Trade Outcome
    gross_pnl_usd DECIMAL(12,4),
    net_pnl_usd DECIMAL(12,4),
    pnl_pct DECIMAL(8,4),
    mae_pct DECIMAL(8,4), -- Maximum Adverse Excursion
    mfe_pct DECIMAL(8,4), -- Maximum Favorable Excursion
    holding_duration_hours DECIMAL(8,2),
    
    -- Trailing Stop Details (if applicable)
    trailing_stop_activated BOOLEAN DEFAULT false,
    trailing_stop_highest_price DECIMAL(18,8),
    trailing_stop_lowest_price DECIMAL(18,8),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Foreign Keys
    FOREIGN KEY (backtest_id) REFERENCES qbn.backtest_runs(backtest_id) ON DELETE CASCADE,
    FOREIGN KEY (asset_id) REFERENCES symbols.symbols(id) ON DELETE CASCADE
);

-- Indexes voor backtest_trades
CREATE INDEX IF NOT EXISTS idx_backtest_trades_backtest_id ON qbn.backtest_trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_asset_id ON qbn.backtest_trades(asset_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_entry_timestamp ON qbn.backtest_trades(entry_timestamp);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_exit_reason ON qbn.backtest_trades(exit_reason);

COMMENT ON TABLE qbn.backtest_trades IS 'Individuele trades uit backtest simulaties met volledige trade lifecycle';

-- ==========================================
-- 3. EQUITY CURVE VIEW (for charting)
-- ==========================================
CREATE OR REPLACE VIEW qbn.backtest_equity_curve AS
SELECT 
    backtest_id,
    exit_timestamp::date AS date,
    SUM(net_pnl_usd) AS daily_pnl,
    SUM(SUM(net_pnl_usd)) OVER (
        PARTITION BY backtest_id 
        ORDER BY exit_timestamp::date
    ) AS cumulative_pnl,
    COUNT(*) AS trade_count
FROM qbn.backtest_trades
WHERE exit_timestamp IS NOT NULL
GROUP BY backtest_id, exit_timestamp::date
ORDER BY backtest_id, date;

COMMENT ON VIEW qbn.backtest_equity_curve IS 'Daily equity curve voor backtest visualisaties';

-- ==========================================
-- 4. GRANT PERMISSIONS
-- ==========================================
GRANT SELECT, INSERT, UPDATE, DELETE ON qbn.backtest_runs TO pipeline;
GRANT SELECT, INSERT, UPDATE, DELETE ON qbn.backtest_trades TO pipeline;
GRANT USAGE, SELECT ON SEQUENCE qbn.backtest_trades_trade_id_seq TO pipeline;
GRANT SELECT ON qbn.backtest_equity_curve TO pipeline;

GRANT SELECT, INSERT, UPDATE, DELETE ON qbn.backtest_runs TO cursor_ai;
GRANT SELECT, INSERT, UPDATE, DELETE ON qbn.backtest_trades TO cursor_ai;
GRANT USAGE, SELECT ON SEQUENCE qbn.backtest_trades_trade_id_seq TO cursor_ai;
GRANT SELECT ON qbn.backtest_equity_curve TO cursor_ai;
