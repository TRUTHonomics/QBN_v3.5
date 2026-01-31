-- QBN Backtest Tables
-- Schema voor opslaan van backtest runs en trade results

-- ============================================================================
-- Table: qbn.backtest_runs
-- Slaat metadata op van elke backtest run
-- ============================================================================
CREATE TABLE IF NOT EXISTS qbn.backtest_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    asset_id INTEGER NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    
    -- Training parameters
    train_window_days INTEGER NOT NULL DEFAULT 90,
    retrain_interval_days INTEGER NOT NULL DEFAULT 30,
    
    -- Entry parameters
    order_type VARCHAR(20) NOT NULL DEFAULT 'market',
    leverage DECIMAL(10,2) NOT NULL DEFAULT 1.0,
    position_size_pct DECIMAL(10,4) NOT NULL DEFAULT 2.0,
    position_size_usd DECIMAL(18,2),
    slippage_pct DECIMAL(10,4) NOT NULL DEFAULT 0.05,
    
    -- Exit parameters
    stop_loss_atr_mult DECIMAL(10,4),
    stop_loss_pct DECIMAL(10,4),
    take_profit_atr_mult DECIMAL(10,4),
    take_profit_pct DECIMAL(10,4),
    use_atr_based_exits BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Position management
    trailing_stop_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    trailing_stop_pct DECIMAL(10,4) NOT NULL DEFAULT 50.0,
    trailing_activation_pct DECIMAL(10,4) NOT NULL DEFAULT 0.5,
    max_holding_time_hours INTEGER,
    use_qbn_exit_timing BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- QBN filters
    min_trade_hypothesis VARCHAR(50),
    min_momentum_prediction VARCHAR(50),
    min_position_confidence VARCHAR(50),
    regime_filter JSONB,
    exit_on_momentum_reversal BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Fees & costs
    maker_fee_pct DECIMAL(10,4) NOT NULL DEFAULT 0.02,
    taker_fee_pct DECIMAL(10,4) NOT NULL DEFAULT 0.05,
    funding_rate_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Capital
    initial_capital_usd DECIMAL(18,2) NOT NULL DEFAULT 10000.0,
    
    -- Results (populated after run)
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate_pct DECIMAL(10,4),
    total_pnl_usd DECIMAL(18,2),
    total_pnl_pct DECIMAL(10,4),
    max_drawdown_pct DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    profit_factor DECIMAL(10,4),
    avg_win_usd DECIMAL(18,2),
    avg_loss_usd DECIMAL(18,2),
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    total_fees_paid DECIMAL(18,2),
    
    -- Metadata
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending, running, completed, failed
    error_message TEXT,
    execution_time_seconds INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    CONSTRAINT backtest_runs_asset_fk FOREIGN KEY (asset_id) 
        REFERENCES symbols.symbols(asset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_asset ON qbn.backtest_runs(asset_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_created ON qbn.backtest_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON qbn.backtest_runs(status);

COMMENT ON TABLE qbn.backtest_runs IS 'Metadata en resultaten van backtest runs';

-- ============================================================================
-- Table: qbn.backtest_trades
-- Slaat individuele trades op van elke backtest run
-- ============================================================================
CREATE TABLE IF NOT EXISTS qbn.backtest_trades (
    trade_id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    asset_id INTEGER NOT NULL,
    
    -- Entry details
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price DECIMAL(18,8) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- long, short
    position_size_usd DECIMAL(18,2) NOT NULL,
    leverage DECIMAL(10,2) NOT NULL,
    entry_atr DECIMAL(18,8) NOT NULL,
    
    -- QBN signals at entry
    trade_hypothesis VARCHAR(50),
    entry_confidence VARCHAR(50),
    position_confidence VARCHAR(50),
    momentum_prediction VARCHAR(50),
    volatility_regime VARCHAR(50),
    exit_timing VARCHAR(50),
    htf_regime VARCHAR(50),
    leading_composite VARCHAR(50),
    coincident_composite VARCHAR(50),
    confirming_composite VARCHAR(50),
    
    -- Planned exits
    planned_stop_loss DECIMAL(18,8) NOT NULL,
    planned_take_profit DECIMAL(18,8) NOT NULL,
    trailing_stop_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Exit details
    exit_time TIMESTAMPTZ,
    exit_price DECIMAL(18,8),
    exit_reason VARCHAR(50), -- stop_loss, take_profit, trailing_stop, timeout, qbn_exit_signal
    
    -- Performance
    pnl_usd DECIMAL(18,2),
    pnl_pct DECIMAL(10,4),
    mae_pct DECIMAL(10,4), -- Maximum Adverse Excursion
    mfe_pct DECIMAL(10,4), -- Maximum Favorable Excursion
    holding_time_hours DECIMAL(10,2),
    entry_slippage_pct DECIMAL(10,4),
    exit_slippage_pct DECIMAL(10,4),
    total_fees_usd DECIMAL(18,2),
    net_pnl_usd DECIMAL(18,2),
    
    -- Running equity (voor equity curve)
    equity_before DECIMAL(18,2),
    equity_after DECIMAL(18,2),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT backtest_trades_run_fk FOREIGN KEY (run_id) 
        REFERENCES qbn.backtest_runs(run_id) ON DELETE CASCADE,
    CONSTRAINT backtest_trades_direction_check CHECK (direction IN ('long', 'short'))
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON qbn.backtest_trades(run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_entry_time ON qbn.backtest_trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_exit_reason ON qbn.backtest_trades(exit_reason);

COMMENT ON TABLE qbn.backtest_trades IS 'Individuele trades van backtest runs';

-- ============================================================================
-- View: qbn.backtest_summary
-- Quick overview van alle backtest runs
-- ============================================================================
CREATE OR REPLACE VIEW qbn.backtest_summary AS
SELECT 
    br.run_id,
    br.asset_id,
    s.symbol,
    br.start_date,
    br.end_date,
    DATE_PART('day', br.end_date - br.start_date) AS days,
    br.initial_capital_usd,
    br.total_trades,
    br.winning_trades,
    br.losing_trades,
    br.win_rate_pct,
    br.total_pnl_usd,
    br.total_pnl_pct,
    br.max_drawdown_pct,
    br.sharpe_ratio,
    br.profit_factor,
    br.status,
    br.created_at
FROM qbn.backtest_runs br
LEFT JOIN symbols.symbols s ON br.asset_id = s.asset_id
ORDER BY br.created_at DESC;

COMMENT ON VIEW qbn.backtest_summary IS 'Overzicht van alle backtest runs met symbool info';
