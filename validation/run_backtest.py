"""
Backtest Runner voor QBN v3.5.

Dit script wordt aangeroepen door het TSEM API endpoint en voert
een volledige backtest uit op basis van de configuratie in de database.
Gebruikt de moderne TradeSimulator engine (v3.5).
Entry-side: infer_batch(); position-side: infer_position() per actieve trade.

Usage:
    python -m validation.run_backtest --backtest-id <backtest_id>
"""

import argparse
import logging
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Setup logging
from core.logging_utils import setup_logging
setup_logging('backtest_runner')
logger = logging.getLogger(__name__)

from database.db import get_cursor
from validation.backtest_config import BacktestConfig
from validation.trade_simulator import TradeSimulator, Trade
from simulation.data_loader import BacktestDataLoader
from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
from inference.gpu.gpu_inference_engine import GPUInferenceEngine
from inference.node_types import SemanticClass


@dataclass
class BacktestInferenceState:
    """
    Lightweight inference state voor backtest: entry-side velden + optioneel position-side.
    Vervangt DualInferenceResult in run_backtest (geen PositionPredictionResult type).
    """
    # Entry-side (uit infer_batch)
    regime: str = ""
    trade_hypothesis: str = ""
    leading_composite: str = ""
    coincident_composite: str = ""
    confirming_composite: str = ""
    timestamp: Optional[datetime] = None
    asset_id: int = 0
    # Position-side (uit infer_position of defaults)
    momentum_prediction: str = ""
    volatility_regime: str = "normal"
    exit_timing: str = "hold"
    position_prediction: str = ""
    position_confidence: str = "medium"


def load_backtest_config(backtest_id: str) -> BacktestConfig:
    """
    Laad backtest configuratie uit de database.
    
    Args:
        backtest_id: Backtest run ID
        
    Returns:
        BacktestConfig object
    """
    with get_cursor() as cur:
        # Haal alle kolommen op (inclusief v3.4 params)
        # We gebruiken SELECT * FROM om flexibel te zijn met schema updates
        cur.execute("SELECT * FROM qbn.backtest_runs WHERE backtest_id = %s", (backtest_id,))
        
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Backtest {backtest_id} niet gevonden in database")
        
        # Haal kolomnamen op
        col_names = [desc[0] for desc in cur.description]
        row_dict = dict(zip(col_names, row))
        
        # Parse regime_filter JSONB
        import json
        regime_filter = row_dict.get('regime_filter', [])
        if isinstance(regime_filter, str):
            regime_filter = json.loads(regime_filter)
        
        # Map DB kolommen naar BacktestConfig
        return BacktestConfig(
            user_id=row_dict.get('user_id', 'default'),
            asset_id=row_dict['asset_id'],
            start_date=row_dict['start_date'],
            end_date=row_dict['end_date'],
            train_window_days=row_dict.get('train_window_days', 90),
            retrain_interval_days=row_dict.get('retrain_interval_days'),
            order_type=row_dict.get('order_type', 'market'),
            leverage=float(row_dict.get('leverage', 1.0)),
            position_size_pct=float(row_dict.get('position_size_pct', 2.0)) if row_dict.get('position_size_pct') else None,
            position_size_usd=float(row_dict.get('position_size_usd')) if row_dict.get('position_size_usd') else None,
            slippage_pct=float(row_dict.get('slippage_pct', 0.05)),
            stop_loss_atr_mult=float(row_dict.get('stop_loss_atr_mult', 1.0)),
            stop_loss_pct=float(row_dict.get('stop_loss_pct')) if row_dict.get('stop_loss_pct') else None,
            take_profit_atr_mult=float(row_dict.get('take_profit_atr_mult', 1.5)),
            take_profit_pct=float(row_dict.get('take_profit_pct')) if row_dict.get('take_profit_pct') else None,
            use_atr_based_exits=row_dict.get('use_atr_based_exits', True),
            trailing_stop_enabled=row_dict.get('trailing_stop_enabled', False),
            trailing_stop_pct=float(row_dict.get('trailing_stop_pct', 50.0)),
            trailing_activation_pct=float(row_dict.get('trailing_activation_pct', 0.5)),
            max_holding_time_hours=row_dict.get('max_holding_time_hours'),
            # v3.4 Parameters
            use_qbn_exit_timing=row_dict.get('use_qbn_exit_timing', True),
            entry_strength_threshold=row_dict.get('entry_strength_threshold', 'weak'),
            regime_filter=regime_filter,
            exit_on_momentum_reversal=row_dict.get('exit_on_momentum_reversal', False),
            volatility_position_sizing=row_dict.get('volatility_position_sizing', False),
            use_position_prediction_exit=row_dict.get('use_position_prediction_exit', False),
            # Fees
            maker_fee_pct=float(row_dict.get('maker_fee_pct', 0.02)),
            taker_fee_pct=float(row_dict.get('taker_fee_pct', 0.05)),
            funding_rate_enabled=row_dict.get('funding_rate_enabled', False),
            initial_capital_usd=float(row_dict.get('initial_capital_usd', 10000.0))
        )


def update_backtest_status(backtest_id: str, status: str, error_message: str = None, metrics: dict = None):
    """Update backtest status en resultaten in database."""
    with get_cursor() as cur:
        if status == 'running':
            cur.execute("""
                UPDATE qbn.backtest_runs
                SET status = %s, started_at = NOW()
                WHERE backtest_id = %s
            """, (status, backtest_id))
        elif status == 'failed':
            cur.execute("""
                UPDATE qbn.backtest_runs
                SET status = %s, error_message = %s, completed_at = NOW()
                WHERE backtest_id = %s
            """, (status, error_message, backtest_id))
        else:  # completed
            # Update metrics
            cur.execute("""
                UPDATE qbn.backtest_runs
                SET status = %s, completed_at = NOW(),
                    total_trades = %s,
                    total_pnl_usd = %s,
                    total_pnl_pct = %s,
                    win_rate_pct = %s,
                    profit_factor = %s,
                    max_drawdown_pct = %s,
                    sharpe_ratio = %s,
                    sortino_ratio = %s
                WHERE backtest_id = %s
            """, (
                status,
                metrics.get('total_trades', 0),
                metrics.get('total_pnl_usd', 0.0),
                metrics.get('total_pnl_pct', 0.0),
                metrics.get('win_rate_pct', 0.0),
                metrics.get('profit_factor', 0.0),
                metrics.get('max_drawdown_pct', 0.0),
                metrics.get('sharpe_ratio', 0.0),
                metrics.get('sortino_ratio', 0.0),
                backtest_id
            ))


def save_trades_to_db(backtest_id: str, trades: list):
    """Sla individuele trades op in qbn.backtest_trades."""
    if not trades:
        return
        
    with get_cursor() as cur:
        # Eerst opschonen voor het geval van re-run
        cur.execute("DELETE FROM qbn.backtest_trades WHERE backtest_id = %s", (backtest_id,))
        
        # Batch insert
        values = []
        for t in trades:
            values.append((
                backtest_id,
                t.entry_timestamp,
                t.exit_timestamp,
                t.direction,
                t.entry_price,
                t.exit_price,
                t.position_size_units,
                t.net_pnl_usd,
                t.pnl_pct,
                t.exit_reason,
                t.holding_duration_hours,
                t.mae_pct,
                t.mfe_pct
            ))
            
        args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", x).decode('utf-8') for x in values)
        cur.execute("INSERT INTO qbn.backtest_trades (backtest_id, entry_time, exit_time, direction, entry_price, exit_price, quantity, pnl_usd, pnl_pct, exit_reason, duration_hours, mae_pct, mfe_pct) VALUES " + args_str)


def run_backtest_internal(config: BacktestConfig) -> Tuple[Dict[str, Any], List[Trade]]:
    """
    Voer backtest uit zonder database interactie (voor validation cycle of tests).

    Args:
        config: BacktestConfig object

    Returns:
        Tuple[metrics_dict, list_of_closed_trades]
    """
    loader = BacktestDataLoader(config.asset_id)
    data_start = config.start_date - pd.Timedelta(days=config.train_window_days)
    logger.info("📥 Fetching full dataset...")
    full_df = loader.fetch_data(data_start, config.end_date)
    if full_df.empty:
        raise ValueError("Geen data gevonden voor opgegeven periode")
    if full_df["time_1"].dt.tz is None:
        full_df["time_1"] = full_df["time_1"].dt.tz_localize("UTC")
    full_df.set_index("time_1", inplace=True)
    full_df.sort_index(inplace=True)

    train_end = config.start_date
    train_df = full_df.loc[:train_end].copy()
    test_df = full_df.loc[train_end:].copy()
    if len(train_df) < 1000:
        raise ValueError(f"Te weinig data voor training ({len(train_df)} rijen)")

    logger.info(f"🧠 Training BN op {len(train_df)} candles...")
    generator = QBNv3CPTGenerator()
    
    # REASON: Preprocess train data om htf_regime en composite labels toe te voegen
    # EXPL: generate_htf_regime_cpt verwacht dat data['htf_regime'] al bestaat
    # Reset index first om pandas ambiguïteit te vermijden ('time_1' mag geen index én kolom zijn)
    train_df_reset = train_df.reset_index()
    train_df_reset = generator.preprocess_dataset(train_df_reset, config.asset_id)
    # Set index terug voor rest van processing
    train_df = train_df_reset.set_index("time_1")
    
    # REASON: Event-based CPTs (Momentum, Volatility, Exit, Position) roepen EventWindowDetector aan
    # die time_1 als kolom verwacht, niet als index. Gebruik train_df_reset voor die vier.
    cpts = {
        "HTF_Regime": generator.generate_htf_regime_cpt(config.asset_id, data=train_df),
        "Trade_Hypothesis": generator.generate_trade_hypothesis_cpt(config.asset_id, data=train_df),
        "Prediction_1h": generator.generate_prediction_cpt(config.asset_id, "1h", data=train_df),
        "Prediction_4h": generator.generate_prediction_cpt(config.asset_id, "4h", data=train_df),
        "Prediction_1d": generator.generate_prediction_cpt(config.asset_id, "1d", data=train_df),
        "Momentum_Prediction": generator.generate_momentum_prediction_cpt(config.asset_id, data=train_df_reset),
        "Volatility_Regime": generator.generate_volatility_regime_cpt(config.asset_id, data=train_df_reset),
        "Exit_Timing": generator.generate_exit_timing_cpt(config.asset_id, data=train_df_reset),
        "Position_Prediction": generator._generate_position_prediction_cpt(config.asset_id, data=train_df_reset),
    }
    generator.load_signal_classification(config.asset_id, "1h")
    for sc in SemanticClass:
        cpts[f"{sc.value.capitalize()}_Composite"] = generator.generate_composite_cpt(
            config.asset_id, sc, data=train_df, horizon="1h"
        )

    logger.info(f"🔮 Running inference op {len(test_df)} candles...")
    engine = GPUInferenceEngine(
        cpts=cpts,
        signal_classification=generator.signal_aggregator.signal_classification,
        threshold_loader=generator._get_threshold_loader(config.asset_id, "1h"),
    )
    predictions = engine.infer_batch(test_df)
    raw_scores = predictions.get("raw_composite_scores") or {}

    def _scalar(arr, idx: int) -> float:
        """Haal scalar uit array (numpy/cupy) voor index idx."""
        if arr is None or idx >= len(arr):
            return 0.0
        try:
            return float(arr[idx])
        except (TypeError, ValueError):
            return 0.0

    def _current_scores(i: int) -> Dict[str, float]:
        return {
            "leading": _scalar(raw_scores.get("leading"), i) if raw_scores else 0.0,
            "coincident": _scalar(raw_scores.get("coincident"), i) if raw_scores else 0.0,
            "confirming": _scalar(raw_scores.get("confirming"), i) if raw_scores else 0.0,
        }

    logger.info("📈 Running Trade Simulator...")
    simulator = TradeSimulator(config)
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_time = row.name if isinstance(test_df.index, pd.DatetimeIndex) else row["time_1"]
        current_price = float(row.get("close", row.get("close_60", 0)))
        atr = float(row.get("atr_14", row.get("atr", 20.0)))

        entry_state = BacktestInferenceState(
            asset_id=config.asset_id,
            timestamp=current_time,
            regime=str(predictions["regime"][i]) if i < len(predictions["regime"]) else "",
            trade_hypothesis=str(predictions["trade_hypothesis"][i]) if i < len(predictions["trade_hypothesis"]) else "",
            leading_composite=str(predictions["leading_composite"][i]) if i < len(predictions["leading_composite"]) else "neutral",
            coincident_composite=str(predictions["coincident_composite"][i]) if i < len(predictions["coincident_composite"]) else "neutral",
            confirming_composite=str(predictions["confirming_composite"][i]) if i < len(predictions["confirming_composite"]) else "neutral",
        )

        if not simulator.open_trades:
            should_enter, direction = simulator.should_enter_trade(entry_state)
            if should_enter and direction:
                entry_scores = _current_scores(i)
                simulator.open_trade(
                    entry_state,
                    current_price,
                    atr,
                    direction,
                    entry_composite_scores=entry_scores,
                )
        else:
            latest_inference = entry_state
            for trade in simulator.open_trades:
                if trade.entry_composite_scores is None:
                    continue
                time_since_min = (current_time - trade.entry_timestamp).total_seconds() / 60.0
                if trade.direction == "long":
                    current_pnl_atr = (current_price - trade.entry_price) / trade.atr_at_entry
                else:
                    current_pnl_atr = (trade.entry_price - current_price) / trade.atr_at_entry
                pos_result = engine.infer_position(
                    current_scores=_current_scores(i),
                    entry_scores=trade.entry_composite_scores,
                    time_since_entry_min=time_since_min,
                    current_pnl_atr=current_pnl_atr,
                )
                latest_inference = BacktestInferenceState(
                    momentum_prediction=pos_result.get("momentum_prediction", ""),
                    volatility_regime=pos_result.get("volatility_regime", "normal"),
                    exit_timing=pos_result.get("exit_timing", "hold"),
                    position_prediction=pos_result.get("position_prediction", ""),
                    position_confidence=pos_result.get("position_confidence", "medium"),
                )
                break

            synthetic_ohlc = pd.DataFrame([{
                "time": current_time,
                "open": row.get("open", current_price),
                "high": row.get("high", current_price),
                "low": row.get("low", current_price),
                "close": current_price,
            }])
            simulator.update_open_trades(current_time, synthetic_ohlc, latest_inference)

    metrics = simulator.get_metrics()
    return metrics, simulator.closed_trades


def run_backtest(backtest_id: str):
    """
    Voer backtest uit met TradeSimulator en schrijf resultaten naar DB.

    Args:
        backtest_id: Backtest run ID uit qbn.backtest_runs
    """
    try:
        logger.info(f"🚀 Start backtest: {backtest_id}")
        update_backtest_status(backtest_id, "running")
        config = load_backtest_config(backtest_id)
        logger.info(
            f"📋 Configuratie geladen: Asset {config.asset_id}, "
            f"{config.start_date.date()} tot {config.end_date.date()}"
        )
        metrics, trades = run_backtest_internal(config)
        save_trades_to_db(backtest_id, trades)
        update_backtest_status(backtest_id, "completed", metrics=metrics)
        logger.info(f"✅ Backtest {backtest_id} succesvol voltooid")
        logger.info(f"   PnL: ${metrics.get('total_pnl_usd', 0):.2f} ({metrics.get('total_pnl_pct', 0):.2f}%)")
        logger.info(f"   Trades: {metrics.get('total_trades', 0)}")
    except Exception as e:
        logger.error(f"❌ Backtest {backtest_id} gefaald: {e}", exc_info=True)
        update_backtest_status(backtest_id, "failed", str(e))
        raise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run QBN backtest")
    parser.add_argument(
        '--backtest-id',
        type=str,
        required=True,
        help='Backtest run ID uit qbn.backtest_runs'
    )
    
    args = parser.parse_args()
    
    try:
        run_backtest(args.backtest_id)
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
