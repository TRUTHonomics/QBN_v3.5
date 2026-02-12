"""
Script om v3.4 position management te activeren en vergelijkende backtest te draaien.

Vergelijkt:
- Baseline: Alleen Entry-side + ATR exits (huidige default)
- Full v3.4: Entry-side + alle 4 position management nodes actief

Usage:
    docker exec QBN_v4_Dagster_Webserver python /app/scripts/compare_v34_backtest.py --asset-id 1
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys

# REASON: Add /app to Python path voor imports
sys.path.insert(0, '/app')

try:
    from core.logging_utils import setup_logging
    setup_logging('compare_v34_backtest')
except ModuleNotFoundError:
    # Fallback als core module niet beschikbaar is
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

from database.db import get_cursor
from validation.backtest_config import BacktestConfig
from validation.run_backtest import run_backtest_internal


def parse_args():
    parser = argparse.ArgumentParser(description='Compare baseline vs full v3.4 position management backtest')
    parser.add_argument('--asset-id', type=int, default=1, help='Asset ID to backtest')
    parser.add_argument('--lookback-days', type=int, default=90, help='Backtest lookback period in days')
    parser.add_argument('--train-window-days', type=int, default=90, help='Training window for walk-forward')
    return parser.parse_args()


def create_baseline_config(asset_id: int, start_date: datetime, end_date: datetime, train_window_days: int) -> BacktestConfig:
    """Baseline configuratie: alleen entry-side + ATR exits."""
    return BacktestConfig(
        user_id='compare_v34',
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
        train_window_days=train_window_days,
        retrain_interval_days=None,
        order_type='market',
        leverage=1.0,
        position_size_pct=2.0,
        slippage_pct=0.05,
        stop_loss_atr_mult=1.0,
        take_profit_atr_mult=1.5,
        use_atr_based_exits=True,
        trailing_stop_enabled=False,
        max_holding_time_hours=None,
        # BASELINE: alleen exit_timing, rest UIT
        use_qbn_exit_timing=True,  # Dit stond al aan
        exit_on_momentum_reversal=False,
        volatility_position_sizing=False,
        use_position_prediction_exit=False,
        entry_strength_threshold='weak',
        regime_filter=[],
        maker_fee_pct=0.02,
        taker_fee_pct=0.05,
        funding_rate_enabled=False,
        initial_capital_usd=10000.0
    )


def create_full_v34_config(asset_id: int, start_date: datetime, end_date: datetime, train_window_days: int) -> BacktestConfig:
    """Full v3.4 configuratie: entry-side + alle 4 position management nodes."""
    return BacktestConfig(
        user_id='compare_v34',
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
        train_window_days=train_window_days,
        retrain_interval_days=None,
        order_type='market',
        leverage=1.0,
        position_size_pct=2.0,
        slippage_pct=0.05,
        stop_loss_atr_mult=1.0,
        take_profit_atr_mult=1.5,
        use_atr_based_exits=True,
        trailing_stop_enabled=False,
        max_holding_time_hours=None,
        # FULL V3.4: alle position management nodes AAN
        use_qbn_exit_timing=True,
        exit_on_momentum_reversal=True,
        volatility_position_sizing=True,
        use_position_prediction_exit=True,
        entry_strength_threshold='weak',
        regime_filter=[],
        maker_fee_pct=0.02,
        taker_fee_pct=0.05,
        funding_rate_enabled=False,
        initial_capital_usd=10000.0
    )


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("v3.4 Position Management Comparison Backtest")
    logger.info("="*80)
    logger.info(f"Asset ID: {args.asset_id}")
    logger.info(f"Lookback: {args.lookback_days} days")
    logger.info(f"Train window: {args.train_window_days} days")
    
    # Bereken date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.lookback_days)
    
    logger.info(f"Date range: {start_date.date()} tot {end_date.date()}")
    
    # Maak configs
    baseline_config = create_baseline_config(args.asset_id, start_date, end_date, args.train_window_days)
    full_v34_config = create_full_v34_config(args.asset_id, start_date, end_date, args.train_window_days)
    
    # Run baseline
    logger.info("\n" + "="*80)
    logger.info("RUN 1: BASELINE (alleen Entry + Exit_Timing)")
    logger.info("="*80)
    logger.info("Config:")
    logger.info(f"  use_qbn_exit_timing: {baseline_config.use_qbn_exit_timing}")
    logger.info(f"  exit_on_momentum_reversal: {baseline_config.exit_on_momentum_reversal}")
    logger.info(f"  volatility_position_sizing: {baseline_config.volatility_position_sizing}")
    logger.info(f"  use_position_prediction_exit: {baseline_config.use_position_prediction_exit}")
    
    try:
        baseline_metrics, baseline_trades = run_backtest_internal(baseline_config)
        logger.info("\nBASELINE RESULTATEN:")
        logger.info(f"  Total Trades: {baseline_metrics.get('total_trades', 0)}")
        logger.info(f"  Total PnL: ${baseline_metrics.get('total_pnl_usd', 0):.2f} ({baseline_metrics.get('total_pnl_pct', 0):.2f}%)")
        logger.info(f"  Win Rate: {baseline_metrics.get('win_rate_pct', 0):.2f}%")
        logger.info(f"  Profit Factor: {baseline_metrics.get('profit_factor', 0):.2f}")
        logger.info(f"  Max Drawdown: {baseline_metrics.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"  Sharpe Ratio: {baseline_metrics.get('sharpe_ratio', 0):.2f}")
    except Exception as e:
        logger.error(f"Baseline backtest failed: {e}", exc_info=True)
        baseline_metrics = None
    
    # Run full v3.4
    logger.info("\n" + "="*80)
    logger.info("RUN 2: FULL V3.4 (alle 4 position management nodes)")
    logger.info("="*80)
    logger.info("Config:")
    logger.info(f"  use_qbn_exit_timing: {full_v34_config.use_qbn_exit_timing}")
    logger.info(f"  exit_on_momentum_reversal: {full_v34_config.exit_on_momentum_reversal}")
    logger.info(f"  volatility_position_sizing: {full_v34_config.volatility_position_sizing}")
    logger.info(f"  use_position_prediction_exit: {full_v34_config.use_position_prediction_exit}")
    
    try:
        full_v34_metrics, full_v34_trades = run_backtest_internal(full_v34_config)
        logger.info("\nFULL V3.4 RESULTATEN:")
        logger.info(f"  Total Trades: {full_v34_metrics.get('total_trades', 0)}")
        logger.info(f"  Total PnL: ${full_v34_metrics.get('total_pnl_usd', 0):.2f} ({full_v34_metrics.get('total_pnl_pct', 0):.2f}%)")
        logger.info(f"  Win Rate: {full_v34_metrics.get('win_rate_pct', 0):.2f}%")
        logger.info(f"  Profit Factor: {full_v34_metrics.get('profit_factor', 0):.2f}")
        logger.info(f"  Max Drawdown: {full_v34_metrics.get('max_drawdown_pct', 0):.2f}%)")
        logger.info(f"  Sharpe Ratio: {full_v34_metrics.get('sharpe_ratio', 0):.2f}")
    except Exception as e:
        logger.error(f"Full v3.4 backtest failed: {e}", exc_info=True)
        full_v34_metrics = None
    
    # Vergelijk
    if baseline_metrics and full_v34_metrics:
        logger.info("\n" + "="*80)
        logger.info("COMPARISON")
        logger.info("="*80)
        
        delta_trades = full_v34_metrics['total_trades'] - baseline_metrics['total_trades']
        delta_pnl_usd = full_v34_metrics['total_pnl_usd'] - baseline_metrics['total_pnl_usd']
        delta_pnl_pct = full_v34_metrics['total_pnl_pct'] - baseline_metrics['total_pnl_pct']
        delta_winrate = full_v34_metrics['win_rate_pct'] - baseline_metrics['win_rate_pct']
        delta_pf = full_v34_metrics['profit_factor'] - baseline_metrics['profit_factor']
        delta_dd = full_v34_metrics['max_drawdown_pct'] - baseline_metrics['max_drawdown_pct']
        delta_sharpe = full_v34_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
        
        logger.info(f"Trades:         {delta_trades:+d} ({full_v34_metrics['total_trades']} vs {baseline_metrics['total_trades']})")
        logger.info(f"PnL USD:        ${delta_pnl_usd:+.2f} ({full_v34_metrics['total_pnl_usd']:.2f} vs {baseline_metrics['total_pnl_usd']:.2f})")
        logger.info(f"PnL %:          {delta_pnl_pct:+.2f}% ({full_v34_metrics['total_pnl_pct']:.2f}% vs {baseline_metrics['total_pnl_pct']:.2f}%)")
        logger.info(f"Win Rate:       {delta_winrate:+.2f}% ({full_v34_metrics['win_rate_pct']:.2f}% vs {baseline_metrics['win_rate_pct']:.2f}%)")
        logger.info(f"Profit Factor:  {delta_pf:+.2f} ({full_v34_metrics['profit_factor']:.2f} vs {baseline_metrics['profit_factor']:.2f})")
        logger.info(f"Max Drawdown:   {delta_dd:+.2f}% ({full_v34_metrics['max_drawdown_pct']:.2f}% vs {baseline_metrics['max_drawdown_pct']:.2f}%)")
        logger.info(f"Sharpe Ratio:   {delta_sharpe:+.2f} ({full_v34_metrics['sharpe_ratio']:.2f} vs {baseline_metrics['sharpe_ratio']:.2f})")
        
        # Bepaal verdict
        improvement_score = 0
        if delta_pnl_pct > 0:
            improvement_score += 3
        if delta_winrate > 0:
            improvement_score += 2
        if delta_pf > 0:
            improvement_score += 2
        if delta_dd < 0:  # Lagere drawdown is beter
            improvement_score += 1
        if delta_sharpe > 0:
            improvement_score += 2
        
        logger.info("\n" + "="*80)
        logger.info(f"IMPROVEMENT SCORE: {improvement_score}/10")
        if improvement_score >= 7:
            logger.info("VERDICT: SIGNIFICANT IMPROVEMENT - activeer v3.4 position management standaard")
        elif improvement_score >= 4:
            logger.info("VERDICT: MARGINAL IMPROVEMENT - overweeg activering na verdere tuning")
        else:
            logger.info("VERDICT: NO IMPROVEMENT - position management heeft geen toegevoegde waarde of heeft verdere tuning nodig")
        logger.info("="*80)
        
        # Sla resultaten op
        output_dir = Path('_validation/v34_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'comparison_{timestamp}.json'
        
        results = {
            'timestamp': timestamp,
            'asset_id': args.asset_id,
            'lookback_days': args.lookback_days,
            'baseline': baseline_metrics,
            'full_v34': full_v34_metrics,
            'deltas': {
                'trades': delta_trades,
                'pnl_usd': delta_pnl_usd,
                'pnl_pct': delta_pnl_pct,
                'win_rate_pct': delta_winrate,
                'profit_factor': delta_pf,
                'max_drawdown_pct': delta_dd,
                'sharpe_ratio': delta_sharpe
            },
            'improvement_score': improvement_score
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResultaten opgeslagen in: {output_file}")
    
    logger.info("\nComparison backtest complete!")


if __name__ == '__main__':
    main()
