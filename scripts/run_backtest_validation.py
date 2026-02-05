#!/usr/bin/env python3
"""
Walk-forward backtest voor validation cycle (Dagster / menu-achtige flow).

Gebruikt BacktestConfig + run_backtest_internal + save_backtest_report.
Niet afhankelijk van qbn.backtest_runs (geen --backtest-id).

Usage:
    python scripts/run_backtest_validation.py --asset-id 1
    python scripts/run_backtest_validation.py --asset-id 1 --start 2024-06-01 --end 2025-01-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest (validation flow)")
    parser.add_argument("--asset-id", type=int, required=True, help="Asset ID")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    try:
        from validation.backtest_config import BacktestConfig
        from validation.run_backtest import run_backtest_internal
        from validation.backtest_report import save_backtest_report

        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        config = BacktestConfig(
            asset_id=args.asset_id,
            start_date=start_dt,
            end_date=end_dt,
            train_window_days=90,
            initial_capital_usd=10000.0,
            leverage=1.0,
            position_size_pct=2.0,
            stop_loss_atr_mult=1.0,
            take_profit_atr_mult=1.5,
            use_atr_based_exits=True,
            use_qbn_exit_timing=True,
            entry_strength_threshold="weak",
        )

        report_dir = PROJECT_ROOT / "_validation" / f"asset_{args.asset_id}" / "13_walk_forward_backtest"
        report_dir.mkdir(parents=True, exist_ok=True)

        metrics, trades = run_backtest_internal(config)
        save_backtest_report(report_dir, metrics, trades, asset_id=args.asset_id)

        print(
            f"Backtest voltooid: {metrics.get('total_trades', 0)} trades, "
            f"PnL ${metrics.get('total_pnl_usd', 0):.2f} ({metrics.get('total_pnl_pct', 0):.2f}%)"
        )
        return 0
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
