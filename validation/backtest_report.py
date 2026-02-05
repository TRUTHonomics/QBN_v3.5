"""
Backtest Report Generator voor QBN Validation.

Slaat backtest resultaten op als markdown rapport in _validation/ structuur.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


def get_mae_mfe_analysis(trades: List[Any]) -> Dict[str, Any]:
    """
    Bereken MAE/MFE statistieken over closed trades.

    Returns:
        Dict met avg_mae, avg_mfe, mae_mfe_ratio, n_good_entries (MFE > 2*MAE)
    """
    if not trades:
        return {"avg_mae": 0.0, "avg_mfe": 0.0, "mae_mfe_ratio": 0.0, "n_good_entries": 0}
    mae_vals = [getattr(t, "mae_pct", 0) or 0 for t in trades]
    mfe_vals = [getattr(t, "mfe_pct", 0) or 0 for t in trades]
    avg_mae = sum(mae_vals) / len(mae_vals)
    avg_mfe = sum(mfe_vals) / len(mfe_vals)
    n_good = sum(1 for mae, mfe in zip(mae_vals, mfe_vals) if (mfe > 2 * abs(mae) if mae != 0 else mfe > 0))
    ratio = (avg_mfe / abs(avg_mae)) if avg_mae != 0 else 0.0
    return {
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
        "mae_mfe_ratio": ratio,
        "n_good_entries": n_good,
        "n_trades": len(trades),
    }


def save_backtest_report(
    report_dir: Path,
    metrics: Dict[str, Any],
    trades: List[Any],
    asset_id: int = 0,
) -> Path:
    """
    Sla backtest resultaten op als markdown rapport.

    Args:
        report_dir: Directory voor rapport (bijv. _validation/asset_X/13_walk_forward_backtest)
        metrics: Dict van get_metrics() (total_pnl_usd, win_rate_pct, sharpe_ratio, etc.)
        trades: Lijst van Trade objecten (closed_trades)
        asset_id: Optioneel asset ID voor header

    Returns:
        Path naar het gegenereerde rapportbestand
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = report_dir / f"backtest_report_{timestamp}.md"

    lines = [
        "# Walk-Forward Backtest Report",
        "",
        f"**Gegenereerd:** {datetime.now().isoformat()}",
        "",
    ]
    if asset_id:
        lines.append(f"**Asset ID:** {asset_id}")
        lines.append("")
    lines.extend([
        "",
        "## P&L Summary",
        "",
        "```text",
        f"Total PnL USD:     ${metrics.get('total_pnl_usd', 0):,.2f}",
        f"Total PnL %:      {metrics.get('total_pnl_pct', 0):.2f}%",
        f"Final Capital:    ${metrics.get('final_capital_usd', metrics.get('total_pnl_usd', 0)):,.2f}",
        "```",
        "",
        "## Win/Loss Stats",
        "",
        "```text",
        f"Total Trades:      {metrics.get('total_trades', 0)}",
        f"Winning Trades:    {metrics.get('winning_trades', 0)}",
        f"Losing Trades:    {metrics.get('losing_trades', 0)}",
        f"Win Rate %:       {metrics.get('win_rate_pct', 0):.1f}%",
        f"Profit Factor:    {metrics.get('profit_factor', 0):.2f}",
        "```",
        "",
        "## Risk Metrics",
        "",
        "```text",
        f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}",
        f"Sortino Ratio:     {metrics.get('sortino_ratio', 0):.2f}",
        f"Max Drawdown %:   {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"Avg Trade Duration: {metrics.get('avg_trade_duration_hours', 0):.1f}h",
        "```",
        "",
    ])

    if trades:
        mae_mfe = get_mae_mfe_analysis(trades)
        lines.extend([
            "## MAE/MFE Analysis",
            "",
            "```text",
            f"Avg MAE %:         {mae_mfe['avg_mae']:.2f}%",
            f"Avg MFE %:         {mae_mfe['avg_mfe']:.2f}%",
            f"MFE/|MAE| ratio:  {mae_mfe['mae_mfe_ratio']:.2f}",
            f"Good entries (MFE>2*MAE): {mae_mfe['n_good_entries']}/{mae_mfe['n_trades']}",
            "```",
            "",
        ])

    if trades:
        lines.extend([
            "## Trade List (sample)",
            "",
            "```text",
            f"{'Entry':<22} {'Exit':<22} {'Dir':<6} {'PnL USD':>10} {'PnL %':>8} {'Reason':<20}",
            "-" * 95,
        ])
        for t in trades[:50]:
            entry_str = str(t.entry_timestamp)[:19] if t.entry_timestamp else ""
            exit_str = str(t.exit_timestamp)[:19] if t.exit_timestamp else ""
            pnl_usd = getattr(t, "net_pnl_usd", 0)
            pnl_pct = getattr(t, "pnl_pct", 0)
            reason = getattr(t, "exit_reason", "") or ""
            lines.append(f"{entry_str:<22} {exit_str:<22} {t.direction:<6} {pnl_usd:>10.2f} {pnl_pct:>7.2f}% {reason:<20}")
        if len(trades) > 50:
            lines.append(f"... en {len(trades) - 50} meer trades")
        lines.append("```")
        lines.append("")

    content = "\n".join(l for l in lines if l is not None)
    filename.write_text(content, encoding="utf-8")
    logger.info(f"Backtest rapport opgeslagen: {filename}")
    return filename
