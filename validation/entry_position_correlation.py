"""
Entry-Position Correlation Analyse.

Meet correlatie tussen entry prediction quality en position outcome.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _db_has_column(schema: str, table: str, column: str) -> bool:
    """Check via information_schema of een kolom bestaat."""
    try:
        from database.db import get_cursor
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND column_name = %s LIMIT 1
                """,
                (schema, table, column),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def analyze_entry_position_correlation(asset_id: int) -> Dict[str, Any]:
    """
    Meet correlatie tussen entry quality en position outcome.

    Laadt backtest trades (qbn.backtest_trades + backtest_runs) en rapporteert
    win rate per direction, avg PnL per exit reason, etc.

    Returns:
        Dict met o.a. report_lines (lijst strings voor markdown)
    """
    report_lines = []
    try:
        from database.db import get_cursor
        if not _db_has_column("qbn", "backtest_trades", "backtest_id"):
            return {"report_lines": ["Tabel qbn.backtest_trades niet gevonden. Run Walk-Forward Backtest (stap 13) eerst."]}
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT bt.direction, bt.exit_reason, bt.pnl_usd, bt.pnl_pct, bt.quantity
                FROM qbn.backtest_trades bt
                JOIN qbn.backtest_runs br ON bt.backtest_id = br.backtest_id
                WHERE br.asset_id = %s
                ORDER BY bt.entry_time DESC
                LIMIT 500
                """,
                (asset_id,),
            )
            rows = cur.fetchall()
    except Exception as e:
        logger.warning(f"Entry-position correlation query failed: {e}")
        return {"report_lines": [f"Query fout: {e}. Run stap 13 eerst."]}

    if not rows:
        return {"report_lines": ["Geen backtest trades voor dit asset. Run Walk-Forward Backtest (stap 13) eerst."]}

    # Win rate per direction
    by_dir = {}
    for r in rows:
        direction = r[0] or "unknown"
        if direction not in by_dir:
            by_dir[direction] = {"wins": 0, "losses": 0, "pnl_sum": 0.0}
        pnl = float(r[2] or 0)
        by_dir[direction]["pnl_sum"] += pnl
        if pnl > 0.01:
            by_dir[direction]["wins"] += 1
        elif pnl < -0.01:
            by_dir[direction]["losses"] += 1

    report_lines.extend([
        "## Win rate per direction",
        "",
        "```text",
        f"{'Direction':<10} {'Wins':>8} {'Losses':>8} {'Win %':>8} {'Total PnL':>12}",
        "-" * 50,
    ])
    for direction, stats in sorted(by_dir.items()):
        total = stats["wins"] + stats["losses"]
        win_pct = (stats["wins"] / total * 100) if total else 0
        report_lines.append(f"{direction:<10} {stats['wins']:>8} {stats['losses']:>8} {win_pct:>7.1f}% ${stats['pnl_sum']:>10.2f}")
    report_lines.append("```")
    report_lines.append("")

    # Exit reason breakdown
    by_reason = {}
    for r in rows:
        reason = (r[1] or "unknown")[:20]
        if reason not in by_reason:
            by_reason[reason] = {"count": 0, "pnl_sum": 0.0}
        by_reason[reason]["count"] += 1
        by_reason[reason]["pnl_sum"] += float(r[2] or 0)
    report_lines.extend([
        "## Exits per reason",
        "",
        "```text",
        f"{'Reason':<22} {'Count':>8} {'Total PnL':>12}",
        "-" * 45,
    ])
    for reason, stats in sorted(by_reason.items(), key=lambda x: -x[1]["count"]):
        report_lines.append(f"{reason:<22} {stats['count']:>8} ${stats['pnl_sum']:>10.2f}")
    report_lines.append("```")

    return {"report_lines": report_lines, "by_direction": by_dir, "by_reason": by_reason}
