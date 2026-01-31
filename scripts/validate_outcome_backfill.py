#!/usr/bin/env python3
"""
DEPRECATED: validate_outcome_backfill.py
================================================================================
Dit script is verouderd en wordt vervangen door validate_barrier_data.py.
Legacy point-in-time outcomes worden niet langer ondersteund voor QBN v3.3+.
================================================================================
"""

import sys
# ... rest of imports
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor
from inference.target_generator import HORIZONS
from core.logging_utils import setup_logging

logger = setup_logging("validate_outcome_backfill")

# REASON: Elke horizon heeft zijn eigen time anchor en signal suffix.
# Dit is consistent met outcome_backfill.py HORIZON_CONFIG.
HORIZON_CONFIG = {
    '1h': {
        'time_col': 'time_60',
        'time_close_col': 'time_close_60',
        'expected_rows_factor': 1.0,  # Baseline
    },
    '4h': {
        'time_col': 'time_240',
        'time_close_col': 'time_close_240',
        'expected_rows_factor': 0.25,  # 1/4 van 1h
    },
    '1d': {
        'time_col': 'time_d',
        'time_close_col': 'time_close_d',
        'expected_rows_factor': 0.042,  # ~1/24 van 1h
    }
}


class OutcomeBackfillValidator:
    """
    Validates outcome backfill correctness and detects lookahead bias.
    Uses the normalized qbn.signal_outcomes table.
    """

    def __init__(self):
        self.validation_results = {}

    def _get_asset_id(self, asset_name: str) -> Optional[int]:
        """Convert asset name to asset_id."""
        query = "SELECT id FROM symbols.symbols WHERE UPPER(bybit_symbol) = UPPER(%s) LIMIT 1"
        with get_cursor() as cur:
            cur.execute(query, (asset_name,))
            row = cur.fetchone()
            return row[0] if row else None

    def check_completeness(self, asset_id: int = None) -> Tuple[bool, Dict]:
        """
        Check 1: Completeness
        Verify that eligible rows have entries in qbn.signal_outcomes.
        
        REASON: Elke horizon heeft zijn eigen time anchor:
        - 1h: ~50.000 uur-rijen
        - 4h: ~12.500 4-uurs rijen
        - 1d: ~2.100 dag-rijen
        """
        logger.info("="*80)
        logger.info("CHECK 1: COMPLETENESS (qbn.signal_outcomes)")
        logger.info("="*80)

        results = {}
        all_passed = True
        
        asset_filter = f"AND mtf.asset_id = {asset_id}" if asset_id else ""

        for horizon, horizon_minutes in HORIZONS.items():
            config = HORIZON_CONFIG[horizon]
            time_col = config['time_col']
            time_close_col = config['time_close_col']
            
            # REASON: Gebruik horizon-specifieke time kolommen
            # DISTINCT ON voorkomt dubbeltelling van 1-min rijen
            query = f"""
            WITH distinct_signals AS (
                SELECT DISTINCT ON ({time_col})
                    asset_id, {time_col} as time_anchor, {time_close_col}
                FROM kfl.mtf_signals_lead mtf
                WHERE {time_close_col} < NOW() - INTERVAL '{horizon_minutes} minutes'
                  {asset_filter}
                ORDER BY {time_col}
            )
            SELECT 
                COUNT(*) as total_eligible,
                COUNT(o.outcome_{horizon}) as filled,
                COUNT(*) - COUNT(o.outcome_{horizon}) as missing
            FROM distinct_signals ds
            LEFT JOIN qbn.signal_outcomes o 
                ON o.asset_id = ds.asset_id AND o.time_1 = ds.time_anchor
            """

            try:
                with get_cursor() as cur:
                    cur.execute(query)
                    row = cur.fetchone()

                total_eligible = row[0] or 0
                filled = row[1] or 0
                missing = row[2] or 0

                coverage = (filled / total_eligible * 100) if total_eligible > 0 else 0
                expected_factor = config['expected_rows_factor']

                results[horizon] = {
                    'total_eligible': total_eligible,
                    'filled': filled,
                    'missing': missing,
                    'coverage_pct': coverage,
                    'time_col': time_col
                }

                if coverage < 90.0: # allow some missing data due to gap analysis
                    logger.warning(f"  ⚠️  {horizon} (via {time_col}): Coverage {coverage:.1f}% (missing {missing:,} rows)")
                    all_passed = False
                else:
                    logger.info(f"  ✅ {horizon} (via {time_col}): Coverage {coverage:.1f}% ({filled:,}/{total_eligible:,})")

            except Exception as e:
                logger.error(f"  ❌ {horizon}: Error - {e}")
                all_passed = False

        return all_passed, results

    def check_lookahead_bias(self, asset_id: int = None) -> Tuple[bool, Dict]:
        """
        Check 2: Lookahead Bias Detection (CRITICAL)
        Verify that no outcomes exist for signals where the horizon hasn't passed yet.
        Reference is the horizon-specific time_close column.
        """
        logger.info("\n" + "="*80)
        logger.info("CHECK 2: LOOKAHEAD BIAS DETECTION (CRITICAL)")
        logger.info("="*80)

        results = {}
        no_violations = True
        asset_filter = f"AND so.asset_id = {asset_id}" if asset_id else ""

        for horizon, horizon_minutes in HORIZONS.items():
            config = HORIZON_CONFIG[horizon]
            time_col = config['time_col']
            time_close_col = config['time_close_col']
            
            # REASON: Join op horizon-specifieke time kolom
            query = f"""
            SELECT COUNT(*) 
            FROM qbn.signal_outcomes so
            JOIN kfl.mtf_signals_lead mtf ON so.asset_id = mtf.asset_id AND so.time_1 = mtf.{time_col}
            WHERE mtf.{time_close_col} + INTERVAL '{horizon_minutes} minutes' > NOW()
              AND so.outcome_{horizon} IS NOT NULL
              {asset_filter}
            """

            try:
                with get_cursor() as cur:
                    cur.execute(query)
                    violation_count = cur.fetchone()[0]

                results[horizon] = {'violation_count': violation_count, 'time_col': time_col}

                if violation_count > 0:
                    logger.error(f"  ❌ {horizon} (via {time_col}): LOOKAHEAD BIAS DETECTED - {violation_count} violations!")
                    no_violations = False
                else:
                    logger.info(f"  ✅ {horizon} (via {time_col}): No lookahead bias violations")

            except Exception as e:
                logger.error(f"  ❌ {horizon}: Error - {e}")
                no_violations = False

        return no_violations, results

    def check_distribution(self, asset_id: int = None) -> Tuple[bool, Dict]:
        """Check 3: Distribution of outcome bins."""
        logger.info("\n" + "="*80)
        logger.info("CHECK 3: DISTRIBUTION CHECK")
        logger.info("="*80)

        results = {}
        distribution_ok = True
        asset_filter = f"WHERE asset_id = {asset_id}" if asset_id else ""

        for horizon in HORIZONS.keys():
            query = f"""
            SELECT
                outcome_{horizon} as outcome,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM qbn.signal_outcomes
            {asset_filter}
            GROUP BY outcome
            ORDER BY outcome
            """

            try:
                with get_cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

                distribution = {row[0]: {'count': row[1], 'pct': float(row[2])} for row in rows if row[0] is not None}
                results[horizon] = distribution

                logger.info(f"\n  {horizon} Distribution:")
                for outcome in range(-3, 4):
                    if outcome in distribution:
                        d = distribution[outcome]
                        logger.info(f"    {outcome:>2}: {d['count']:>10,} ({d['pct']:>6.2f}%)")
                    else:
                        logger.warning(f"    {outcome:>2}: Missing data")

            except Exception as e:
                logger.error(f"  ❌ {horizon}: Error - {e}")
                distribution_ok = False

        return distribution_ok, results

    def check_atr_correlation(self, asset_id: int = None) -> Tuple[bool, Dict]:
        """Check 4: Verify that outcomes match ATR-normalized returns."""
        logger.info("\n" + "="*80)
        logger.info("CHECK 4: ATR CORRELATION CHECK")
        logger.info("="*80)

        results = {}
        correlation_ok = True
        asset_filter = f"WHERE asset_id = {asset_id}" if asset_id else ""

        for horizon in HORIZONS.keys():
            # REASON: Join with klines_raw to get entry price for correct normalization
            # REASON: Use horizon-specific ATR (atr_1h, etc.) instead of 1m ATR for correlation check
            query = f"""
            SELECT
                so.outcome_{horizon} as outcome,
                AVG(so.return_{horizon}_pct / (NULLIF(so.atr_{horizon}, 0) / NULLIF(k.close, 0) * 100)) as avg_atr_norm
            FROM qbn.signal_outcomes so
            JOIN kfl.klines_raw k ON k.asset_id = so.asset_id AND k.time = so.time_1 AND k.interval_min = '1'
            {asset_filter.replace('WHERE', 'WHERE so.') if asset_filter else ""}
            GROUP BY outcome
            ORDER BY outcome
            """

            try:
                with get_cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

                logger.info(f"\n  {horizon} ATR Correlation (Expected: -3 < -2.0, +3 > 2.0, 0 ~ 0):")
                for outcome, avg_norm in rows:
                    if outcome is None: continue
                    logger.info(f"    {outcome:>2}: Avg Norm Return = {avg_norm:>6.2f}")
                    
                    # Basic sanity checks
                    if outcome == 3 and avg_norm < 1.5: correlation_ok = False
                    if outcome == -3 and avg_norm > -1.5: correlation_ok = False
                    if outcome == 0 and abs(avg_norm) > 0.5: correlation_ok = False

            except Exception as e:
                logger.error(f"  ❌ {horizon}: Error - {e}")
                correlation_ok = False

        return correlation_ok, results

    def run_all_checks(self, asset_name: str = None, asset_id: int = None) -> bool:
        if asset_name and not asset_id:
            asset_id = self._get_asset_id(asset_name)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OUTCOME BACKFILL VALIDATION REPORT")
        logger.info(f"Target Table: qbn.signal_outcomes")
        if asset_id: 
            logger.info(f"Asset ID: {asset_id}")
        elif asset_name:
            logger.info(f"Asset: {asset_name} (ID: {asset_id})")
        else:
            logger.info(f"Scope: ALL ASSETS")
        logger.info(f"Generated: {datetime.now()}")
        logger.info(f"{'='*80}\n")

        c_ok, _ = self.check_completeness(asset_id)
        l_ok, _ = self.check_lookahead_bias(asset_id)
        d_ok, _ = self.check_distribution(asset_id)
        a_ok, _ = self.check_atr_correlation(asset_id)

        all_passed = c_ok and l_ok and d_ok and a_ok
        
        logger.info("\n" + "="*80)
        if all_passed:
            logger.info("✅ ALL VALIDATIONS PASSED")
        else:
            logger.error("❌ VALIDATION FAILURES DETECTED")
        logger.info("="*80)
        
        return all_passed


def main():
    import time
    print("\n" + "!"*80)
    print("! DEPRECATED: Dit script (validate_outcome_backfill.py) is verouderd.")
    print("! Gebruik scripts/validate_barrier_data.py voor QBN v3.3+.")
    print("!"*80 + "\n")
    time.sleep(2)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', help='Asset symbol (e.g. BTCUSDT)')
    parser.add_argument('--asset-id', type=int, help='Asset ID')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--selected', action='store_true', help='Validate selected assets')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    validator = OutcomeBackfillValidator()
    
    if args.selected:
        from database.db import get_cursor
        with get_cursor() as cur:
            cur.execute("SELECT id FROM symbols.symbols WHERE selected_in_current_run = 1 ORDER BY id")
            asset_ids = [row[0] for row in cur.fetchall()]
        
        if not asset_ids:
            logger.error("No selected assets found.")
            sys.exit(1)
            
        logger.info(f"Validating {len(asset_ids)} selected assets...")
        all_success = True
        for aid in asset_ids:
            if not validator.run_all_checks(asset_id=aid):
                all_success = False
        sys.exit(0 if all_success else 1)
    else:
        success = validator.run_all_checks(asset_name=args.asset, asset_id=args.asset_id)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
