#!/usr/bin/env python3
"""validate_barrier_data.py - Valideer barrier outcomes integriteit"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor
from core.logging_utils import setup_logging

logger = setup_logging("validate_barrier_data")

INTEGRITY_CHECKS = [
    {
        'name': 'barrier_time_ordering',
        'query': """
            SELECT COUNT(*) FROM qbn.barrier_outcomes
            WHERE (time_to_up_025_atr > time_to_up_050_atr 
                   AND time_to_up_025_atr IS NOT NULL 
                   AND time_to_up_050_atr IS NOT NULL)
               OR (time_to_up_050_atr > time_to_up_075_atr 
                   AND time_to_up_050_atr IS NOT NULL 
                   AND time_to_up_075_atr IS NOT NULL)
               OR (time_to_down_025_atr > time_to_down_050_atr 
                   AND time_to_down_025_atr IS NOT NULL 
                   AND time_to_down_050_atr IS NOT NULL)
        """,
        'expected': 0,
        'severity': 'error'
    },
    {
        'name': 'max_atr_consistency',
        'query': """
            SELECT COUNT(*) FROM qbn.barrier_outcomes
            WHERE (time_to_up_100_atr IS NOT NULL AND max_up_atr < 1.0)
               OR (time_to_down_100_atr IS NOT NULL AND max_down_atr > -1.0)
        """,
        'expected': 0,
        'severity': 'error'
    },
    {
        'name': 'first_significant_consistency',
        'query': """
            SELECT COUNT(*) FROM qbn.barrier_outcomes
            WHERE (first_significant_barrier = 'up_075' AND time_to_up_075_atr IS NULL)
               OR (first_significant_barrier = 'down_075' AND time_to_down_075_atr IS NULL)
        """,
        'expected': 0,
        'severity': 'error'
    },
    {
        'name': 'null_atr_check',
        'query': """
            SELECT COUNT(*) FROM qbn.barrier_outcomes
            WHERE atr_at_signal IS NULL OR atr_at_signal <= 0
        """,
        'expected': 0,
        'severity': 'error'
    },
    {
        'name': 'completeness_check',
        'query': """
            SELECT 
                COUNT(*) FILTER (WHERE first_significant_barrier IS NULL) as missing_sig,
                COUNT(*) as total
            FROM qbn.barrier_outcomes
        """,
        'check': lambda r: r[0] / r[1] < 0.01 if r[1] > 0 else True,  # Max 1% missing
        'severity': 'warning'
    }
]

def run_validation(asset_id: int = None):
    """Run alle integriteitscontroles."""
    results = []
    
    logger.info(f"🚀 Start validatie van barrier data{' voor asset ' + str(asset_id) if asset_id else ' over alle assets'}...")
    
    # 1. Integriteitscontroles
    for check in INTEGRITY_CHECKS:
        with get_cursor() as cur:
            query = check['query']
            if asset_id:
                if 'WHERE' in query:
                    query = query.replace('WHERE', f'WHERE asset_id = {asset_id} AND')
                else:
                    query = query.replace('FROM qbn.barrier_outcomes', f'FROM qbn.barrier_outcomes WHERE asset_id = {asset_id}')
            
            cur.execute(query)
            row = cur.fetchone()
            result = row[0] if len(row) == 1 else row
        
        if 'expected' in check:
            passed = result == check['expected']
        else:
            passed = check['check'](result)
        
        status = 'PASS' if passed else check['severity'].upper()
        results.append({
            'check': check['name'],
            'result': result,
            'status': status
        })
        
        log_msg = f"{status}: {check['name']} = {result}"
        if not passed:
            if check['severity'] == 'error':
                logger.error(log_msg)
            else:
                logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    # 2. Toon distributie rapport
    show_distribution(asset_id)
    
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    warnings = sum(1 for r in results if r['status'] == 'WARNING')
    
    return {
        'checks': results,
        'errors': errors,
        'warnings': warnings,
        'passed': errors == 0
    }

def show_distribution(asset_id: int = None):
    """Toon distributie van barrier outcomes."""
    query = """
        SELECT 
            first_significant_barrier,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM qbn.barrier_outcomes
    """
    if asset_id:
        query += f" WHERE asset_id = {asset_id}"
    query += " GROUP BY first_significant_barrier ORDER BY count DESC"
    
    print("\n" + "="*60)
    print(f"📊 BARRIER DISTRIBUTION {'(Asset '+str(asset_id)+')' if asset_id else '(All Assets)'}")
    print("="*60)
    
    with get_cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        
        if not rows:
            print("Geen data gevonden.")
            return
            
        print(f"{'Barrier':<25} {'Count':>12} {'Percentage':>12}")
        print("-" * 60)
        for name, count, pct in rows:
            print(f"{name:<25} {count:>12,} {pct:>11.2f}%")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-id', type=int, default=None)
    args = parser.parse_args()
    
    result = run_validation(args.asset_id)
    print(f"\nSummary for {'Asset ' + str(args.asset_id) if args.asset_id else 'All Assets'}:")
    print(f"Validation: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"Errors: {result['errors']}, Warnings: {result['warnings']}")
    
    if not result['passed']:
        sys.exit(1)
