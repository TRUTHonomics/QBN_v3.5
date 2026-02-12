"""
Onderzoek Position Delta MI=0 probleem.

Analyseert waarom Coincident en Confirming delta thresholds MI=0 hebben.

Mogelijke oorzaken:
1. Te weinig event windows (57→32 in recente runs)
2. Verkeerde threshold range in grid search  
3. Delta's zijn daadwerkelijk niet informatief

Usage:
    docker exec QBN_v4_Dagster_Webserver python /app/scripts/investigate_delta_mi.py --asset-id 1
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/app')

try:
    from core.logging_utils import setup_logging
    setup_logging('investigate_delta_mi')
except ModuleNotFoundError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from database.db import get_cursor
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score


def parse_args():
    parser = argparse.ArgumentParser(description='Investigate Position Delta MI=0')
    parser.add_argument('--asset-id', type=int, default=1)
    parser.add_argument('--run-id', type=str, default=None, help='Specific run_id, or None for latest')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Position Delta MI Investigation")
    logger.info("="*80)
    
    # 1. Check event window count
    logger.info("\n1. EVENT WINDOW VOLUME CHECK")
    with get_cursor() as cur:
        if args.run_id:
            cur.execute("""
                SELECT count(*) as n_events,
                       min(duration_minutes) as min_dur,
                       avg(duration_minutes) as avg_dur,
                       max(duration_minutes) as max_dur,
                       sum(case when duration_minutes < 120 then 1 else 0 end) as short_events
                FROM qbn.event_windows
                WHERE asset_id = %s AND run_id = %s
            """, (args.asset_id, args.run_id))
        else:
            cur.execute("""
                SELECT count(*) as n_events,
                       min(duration_minutes) as min_dur,
                       avg(duration_minutes) as avg_dur,
                       max(duration_minutes) as max_dur,
                       sum(case when duration_minutes < 120 then 1 else 0 end) as short_events
                FROM qbn.event_windows
                WHERE asset_id = %s
                ORDER BY generated_at DESC
                LIMIT 100
            """, (args.asset_id,))
        
        row = cur.fetchone()
        if row and row[0] > 0:
            logger.info(f"   Events: {row[0]}")
            logger.info(f"   Duration: min={row[1]:.0f}m, avg={row[2]:.0f}m, max={row[3]:.0f}m")
            logger.info(f"   Short events (<2h): {row[4]} ({100*row[4]/row[0]:.1f}%)")
            
            if row[0] < 50:
                logger.warning(f"   ⚠️  ISSUE: Only {row[0]} events - insufficient for robust MI estimation (recommend ≥100)")
        else:
            logger.error("   ❌ No event windows found!")
            return 1
    
    # 2. Check delta threshold config MI scores
    logger.info("\n2. DELTA THRESHOLD CONFIG MI SCORES")
    with get_cursor() as cur:
        cur.execute("""
            SELECT delta_type, score_type, threshold, mi_score, distribution, source_method
            FROM qbn.position_delta_threshold_config
            WHERE asset_id = %s
            ORDER BY score_type, delta_type
        """, (args.asset_id,))
        
        rows = cur.fetchall()
        if not rows:
            logger.error("   ❌ No position delta threshold config found!")
            return 1
        
        logger.info(f"   {'Score Type':<12} {'Delta Type':<12} {'Threshold':<10} {'MI Score':<10} {'Method':<20}")
        logger.info("   " + "-" * 75)
        
        for r in rows:
            logger.info(f"   {r[1]:<12} {r[0]:<12} {float(r[2]):<10.3f} {float(r[3]):<10.6f} {r[5]:<20}")
            dist = r[4] if isinstance(r[4], dict) else {}
            logger.info(f"      Distribution: {dist}")
            
            if float(r[3]) == 0.0:
                logger.warning(f"      ⚠️  MI=0 for {r[1]} {r[0]} - no predictive value!")
    
    # 3. Fetch barrier outcomes binnen event windows en bereken delta's
    logger.info("\n3. DELTA VALUES ANALYSIS")
    with get_cursor() as cur:
        # Haal barrier_outcomes op die binnen een event window vallen
        cur.execute("""
            SELECT 
                bo.asset_id,
                bo.time_1,
                bo.event_id,
                bo.coincident_score,
                bo.confirming_score,
                bo.first_significant_barrier,
                ew.entry_coincident_score,
                ew.entry_confirming_score,
                bo.time_since_entry_minutes
            FROM qbn.barrier_outcomes bo
            JOIN qbn.event_windows ew ON bo.event_id = ew.event_id
            WHERE bo.asset_id = %s 
              AND bo.event_id IS NOT NULL
            ORDER BY bo.event_id, bo.time_since_entry_minutes
            LIMIT 10000
        """, (args.asset_id,))
        
        rows = cur.fetchall()
        
        if not rows:
            logger.error("   ❌ No barrier_outcomes with event_id found!")
            return 1
        
        logger.info(f"   Loaded {len(rows)} rows with event_id")
        
        # Bereken deltas
        df = pd.DataFrame(rows, columns=['asset_id', 'time_1', 'event_id', 'coincident_score', 
                                          'confirming_score', 'barrier', 'entry_coincident', 
                                          'entry_confirming', 'time_since_entry'])
        
        df['delta_coincident'] = df['coincident_score'] - df['entry_coincident']
        df['delta_confirming'] = df['confirming_score'] - df['entry_confirming']
        
        # Discretize deltas (0.03 threshold zoals in config)
        def discretize_delta(val, threshold=0.03):
            if val < -threshold:
                return 'deteriorating'
            elif val > threshold:
                return 'improving'
            else:
                return 'stable'
        
        df['delta_coincident_disc'] = df['delta_coincident'].apply(lambda x: discretize_delta(x, 0.03))
        df['delta_confirming_disc'] = df['delta_confirming'].apply(lambda x: discretize_delta(x, 0.03))
        
        # Map barrier to binary outcome (up vs down)
        def barrier_to_outcome(b):
            if b is None or b == 'none':
                return 'timeout'
            elif 'up_' in b:
                return 'up'
            elif 'down_' in b:
                return 'down'
            else:
                return 'other'
        
        df['outcome'] = df['barrier'].apply(barrier_to_outcome)
        
        # Bereken MI scores
        logger.info("\n   Delta Statistics:")
        logger.info(f"   Coincident: mean={df['delta_coincident'].mean():.4f}, std={df['delta_coincident'].std():.4f}, min={df['delta_coincident'].min():.4f}, max={df['delta_coincident'].max():.4f}")
        logger.info(f"   Confirming: mean={df['delta_confirming'].mean():.4f}, std={df['delta_confirming'].std():.4f}, min={df['delta_confirming'].min():.4f}, max={df['delta_confirming'].max():.4f}")
        
        logger.info("\n   Discretized Delta Distributions:")
        logger.info(f"   Coincident: {dict(df['delta_coincident_disc'].value_counts())}")
        logger.info(f"   Confirming: {dict(df['delta_confirming_disc'].value_counts())}")
        
        logger.info(f"\n   Outcome Distribution: {dict(df['outcome'].value_counts())}")
        
        # Bereken MI
        if len(df['delta_coincident_disc'].unique()) > 1 and len(df['outcome'].unique()) > 1:
            mi_coincident = mutual_info_score(df['delta_coincident_disc'], df['outcome'])
            logger.info(f"\n   ✅ MI(delta_coincident_disc, outcome) = {mi_coincident:.6f}")
        else:
            logger.warning("\n   ⚠️  Insufficient variation in delta_coincident or outcome for MI calculation")
            mi_coincident = 0.0
        
        if len(df['delta_confirming_disc'].unique()) > 1 and len(df['outcome'].unique()) > 1:
            mi_confirming = mutual_info_score(df['delta_confirming_disc'], df['outcome'])
            logger.info(f"   ✅ MI(delta_confirming_disc, outcome) = {mi_confirming:.6f}")
        else:
            logger.warning("   ⚠️  Insufficient variation in delta_confirming or outcome for MI calculation")
            mi_confirming = 0.0
    
    # 4. Diagnose
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSIS")
    logger.info("="*80)
    
    issues = []
    
    if row[0] < 50:
        issues.append(f"LOW EVENT COUNT: {row[0]} events (recommend ≥100)")
    
    if mi_coincident == 0.0:
        issues.append("COINCIDENT MI=0: No mutual information with outcomes")
    
    if mi_confirming == 0.0:
        issues.append("CONFIRMING MI=0: No mutual information with outcomes")
    
    # Check if deltas zijn te klein
    if df['delta_coincident'].std() < 0.05:
        issues.append(f"LOW COINCIDENT VARIATION: std={df['delta_coincident'].std():.4f} (deltas veranderen weinig)")
    
    if df['delta_confirming'].std() < 0.05:
        issues.append(f"LOW CONFIRMING VARIATION: std={df['delta_confirming'].std():.4f} (deltas veranderen weinig)")
    
    # Check if threshold te breed is
    stable_pct_coin = 100 * (df['delta_coincident_disc'] == 'stable').sum() / len(df)
    stable_pct_conf = 100 * (df['delta_confirming_disc'] == 'stable').sum() / len(df)
    
    if stable_pct_coin > 70:
        issues.append(f"COINCIDENT THRESHOLD TOO WIDE: {stable_pct_coin:.1f}% stable (threshold=0.03 is te breed)")
    
    if stable_pct_conf > 70:
        issues.append(f"CONFIRMING THRESHOLD TOO WIDE: {stable_pct_conf:.1f}% stable (threshold=0.03 is te breed)")
    
    if not issues:
        logger.info("✅ No obvious issues detected")
    else:
        logger.warning(f"❌ {len(issues)} issues detected:")
        for i, issue in enumerate(issues, 1):
            logger.warning(f"   {i}. {issue}")
    
    # 5. Recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)
    
    if row[0] < 100:
        logger.info("1. INCREASE EVENT WINDOW DATA")
        logger.info("   - Run full pipeline with longer lookback (365d instead of 90d)")
        logger.info("   - Or lower event detection threshold to capture more events")
    
    if stable_pct_coin > 70 or stable_pct_conf > 70:
        logger.info("2. ADJUST DELTA THRESHOLDS")
        logger.info(f"   - Current: 0.03 (±3%)")
        logger.info(f"   - Recommend: 0.01-0.02 (±1-2%) to increase sensitivity")
        logger.info("   - Run: scripts/run_position_delta_threshold_analysis.py with --threshold-range 0.005 0.05")
    
    if mi_coincident < 0.01 and mi_confirming < 0.01:
        logger.info("3. CONSIDER ALTERNATIVE FEATURES")
        logger.info("   - Delta scores lijken niet informatief voor dit asset")
        logger.info("   - Overweeg absolute scores i.p.v. deltas")
        logger.info("   - Of gebruik andere position management features (volume, ATR ratio, time-based)")
    
    # Save results
    output_dir = Path('_validation/delta_mi_investigation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'investigation_{timestamp}.json'
    
    output_data = {
        'timestamp': timestamp,
        'asset_id': args.asset_id,
        'event_count': int(row[0]),
        'mi_coincident': float(mi_coincident),
        'mi_confirming': float(mi_confirming),
        'delta_stats': {
            'coincident': {
                'mean': float(df['delta_coincident'].mean()),
                'std': float(df['delta_coincident'].std()),
                'distribution': dict(df['delta_coincident_disc'].value_counts())
            },
            'confirming': {
                'mean': float(df['delta_confirming'].mean()),
                'std': float(df['delta_confirming'].std()),
                'distribution': dict(df['delta_confirming_disc'].value_counts())
            }
        },
        'issues': issues
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("\nInvestigation complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
