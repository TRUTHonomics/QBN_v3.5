#!/usr/bin/env python3
"""
QBN Pipeline Runner - Bridge script to run Alpha Analysis, CPT Generation, and Validation.
Used by the KFL Backend pipeline to automate QBN updates.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_utils import setup_logging
from database.db import get_cursor
from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
from datetime import datetime

logger = setup_logging("qbn_pipeline_runner")

def get_selected_assets():
    """Haal asset IDs op die gemarkeerd zijn voor de huidige run."""
    with get_cursor() as cur:
        cur.execute("SELECT id FROM symbols.symbols WHERE selected_in_current_run = 1 ORDER BY id")
        return [row[0] for row in cur.fetchall()]

def run_threshold_analysis(asset_ids, lookback_days=365):
    """Draait de Threshold Analysis en past resultaten toe op de DB."""
    logger.info(f"📏 Starting Threshold Analysis for {len(asset_ids)} assets (lookback={lookback_days})...")
    script_path = PROJECT_ROOT / 'scripts' / 'run_threshold_analysis.py'
    
    for aid in asset_ids:
        logger.info(f"   🔄 Optimizing Thresholds for Asset {aid}...")
        # REASON: Run met --apply-results om direct op te slaan in DB
        cmd = [
            sys.executable, str(script_path), 
            '--asset-id', str(aid),
            '--methods', 'mi',
            '--targets', 'leading',
            '--horizons', '1h',
            '--lookback-days', str(lookback_days),
            '--apply-results'
        ]
        
        import subprocess
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            logger.error(f"   ❌ Threshold Analysis failed for asset {aid}")
        else:
            logger.info(f"   ✅ Thresholds optimized & saved for asset {aid}")

def _backfill_single_asset(args):
    """
    Helper functie voor parallelle asset backfill.
    
    REASON: ProcessPoolExecutor vereist een top-level pickable function.
    """
    aid, script_path, project_root = args
    cmd = [sys.executable, str(script_path), '--asset-id', str(aid), '--overwrite']
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    return aid, result.returncode, result.stderr


def run_outcome_backfill(asset_ids, max_workers: int = 2):
    """
    Ververst barrier_outcomes op basis van de nieuwste thresholds.
    
    REASON: Parallelle verwerking voor snellere backfill van meerdere assets.
    LET OP: max_workers=2 is optimaal voor GPU (1 actief + 1 prefetching).
    Hogere waarden kunnen GPU contention veroorzaken.
    
    Args:
        asset_ids: Lijst van asset IDs om te verwerken
        max_workers: Aantal parallelle workers (default: 2)
    """
    logger.info(f"🔄 Refreshing Barrier Outcomes for {len(asset_ids)} assets (workers={max_workers})...")
    script_path = PROJECT_ROOT / 'scripts' / 'barrier_backfill.py'
    
    if len(asset_ids) == 1 or max_workers == 1:
        # REASON: Geen overhead voor single asset
        for aid in asset_ids:
            cmd = [sys.executable, str(script_path), '--asset-id', str(aid), '--overwrite']
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                logger.error(f"   ❌ Outcome Backfill failed for asset {aid}")
            else:
                logger.info(f"   ✅ Barrier Outcomes refreshed for asset {aid}")
        return
    
    # REASON: Parallelle verwerking voor meerdere assets
    # ProcessPoolExecutor voor echte parallelisme (geen GIL issues)
    args_list = [(aid, script_path, PROJECT_ROOT) for aid in asset_ids]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_backfill_single_asset, args): args[0] for args in args_list}
        
        for future in as_completed(futures):
            aid = futures[future]
            try:
                aid, returncode, stderr = future.result()
                if returncode != 0:
                    logger.error(f"   ❌ Outcome Backfill failed for asset {aid}: {stderr[:200] if stderr else 'unknown error'}")
                else:
                    logger.info(f"   ✅ Barrier Outcomes refreshed for asset {aid}")
            except Exception as e:
                logger.error(f"   ❌ Outcome Backfill exception for asset {aid}: {e}")

def run_alpha_analysis(asset_ids):
    """Draait de Alpha Analysis voor de opgegeven assets."""
    logger.info(f"🧪 Starting Alpha Analysis for {len(asset_ids)} assets...")
    script_path = PROJECT_ROOT / 'alpha-analysis' / 'analyze_signal_alpha.py'
    
    for aid in asset_ids:
        logger.info(f"   🔄 Analyzing Asset {aid}...")
        cmd = [sys.executable, str(script_path), '--asset', str(aid)]
        import subprocess
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            logger.error(f"   ❌ Alpha Analysis failed for asset {aid}")
        else:
            logger.info(f"   ✅ Alpha Analysis completed for asset {aid}")

def run_cpt_generation(asset_ids, lookback_days=None):
    """Draait CPT Generatie voor de opgegeven assets."""
    logger.info(f"📈 Starting CPT Generation for {len(asset_ids)} assets (lookback={lookback_days})...")
    generator = QBNv3CPTGenerator()
    
    for aid in asset_ids:
        logger.info(f"   🔄 Generating CPTs for Asset {aid}...")
        try:
            generator.generate_all_cpts(aid, lookback_days=lookback_days, save_to_db=True)
            logger.info(f"   ✅ CPTs saved for asset {aid}")
        except Exception as e:
            logger.error(f"   ❌ CPT Generation failed for asset {aid}: {e}")

def run_validation_suite(asset_ids, lookback_days=30):
    """Draait de Validation Suite voor de opgegeven assets."""
    logger.info(f"🧪 Starting Validation Suite for {len(asset_ids)} assets (lookback={lookback_days})...")
    generator = QBNv3CPTGenerator()
    
    for aid in asset_ids:
        logger.info(f"   🔄 Validating Asset {aid}...")
        try:
            generator.validate_existing_cpts(aid, lookback_days=lookback_days)
            logger.info(f"   ✅ Validation metrics updated for asset {aid}")
        except Exception as e:
            logger.error(f"   ❌ Validation failed for asset {aid}: {e}")

def main():
    parser = argparse.ArgumentParser(description='QBN Pipeline Runner')
    parser.add_argument('--tasks', type=str, default='thresh,outcome,alpha,cpt,val', help='Comma-separated tasks')
    parser.add_argument('--assets', type=str, default='selected', help='Comma-separated asset IDs or "selected"')
    parser.add_argument('--lookback', type=int, default=365, help='Lookback days')
    parser.add_argument('--val-lookback', type=int, default=30, help='Lookback days for stability validation')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers for outcome backfill (default: 2)')
    
    args = parser.parse_args()
    
    # ... (asset selectie blijft gelijk) ...
    if args.assets == 'selected':
        asset_ids = get_selected_assets()
    else:
        asset_ids = [int(aid.strip()) for aid in args.assets.split(',')]
        
    if not asset_ids:
        logger.warning("No assets to process.")
        return

    # 2. Voer taken uit
    tasks = [t.strip().lower() for t in args.tasks.split(',')]
    
    # Map legacy letters/numbers to new keys if needed
    task_map = {
        'd': 'alpha',
        'e': 'cpt',
        'h': 'val',
        '8': 'alpha',
        '9': 'cpt',
        '12': 'val',
        'thresh': 'thresh',
        '6': 'thresh',
        'outcome': 'outcome',
        '7': 'outcome' # New menu number for Outcome
    }
    tasks = [task_map.get(t, t) for t in tasks]
    
    if 'thresh' in tasks:
        run_threshold_analysis(asset_ids, lookback_days=args.lookback)

    if 'outcome' in tasks:
        run_outcome_backfill(asset_ids, max_workers=args.workers)

    if 'alpha' in tasks:
        run_alpha_analysis(asset_ids)
        
    if 'cpt' in tasks:
        run_cpt_generation(asset_ids, lookback_days=args.lookback)
        
    if 'val' in tasks:
        run_validation_suite(asset_ids, lookback_days=args.val_lookback)

    logger.info("🏁 QBN Pipeline Runner finished.")

if __name__ == "__main__":
    main()

