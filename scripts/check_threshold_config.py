#!/usr/bin/env python3
"""
Health Check Script voor qbn.composite_threshold_config tabel.

Controleert of de database correct is geconfigureerd voor DB-driven thresholds.
Dit script kan handmatig worden uitgevoerd of als startup check in de inference loop.

USAGE:
    python scripts/check_threshold_config.py
    python scripts/check_threshold_config.py --asset-id 1
    python scripts/check_threshold_config.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Project root toevoegen aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_utils import setup_logging
from config.threshold_loader import ThresholdLoader

logger = setup_logging("check_threshold_config")


def check_global_availability() -> Dict[str, Any]:
    """Check of threshold_config tabel data bevat."""
    return ThresholdLoader.check_database_availability()


def check_asset_thresholds(asset_id: int, verbose: bool = False) -> Dict[str, Any]:
    """
    Controleer thresholds voor een specifiek asset.
    
    Args:
        asset_id: Asset ID om te controleren
        verbose: Extra logging output
        
    Returns:
        Dict met status per horizon
    """
    horizons = ['1h', '4h', '1d']
    results = {
        'asset_id': asset_id,
        'horizons': {},
        'all_from_db': True,
        'status': 'ok'
    }
    
    for horizon in horizons:
        try:
            loader = ThresholdLoader(asset_id, horizon)
            horizon_result = {
                'neutral_band': loader.composite_neutral_band,
                'strong_threshold': loader.composite_strong_threshold,
                'alignment_high': loader.alignment_high_threshold,
                'alignment_low': loader.alignment_low_threshold,
                'source': loader.source,
                'from_db': loader.is_from_database
            }
            results['horizons'][horizon] = horizon_result
            
            if not loader.is_from_database:
                results['all_from_db'] = False
                
            if verbose:
                logger.info(f"  {horizon}: {horizon_result}")
                
        except Exception as e:
            results['horizons'][horizon] = {
                'error': str(e),
                'from_db': False
            }
            results['all_from_db'] = False
            results['status'] = 'error'
    
    return results


def run_health_check(asset_ids: List[int] = None, verbose: bool = False) -> bool:
    """
    Voer volledige health check uit.
    
    Args:
        asset_ids: Optionele lijst van asset IDs om te checken
        verbose: Extra logging output
        
    Returns:
        True als alle checks slagen, False anders
    """
    logger.info("=" * 60)
    logger.info("QBN Threshold Config Health Check")
    logger.info("=" * 60)
    
    # 1. Global availability check
    logger.info("\n📊 Global Database Check:")
    global_status = check_global_availability()
    
    if not global_status['available']:
        logger.error(f"  ❌ composite_threshold_config tabel is LEEG of niet bereikbaar!")
        if 'error' in global_status:
            logger.error(f"     Error: {global_status['error']}")
        logger.warning("  ⚠️ Fallback defaults worden gebruikt. Run Menu 18 (Threshold Optimalisatie) om thresholds te genereren.")
        return False
    
    logger.info(f"  ✅ threshold_config tabel bevat {global_status['total_entries']} entries")
    
    # 2. Asset-specific checks
    if asset_ids is None:
        # Gebruik default test asset
        asset_ids = [1]
    
    all_ok = True
    for asset_id in asset_ids:
        logger.info(f"\n🔍 Asset {asset_id} Threshold Check:")
        asset_result = check_asset_thresholds(asset_id, verbose)
        
        if asset_result['status'] == 'error':
            logger.error(f"  ❌ Fout bij laden thresholds")
            all_ok = False
            continue
            
        for horizon, data in asset_result['horizons'].items():
            if 'error' in data:
                logger.error(f"  ❌ {horizon}: {data['error']}")
                all_ok = False
            elif data['from_db']:
                logger.info(f"  ✅ {horizon}: DB thresholds geladen (neutral_band={data['neutral_band']:.4f})")
            else:
                logger.warning(f"  ⚠️ {horizon}: Fallback defaults gebruikt (neutral_band={data['neutral_band']:.4f})")
                all_ok = False
    
    # 3. Summary
    logger.info("\n" + "=" * 60)
    if all_ok:
        logger.info("✅ HEALTH CHECK PASSED - Alle thresholds uit database geladen")
    else:
        logger.warning("⚠️ HEALTH CHECK WARNING - Sommige thresholds gebruiken fallback defaults")
        logger.info("   Run Menu 18 (Threshold Optimalisatie) om ontbrekende thresholds te genereren")
    logger.info("=" * 60)
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Health check voor threshold_config tabel')
    parser.add_argument('--asset-id', type=int, nargs='+', help='Asset ID(s) om te controleren')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--exit-on-warning', action='store_true', help='Exit met code 1 bij warnings')
    
    args = parser.parse_args()
    
    success = run_health_check(
        asset_ids=args.asset_id,
        verbose=args.verbose
    )
    
    if args.exit_on_warning and not success:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()

