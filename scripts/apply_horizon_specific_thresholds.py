"""
Script om horizon-specifieke Leading Composite thresholds toe te passen.

Huidige situatie: neutral_band=0.05 en strong_threshold=0.15 voor ALLE horizons (1h, 4h, 1d).
Doel: Differentieer thresholds per horizon zodat langere horizons bredere filters hebben.

Aanbevolen configuratie:
- 1h: nb=0.03, st=0.12 (snelle reactie, kortere timeframe)
- 4h: nb=0.05, st=0.15 (huidige baseline)
- 1d: nb=0.07, st=0.18 (bredere filter voor langere timeframe)

Usage:
    docker exec QBN_v4_Dagster_Webserver python /app/scripts/apply_horizon_specific_thresholds.py --asset-id 1 --apply
"""

import argparse
import logging
import sys
from datetime import datetime

sys.path.insert(0, '/app')

try:
    from core.logging_utils import setup_logging
    setup_logging('horizon_thresholds')
except ModuleNotFoundError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from database.db import get_cursor


def parse_args():
    parser = argparse.ArgumentParser(description='Apply horizon-specific Leading Composite thresholds')
    parser.add_argument('--asset-id', type=int, default=1)
    parser.add_argument('--apply', action='store_true', help='Apply changes to database (default: dry-run)')
    parser.add_argument('--run-id', type=str, default=None, help='Run ID (auto-generated if None)')
    return parser.parse_args()


# Horizon-specifieke configuratie
HORIZON_CONFIGS = {
    '1h': {
        'neutral_band': 0.03,
        'strong_threshold': 0.12,
        'bearish_neutral_band': 0.03,
        'bullish_neutral_band': 0.03,
        'bearish_strong_threshold': 0.12,
        'bullish_strong_threshold': 0.12,
        'rationale': 'Snelle reactie voor korte timeframe'
    },
    '4h': {
        'neutral_band': 0.05,
        'strong_threshold': 0.15,
        'bearish_neutral_band': 0.05,
        'bullish_neutral_band': 0.05,
        'bearish_strong_threshold': 0.15,
        'bullish_strong_threshold': 0.15,
        'rationale': 'Baseline (huidige productie config)'
    },
    '1d': {
        'neutral_band': 0.07,
        'strong_threshold': 0.18,
        'bearish_neutral_band': 0.07,
        'bullish_neutral_band': 0.07,
        'bearish_strong_threshold': 0.18,
        'bullish_strong_threshold': 0.18,
        'rationale': 'Bredere filter voor lange timeframe'
    },
}


def get_current_config(asset_id: int):
    """Haal huidige threshold config op."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT horizon, config_type, param_name, param_value, source_method
            FROM qbn.composite_threshold_config
            WHERE asset_id = %s AND config_type = 'leading_composite'
            ORDER BY horizon, param_name
        """, (asset_id,))
        
        rows = cur.fetchall()
        
        config = {}
        for horizon in ['1h', '4h', '1d']:
            config[horizon] = {}
        
        for row in rows:
            horizon, config_type, param_name, param_value, source_method = row
            if horizon not in config:
                config[horizon] = {}
            config[horizon][param_name] = {
                'value': float(param_value),
                'method': source_method
            }
        
        return config


def apply_horizon_config(asset_id: int, run_id: str, dry_run: bool = True):
    """Apply horizon-specifieke thresholds naar database."""
    
    logger.info("="*80)
    logger.info("Horizon-Specific Leading Composite Thresholds")
    logger.info("="*80)
    logger.info(f"Asset ID: {asset_id}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'APPLY TO DATABASE'}")
    
    # Haal huidige config op
    logger.info("\n1. CURRENT CONFIGURATION")
    current = get_current_config(asset_id)
    
    for horizon in ['1h', '4h', '1d']:
        logger.info(f"\n   {horizon}:")
        if horizon in current and current[horizon]:
            for param, info in sorted(current[horizon].items()):
                logger.info(f"      {param}: {info['value']} ({info['method']})")
        else:
            logger.info("      (no config found)")
    
    # Toon nieuwe config
    logger.info("\n2. NEW CONFIGURATION")
    for horizon, config in HORIZON_CONFIGS.items():
        logger.info(f"\n   {horizon}: ({config['rationale']})")
        logger.info(f"      neutral_band: {config['neutral_band']} (was: {current.get(horizon, {}).get('neutral_band', {}).get('value', 'N/A')})")
        logger.info(f"      strong_threshold: {config['strong_threshold']} (was: {current.get(horizon, {}).get('strong_threshold', {}).get('value', 'N/A')})")
    
    # Apply changes
    if not dry_run:
        logger.info("\n3. APPLYING CHANGES")
        
        with get_cursor() as cur:
            for horizon, config in HORIZON_CONFIGS.items():
                for param_name, param_value in config.items():
                    if param_name == 'rationale':
                        continue
                    
                    # Upsert
                    cur.execute("""
                        INSERT INTO qbn.composite_threshold_config 
                            (asset_id, horizon, config_type, param_name, param_value, source_method, run_id, created_at, updated_at)
                        VALUES 
                            (%s, %s, 'leading_composite', %s, %s, 'horizon-specific tuning', %s, NOW(), NOW())
                        ON CONFLICT (asset_id, horizon, config_type, param_name, run_id)
                        DO UPDATE SET
                            param_value = EXCLUDED.param_value,
                            source_method = EXCLUDED.source_method,
                            updated_at = NOW()
                    """, (asset_id, horizon, param_name, param_value, run_id))
                    
                    logger.info(f"   ✓ {horizon} {param_name} = {param_value}")
        
        logger.info("\n✅ Changes applied to database")
        logger.info("   Next steps:")
        logger.info("   1. Re-generate CPTs: docker exec QBN_v4_Dagster_Webserver python -m scripts.qbn_pipeline_runner --asset-id 1")
        logger.info("   2. Run backtest to measure impact")
    else:
        logger.info("\n⚠️  DRY RUN MODE - No changes applied")
        logger.info("   Use --apply flag to apply changes to database")
    
    logger.info("\n" + "="*80)
    return 0


def main():
    args = parse_args()
    
    # Generate run_id if not provided
    run_id = args.run_id or datetime.now().strftime('%Y%m%d-%H%M%S') + '-horizon_thresholds'
    
    return apply_horizon_config(args.asset_id, run_id, dry_run=not args.apply)


if __name__ == '__main__':
    sys.exit(main())
