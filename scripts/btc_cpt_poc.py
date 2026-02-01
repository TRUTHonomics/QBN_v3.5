"""
BTC Proof of Concept - CPT Generation.
Runs the new QBN v2 CPT Generator for asset_id 1 (BTC).
"""

import logging
import sys
import os
from pathlib import Path

# Voeg root directory toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_utils import setup_logging
from inference.qbn_v2_cpt_generator import QBNv2CPTGenerator

logger = setup_logging("btc_cpt_poc")

def run_btc_poc():
    asset_id = 1 # BTC
    logger.info(f"Starting CPT generation POC for asset {asset_id} (BTC)")
    
    generator = QBNv2CPTGenerator()
    
    try:
        # Genereer alle CPT's voor BTC (met lookback van 30 dagen voor de test)
        cpts = generator.generate_all_cpts(asset_id, lookback_days=30, save_to_db=True)
        
        logger.info(f"Successfully generated {len(cpts)} CPTs for BTC.")
        for node, data in cpts.items():
            obs = data.get('observations', 0)
            logger.info(f" - {node}: {obs} observations")
            
    except Exception as e:
        logger.error(f"POC failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_btc_poc()

