"""
BTC Proof of Concept - CPT Generation.
Runs the new QBN v2 CPT Generator for asset_id 1 (BTC).
"""

import logging
import sys
import os
import shutil
from datetime import datetime

# Voeg root directory toe aan path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from inference.qbn_v2_cpt_generator import QBNv2CPTGenerator

# REASON: Volg logregels voor BTC POC
def setup_logging():
    log_dir = os.path.join(PROJECT_ROOT, '_log')
    archive_dir = os.path.join(log_dir, 'archive')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    script_name = "btc_cpt_poc"
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    
    # Archiveer oude logs
    import glob
    for old_log in glob.glob(os.path.join(log_dir, f"{script_name}_*.log")):
        try:
            shutil.move(old_log, os.path.join(archive_dir, os.path.basename(old_log)))
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True
    )
    l = logging.getLogger(__name__)
    l.info(f"🚀 New {script_name} run started. Logging to: {log_file}")
    return l

logger = setup_logging()

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

