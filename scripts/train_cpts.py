
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.qbn_v3_cpt_generator import QBNv3CPTGenerator
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_asset(asset_id=1):
    print(f"\n🚀 Starting FORCED TRAINING for Asset {asset_id}...")
    print("Dit genereert nieuwe CPT's met de gefixte Trade_Hypothesis logica.")
    
    # Init generator
    generator = QBNv3CPTGenerator()
    
    # Generate ALL CPTs (this includes Trade_Hypothesis, Predictions, etc.)
    # We gebruiken lookback_days=3650 (10 jaar) om alle data mee te pakken
    cpts = generator.generate_all_cpts(
        asset_id=asset_id,
        lookback_days=3650,
        save_to_db=True,
        validate_quality=True
    )
    
    print(f"\n✅ Training voltooid! {len(cpts)} CPT's gegenereerd en opgeslagen.")
    print("Controleer nu de validatie rapporten (Stap 9 & 11).")

if __name__ == "__main__":
    train_asset(1)
