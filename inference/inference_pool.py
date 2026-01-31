"""
Inference Pool voor QBN v2.
Preloads inference engines voor alle tracked assets bij startup om latency te minimaliseren.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .trade_aligned_inference import TradeAlignedInference
from .inference_loader import InferenceLoader

logger = logging.getLogger(__name__)

class InferencePool:
    """
    Pool van pre-loaded inference engines.
    
    Elimineert de 'cold-start' latency (~200-500ms voor CPT load) 
    door alle tracked assets bij startup in memory te laden.
    """
    
    def __init__(self, tracked_assets: Optional[List[int]] = None):
        """
        Args:
            tracked_assets: Lijst van asset IDs om te preloade. 
                           Indien None, worden ze later handmatig of via preload_all toegevoegd.
        """
        self.engines: Dict[int, TradeAlignedInference] = {}
        self.loader = InferenceLoader()
        self.tracked_assets = tracked_assets or []
        
    def preload_all(self, max_workers: int = 4):
        """
        Laad alle tracked assets parallel in de pool.
        """
        if not self.tracked_assets:
            logger.warning("Geen tracked assets geconfigureerd voor preloading.")
            return

        start_time = time.time()
        logger.info(f"Start preloading {len(self.tracked_assets)} inference engines...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_asset = {
                executor.submit(self.loader.load_inference_engine, aid): aid 
                for aid in self.tracked_assets
            }
            
            # Verwerk resultaten
            for future in as_completed(future_to_asset):
                aid = future_to_asset[future]
                try:
                    engine = future.result()
                    self.engines[aid] = engine
                except Exception as e:
                    logger.error(f"Fout bij preloading engine voor asset {aid}: {e}")

        elapsed = time.time() - start_time
        logger.info(f"Preloading voltooid in {elapsed:.2f}s ({len(self.engines)} assets geladen).")

    def get_engine(self, asset_id: int) -> TradeAlignedInference:
        """
        Haal engine op voor asset. Indien niet in pool, laad 'on-the-fly' (lazy load).
        """
        if asset_id not in self.engines:
            logger.info(f"Engine voor asset {asset_id} niet in pool, lazy loading...")
            self.engines[asset_id] = self.loader.load_inference_engine(asset_id)
            
        return self.engines[asset_id]

    def refresh_engine(self, asset_id: int):
        """
        Herlaad de engine voor een specifiek asset (bv na nieuwe CPT training).
        """
        logger.info(f"Refreshing inference engine voor asset {asset_id}...")
        try:
            self.engines[asset_id] = self.loader.load_inference_engine(asset_id)
        except Exception as e:
            logger.error(f"Fout bij refreshen engine voor asset {asset_id}: {e}")

    def get_pool_info(self) -> Dict[str, Any]:
        """Retourneert status informatie over de pool."""
        return {
            'loaded_assets': list(self.engines.keys()),
            'count': len(self.engines),
            'tracked_count': len(self.tracked_assets)
        }

def initialize_pool(asset_ids: Optional[List[int]] = None) -> InferencePool:
    """Factory functie om pool te initialiseren."""
    pool = InferencePool(asset_ids)
    if asset_ids:
        pool.preload_all()
    return pool

