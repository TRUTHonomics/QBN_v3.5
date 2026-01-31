"""
Real-time Inference Loop Service voor QBN v2.
Luistert naar PostgreSQL NOTIFY events en triggert inference via de InferencePool.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Dict, Set, Optional, List
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
from asyncpg import Pool

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.inference_pool import InferencePool
from inference.inference_loader import InferenceLoader
from services.prediction_writer import PredictionWriter

logger = logging.getLogger(__name__)

class InferenceLoopService:
    """
    Asynchrone service die luistert naar signal updates en inference uitvoert.
    """
    
    def __init__(self, asset_ids: List[int], db_config: Dict):
        self.asset_ids = set(asset_ids)
        self.db_config = db_config
        self.pool: Optional[Pool] = None
        self.running = False
        
        # Componenten
        self.inference_pool = InferencePool(asset_ids)
        self.loader = InferenceLoader()
        self.writer = PredictionWriter()
        
        # Queuing & Locking
        self.queue = asyncio.Queue()
        self.locks: Dict[int, asyncio.Lock] = {aid: asyncio.Lock() for aid in asset_ids}
        self.processing_tasks: Set[int] = set()
        
        # Heartbeat voor watchdog
        self.last_heartbeat = None

    async def start(self):
        """Start de service."""
        logger.info(f"🚀 Start Inference Loop Service voor {len(self.asset_ids)} assets...")
        
        # 1. Preload engines (parallel)
        self.inference_pool.preload_all(max_workers=4)
        
        # 2. Database connection pool
        self.pool = await asyncpg.create_pool(**self.db_config)
        self.writer.set_pool(self.pool)
        
        self.running = True
        self.last_heartbeat = datetime.now(timezone.utc)
        
        # 3. Start workers
        await asyncio.gather(
            self._listen_for_notifications(),
            self._process_queue()
        )

    async def stop(self):
        """Stop de service gracefully."""
        logger.info("🛑 Stoppen van Inference Loop Service...")
        self.running = False
        if self.pool:
            await self.pool.close()

    async def _listen_for_notifications(self):
        """Luister naar NOTIFY events van PostgreSQL."""
        channel = 'signals_lead_updated'
        
        async def on_notify(connection, pid, channel, payload):
            try:
                data = json.loads(payload)
                asset_id = data.get('asset_id')
                interval = data.get('interval_min')
                
                if asset_id in self.asset_ids:
                    # We voegen het asset toe aan de queue voor verwerking
                    await self.queue.put((asset_id, data))
                    logger.debug(f"🔔 Notify ontvangen voor Asset {asset_id} (interval: {interval})")
            except Exception as e:
                logger.error(f"❌ Fout bij verwerken notify: {e}")

        async with self.pool.acquire() as conn:
            await conn.add_listener(channel, on_notify)
            logger.info(f"🎧 Luisteren op kanaal: {channel}")
            
            while self.running:
                self.last_heartbeat = datetime.now(timezone.utc)
                await asyncio.sleep(1)

    async def _process_queue(self):
        """Verwerk inkomende taken uit de queue."""
        logger.info("👷 Worker gestart voor queue verwerking")
        
        while self.running:
            try:
                # Wacht op item (met timeout voor heartbeat)
                try:
                    asset_id, data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check of we al bezig zijn met dit asset
                if asset_id in self.processing_tasks:
                    self.queue.task_done()
                    continue
                
                # Start asynchrone verwerking voor dit asset
                asyncio.create_task(self._safe_infer(asset_id))
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Fout in queue processor: {e}")
                await asyncio.sleep(1)

    async def _safe_infer(self, asset_id: int):
        """Voert inference uit met locking per asset."""
        if asset_id not in self.locks:
            self.locks[asset_id] = asyncio.Lock()
            
        lock = self.locks[asset_id]
        
        if lock.locked():
            return # Al bezig
            
        async with lock:
            self.processing_tasks.add(asset_id)
            try:
                # 1. Evidence laden
                # REASON: De loader gebruikt sync psycopg2, we draaien dit in een executor
                loop = asyncio.get_event_loop()
                evidence = await loop.run_in_executor(None, self.loader.load_current_evidence, asset_id)
                
                # 2. Inference
                engine = self.inference_pool.get_engine(asset_id)
                result = engine.infer(evidence)
                
                # 3. Opslaan
                await self.writer.write_prediction(asset_id, result)
                
                logger.info(f"✅ Inference voltooid voor Asset {asset_id} (Time: {result.inference_time_ms:.2f}ms)")
                
            except Exception as e:
                logger.error(f"❌ Inference gefaald voor Asset {asset_id}: {e}")
            finally:
                self.processing_tasks.remove(asset_id)

    def get_heartbeat(self):
        """Retourneert de laatste hartslag voor de watchdog."""
        return self.last_heartbeat

