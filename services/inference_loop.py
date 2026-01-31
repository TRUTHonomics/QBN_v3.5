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
from typing import Dict, Set, Optional, List, Any
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
from services.signal_writer import QBNOutputWriter

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
        self.output_writer = QBNOutputWriter()

        # Symbol mapping (asset_id -> symbol)
        self.symbol_map: Dict[int, str] = self._load_symbol_map(asset_ids)
        
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
        # NOTE: QBNOutputWriter gebruikt sync get_cursor(), geen async pool nodig
        
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
        # REASON: Luister naar het rolling signal kanaal i.p.v. statische MTF updates
        channel = 'rolling_signal_update'
        
        async def on_notify(connection, pid, channel, payload):
            try:
                data = json.loads(payload)
                asset_id = data.get('asset_id')
                # rolling_signal_update payload bevat asset_id, time, completeness
                
                if asset_id in self.asset_ids:
                    # We voegen het asset toe aan de queue voor verwerking
                    await self.queue.put((asset_id, data))
                    logger.debug(f"🔔 Rolling Notify ontvangen voor Asset {asset_id}")
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
                loop = asyncio.get_event_loop()
                evidence = await loop.run_in_executor(None, self.loader.load_rolling_evidence, asset_id)
                
                # 2. Check voor actieve positie (v3.1)
                position_context = await self._get_active_position_context(asset_id)
                
                # 3. Inference (run_inference voor dual output)
                engine = self.inference_pool.get_engine(asset_id)
                result = engine.run_inference(evidence, position_context)

                # 4. Entry Output (qbn.output_entry)
                symbol = self.symbol_map.get(asset_id, f"ASSET_{asset_id}")
                # Note: QBNOutputWriter.from_inference_result moet mogelijk geupdate worden 
                # of ik gebruik de legacy 'infer' voor qbn.output_entry.
                # Maar Plan 06 suggereert een nieuwe dual flow.
                
                output = QBNOutputWriter.from_inference_result(result, symbol)
                self.output_writer.write(output)
                
                # 5. Position Output (qbn.output_position)
                if position_context and result.position_prediction:
                    self._write_position_output(result, position_context)

                # REASON: Timestamp en asset_id expliciet in logbericht voor betere monitoring in Docker Desktop.
                ts_str = evidence.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(evidence.timestamp, 'strftime') else str(evidence.timestamp)
                hypothesis_status = f"📤 {result.trade_hypothesis}" if result.trade_hypothesis != 'no_setup' else "⏭️ no_setup"
                logger.info(f"[{ts_str}] Asset {asset_id}: ✅ Inference voltooid ({hypothesis_status}, {result.inference_time_ms:.2f}ms)")
                
            except Exception as e:
                # REASON: Foutmeldingen ook voorzien van asset_id en context voor Docker logs.
                ts_err = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                logger.error(f"[{ts_err}] Asset {asset_id}: ❌ Inference gefaald: {e}")
            finally:
                self.processing_tasks.remove(asset_id)

    async def _get_active_position_context(self, asset_id: int) -> Optional[Any]:
        """
        Haal actieve positie context op (Hook voor TSEM integratie).
        In v3.1 migration base: returned None (geen actieve positie gedetecteerd).
        """
        # Hier zou een query naar een 'active_positions' tabel of TSEM API komen.
        return None

    def _write_position_output(self, result: Any, context: Any):
        """
        Schrijf resultaten naar qbn.output_position.
        
        v3.4: RAC velden worden als NULL geschreven (deprecated)
        v3.3: Alle velden inclusief RAC
        v3.2: Alleen legacy velden
        """
        from database.db import get_cursor
        pred = result.position_prediction
        
        # Legacy v3.2 confidence distribution
        dist = getattr(result, 'position_confidence_distribution', {})
        p_low = dist.get('low', 0.0)
        p_medium = dist.get('medium', 0.0)
        p_high = dist.get('high', 0.0)
        
        # v3.4 outputs (MP, VR, ET - direct naar TSEM)
        momentum_dist = getattr(result, 'momentum_distribution', {})
        volatility_dist = getattr(result, 'volatility_distribution', {})
        exit_dist = getattr(result, 'exit_timing_distribution', {})
        
        with get_cursor() as cur:
            cur.execute("""
                INSERT INTO qbn.output_position (
                    asset_id, timestamp, position_confidence, confidence_score,
                    time_since_entry_min, current_pnl_atr,
                    p_target_hit, p_stoploss_hit, p_timeout,
                    dominant_outcome, prediction_confidence,
                    coincident_composite, confirming_composite,
                    inference_time_ms, model_version,
                    p_conf_low, p_conf_medium, p_conf_high,
                    momentum_prediction, p_momentum_bearish, p_momentum_neutral, p_momentum_bullish,
                    volatility_regime, p_vol_low, p_vol_normal, p_vol_high,
                    exit_timing, p_exit_now, p_hold, p_extend,
                    delta_leading, delta_coincident, delta_confirming
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
            """, (
                result.asset_id,
                result.timestamp,
                result.position_confidence,
                result.position_confidence_score,
                context.time_since_entry_min,
                context.current_pnl_atr,
                pred.target_hit,
                pred.stoploss_hit,
                pred.timeout,
                pred.dominant_outcome,
                pred.confidence,
                result.coincident_composite,
                result.confirming_composite,
                result.inference_time_ms,
                result.model_version,
                p_low, p_medium, p_high,
                # v3.4: Momentum Prediction (direct naar TSEM)
                getattr(result, 'momentum_prediction', None),
                momentum_dist.get('bearish', None),
                momentum_dist.get('neutral', None),
                momentum_dist.get('bullish', None),
                # v3.4: Volatility Regime (direct naar TSEM)
                getattr(result, 'volatility_regime', None),
                volatility_dist.get('low_vol', None),
                volatility_dist.get('normal', None),
                volatility_dist.get('high_vol', None),
                # v3.4: Exit Timing (direct naar TSEM)
                getattr(result, 'exit_timing', None),
                exit_dist.get('exit_now', None),
                exit_dist.get('hold', None),
                exit_dist.get('extend', None),
                # v3.4: Delta scores
                getattr(result, 'delta_leading', None),
                getattr(result, 'delta_coincident', None),
                getattr(result, 'delta_confirming', None)
            ))

    def get_heartbeat(self):
        """Retourneert de laatste hartslag voor de watchdog."""
        return self.last_heartbeat

    def _load_symbol_map(self, asset_ids: List[int]) -> Dict[int, str]:
        """Laad asset_id -> symbol mapping uit symbols.symbols."""
        from database.db import get_cursor
        symbol_map = {}
        with get_cursor() as cur:
            cur.execute("""
                SELECT id, kraken_symbol
                FROM symbols.symbols
                WHERE id = ANY(%s)
            """, (list(asset_ids),))
            for row in cur.fetchall():
                symbol_map[row[0]] = row[1]
        logger.info(f"📋 Symbol mapping geladen voor {len(symbol_map)} assets")
        return symbol_map

