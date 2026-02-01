#!/usr/bin/env python3
"""
barrier_backfill.py - GPU-accelerated barrier outcome backfill

Usage:
    python scripts/barrier_backfill.py --asset-id 1 --batch-size 5000
    python scripts/barrier_backfill.py --asset-id 1 --incremental --since 2026-01-01
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
from psycopg2.extras import execute_values

# Voeg project root toe aan path voor imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.barrier_config import BarrierConfig
from inference.barrier_outcome_generator import BarrierOutcomeGenerator
from inference.gpu_barrier_calculator import GPUBarrierCalculator
from database.db import get_cursor
from core.logging_utils import setup_logging


class BarrierBackfill:
    """Orchestrator voor barrier backfill met GPU-versnelling en bulk prefetch."""
    
    def __init__(
        self,
        asset_id: int,
        config: BarrierConfig,
        batch_size: int = 100000,
        checkpoint_file: str = None,
        run_id: str = None
    ):
        self.asset_id = asset_id
        self.config = config
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file or f'_checkpoint_barrier_{asset_id}.json'
        self.run_id = run_id
        
        self.gpu_calc = GPUBarrierCalculator(
            barriers=config.up_barriers,
            max_obs_min=config.max_observation_min
        )
        self.generator = BarrierOutcomeGenerator(config)
        self.logger = logging.getLogger("barrier_backfill")
        
        # REASON: Bulk kline cache (Numpy arrays voor memory efficiency)
        self._kline_times = np.array([], dtype=np.int64)
        self._kline_highs = np.array([], dtype=np.float32)
        self._kline_lows = np.array([], dtype=np.float32)
        self._cache_min_time = None
        self._cache_max_time = None
    
    def get_pending_timestamps_chunk(self, start_time: datetime, end_time: datetime, overwrite: bool = False) -> list:
        """Haal timestamps op voor een specifieke periode."""
        # REASON: Haal alleen data op uit mtf_signals_lead en indicators (voor ATR)
        # De zware JOIN met klines_raw en barrier_outcomes doen we hier NIET meer
        query = """
            SELECT mtf.time_1, ind.atr_14, k.close
            FROM kfl.mtf_signals_lead mtf
            JOIN kfl.indicators ind
                ON ind.asset_id = mtf.asset_id
                AND ind.time = mtf.time_d
                AND ind.interval_min = 'D'
            JOIN kfl.klines_raw k
                ON k.asset_id = mtf.asset_id
                AND k.time = mtf.time_1 + INTERVAL '59 minutes'
                AND k.interval_min = '1'
            WHERE mtf.asset_id = %s
              AND mtf.time_1 >= %s
              AND mtf.time_1 < %s
              AND EXTRACT(MINUTE FROM mtf.time_1) = 0
            ORDER BY mtf.time_1
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id, start_time, end_time))
            rows = cur.fetchall()
            
        if not rows:
            return []
            
        # REASON: Filter lokaal op bestaande outcomes als overwrite=False
        if not overwrite:
            timestamps = [row[0] for row in rows]
            existing_times = set()
            
            # Check in batches welke al bestaan
            batch_size = 10000
            for i in range(0, len(timestamps), batch_size):
                batch = timestamps[i:i+batch_size]
                with get_cursor() as cur:
                    cur.execute("""
                        SELECT time_1 FROM qbn.barrier_outcomes 
                        WHERE asset_id = %s AND time_1 = ANY(%s)
                    """, (self.asset_id, batch))
                    existing_times.update(row[0] for row in cur.fetchall())
            
            # Filter rows
            rows = [row for row in rows if row[0] not in existing_times]
            
        return rows
    
    def prefetch_all_klines(self, min_time: datetime, max_time: datetime) -> int:
        """
        Prefetch ALLE klines voor het hele bereik in één keer naar lokale Numpy cache.
        
        REASON: Dit elimineert per-batch database calls en versnelt de backfill 3-4x.
        De cache wordt in-memory gehouden als Numpy arrays voor minimale memory footprint.
        
        Args:
            min_time: Vroegste signaal timestamp
            max_time: Laatste signaal timestamp + max_observation_min
            
        Returns:
            Aantal gecachte klines
        """
        # REASON: Voeg buffer toe voor observation window
        fetch_max = max_time + timedelta(minutes=self.config.max_observation_min + 60)
        
        self.logger.info(f"📥 Prefetching klines van {min_time} tot {fetch_max}...")
        
        # REASON: Haal epoch seconds op voor directe opslag in int64 array
        query = """
            SELECT EXTRACT(EPOCH FROM time)::bigint, high, low
            FROM kfl.klines_raw
            WHERE asset_id = %s
              AND interval_min = '1'
              AND time >= %s
              AND time <= %s
            ORDER BY time
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id, min_time, fetch_max))
            rows = cur.fetchall()
        
        if not rows:
            self._kline_times = np.array([], dtype=np.int64)
            self._kline_highs = np.array([], dtype=np.float32)
            self._kline_lows = np.array([], dtype=np.float32)
            return 0

        # REASON: Direct converteren naar Numpy arrays (veel efficiënter dan dict)
        # rows is een list of tuples, we kunnen dit efficient omzetten
        data = np.array(rows, dtype=np.float64) # Eerst als float64 inlezen
        
        self._kline_times = data[:, 0].astype(np.int64)
        self._kline_highs = data[:, 1].astype(np.float32)
        self._kline_lows = data[:, 2].astype(np.float32)
        
        self._cache_min_time = min_time
        self._cache_max_time = fetch_max
        
        # Geheugengebruik berekenen
        mem_usage_mb = (self._kline_times.nbytes + self._kline_highs.nbytes + self._kline_lows.nbytes) / (1024 * 1024)
        self.logger.info(f"✅ Gecacht: {len(self._kline_times)} klines ({mem_usage_mb:.1f} MB)")
        
        return len(self._kline_times)
    
    def prefetch_klines_to_gpu(self, min_time: datetime, max_time: datetime) -> int:
        """
        Prefetch alle klines direct naar GPU geheugen.
        
        REASON: Voor maximale performance, laad klines direct naar GPU.
        Dit elimineert CPU-GPU transfers tijdens de backfill loop.
        
        Args:
            min_time: Vroegste signaal timestamp
            max_time: Laatste signaal timestamp
            
        Returns:
            Aantal geladen klines
        """
        # REASON: Voeg buffer toe voor observation window
        fetch_max = max_time + timedelta(minutes=self.config.max_observation_min + 60)
        
        self.logger.info(f"📥 Prefetching klines naar GPU van {min_time} tot {fetch_max}...")
        
        query = """
            SELECT EXTRACT(EPOCH FROM time)::bigint as epoch, high, low
            FROM kfl.klines_raw
            WHERE asset_id = %s
              AND interval_min = '1'
              AND time >= %s
              AND time <= %s
            ORDER BY time
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id, min_time, fetch_max))
            rows = cur.fetchall()
        
        if not rows:
            self.logger.warning("Geen klines gevonden!")
            return 0
        
        # REASON: Converteer naar numpy arrays voor GPU transfer
        kline_times = np.array([row[0] for row in rows], dtype=np.int64)
        kline_highs = np.array([row[1] for row in rows], dtype=np.float32)
        kline_lows = np.array([row[2] for row in rows], dtype=np.float32)
        
        # REASON: Laad naar GPU
        count = self.gpu_calc.load_klines_to_gpu(kline_times, kline_highs, kline_lows)
        
        self.logger.info(f"✅ GPU cache: {count} klines geladen")
        return count
    
    def fetch_klines_batch_from_cache(
        self,
        timestamps: list,
        ref_prices: np.ndarray
    ) -> np.ndarray:
        """
        Haal klines op uit lokale Numpy cache (geen DB call).
        
        REASON: Gebruikt np.searchsorted voor razendsnelle lookups in de gesorteerde time array.
        
        Returns:
            (N, T, 2) array met high/low per timestamp per minuut
        """
        N = len(timestamps)
        T = self.config.max_observation_min
        prices = np.zeros((N, T, 2), dtype=np.float32)
        
        # Converteer timestamps naar epoch seconds voor lookup
        # We voegen 60s toe omdat we starten na de signal candle (net als in de oude logica)
        # Shape: (N, 1)
        start_times = np.array([ts.timestamp() for ts in timestamps], dtype=np.int64) + 60
        
        # Maak matrix van alle benodigde timestamps: (N, T)
        # offsets: (1, T) -> [0, 60, 120, ...]
        offsets = np.arange(T, dtype=np.int64) * 60
        target_times = start_times[:, None] + offsets[None, :]
        
        # Flatten voor bulk search: (N*T,)
        target_times_flat = target_times.ravel()
        
        # Zoek indices in cache: O(log M)
        # side='left' geeft index i zodat cache[i-1] < v <= cache[i]
        idxs = np.searchsorted(self._kline_times, target_times_flat, side='left')
        
        # Clip indices om out-of-bounds te voorkomen bij check
        idxs_clipped = np.clip(idxs, 0, len(self._kline_times) - 1)
        
        # Check of gevonden timestamps exact matchen
        # Als cache leeg is, is len=0 en werkt dit ook (found_mask is False)
        if len(self._kline_times) > 0:
            found_mask = (self._kline_times[idxs_clipped] == target_times_flat)
        else:
            found_mask = np.zeros_like(target_times_flat, dtype=bool)
            
        # Vul resultaten
        # Default naar ref_prices (voor missing data)
        # prices shape: (N, T, 2) -> flatten naar (N*T, 2)
        prices_flat = prices.reshape(-1, 2)
        
        # Vul met ref_prices (herhaald voor elke minuut)
        # ref_prices is (N,), we hebben (N*T,) nodig
        ref_prices_expanded = np.repeat(ref_prices, T)
        prices_flat[:, 0] = ref_prices_expanded
        prices_flat[:, 1] = ref_prices_expanded
        
        # Overschrijf gevonden waarden
        if found_mask.any():
            valid_idxs = idxs_clipped[found_mask]
            prices_flat[found_mask, 0] = self._kline_highs[valid_idxs]
            prices_flat[found_mask, 1] = self._kline_lows[valid_idxs]
            
        return prices.reshape(N, T, 2)
    
    def fetch_klines_batch(
        self,
        timestamps: list,
        ref_prices: np.ndarray
    ) -> np.ndarray:
        """
        Haal klines op voor batch timestamps.
        
        Returns:
            (N, T, 2) array met high/low per timestamp per minuut
        """
        N = len(timestamps)
        T = self.config.max_observation_min
        prices = np.zeros((N, T, 2), dtype=np.float32)
        
        # Fetch in bulk
        min_time = min(timestamps)
        max_time = max(timestamps) + timedelta(minutes=T)
        
        query = """
            SELECT time, high, low
            FROM kfl.klines_raw
            WHERE asset_id = %s
              AND interval_min = '1'
              AND time > %s
              AND time <= %s
            ORDER BY time
        """
        
        with get_cursor() as cur:
            cur.execute(query, (self.asset_id, min_time, max_time))
            klines = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
        
        # Map naar batch array
        for i, ts in enumerate(timestamps):
            for j in range(T):
                t = ts + timedelta(minutes=j+60)
                if t in klines:
                    prices[i, j, 0] = klines[t][0]  # high
                    prices[i, j, 1] = klines[t][1]  # low
                else:
                    # Fill met ref_price als missing
                    prices[i, j, 0] = ref_prices[i]
                    prices[i, j, 1] = ref_prices[i]
        
        return prices
    
    def save_checkpoint(self, last_timestamp: datetime):
        """Sla checkpoint op voor resume."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'asset_id': self.asset_id,
                'last_timestamp': last_timestamp.isoformat(),
                'updated_at': datetime.now().isoformat()
            }, f)
    
    def load_checkpoint(self) -> datetime:
        """Laad laatste checkpoint."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                if data['asset_id'] == self.asset_id:
                    return datetime.fromisoformat(data['last_timestamp'])
        except FileNotFoundError:
            pass
        return None
    
    def process_chunk(self, start_time: datetime, end_time: datetime, overwrite: bool, use_gpu_lookup: bool):
        """Verwerk een tijd-chunk (bijv. 6 maanden)."""
        self.logger.info(f"Processing chunk: {start_time} -> {end_time}")
        
        # 1. Haal pending timestamps op voor deze chunk
        pending = self.get_pending_timestamps_chunk(start_time, end_time, overwrite)
        if not pending:
            self.logger.info("  Geen pending timestamps in deze chunk")
            return 0
            
        self.logger.info(f"  {len(pending)} timestamps te verwerken")
        
        # 2. Prefetch klines voor deze chunk
        # Let op: we hebben klines nodig tot end_time + max_obs
        chunk_max_time = pending[-1][0]
        
        if use_gpu_lookup:
            self.prefetch_klines_to_gpu(pending[0][0], chunk_max_time)
        else:
            self.prefetch_all_klines(pending[0][0], chunk_max_time)
            
        # 3. Verwerk in batches (zoals voorheen)
        processed = 0
        write_future = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            for batch_start in range(0, len(pending), self.batch_size):
                batch = pending[batch_start:batch_start + self.batch_size]
                
                timestamps = [row[0] for row in batch]
                atrs = np.array([row[1] or 1.0 for row in batch], dtype=np.float32)
                ref_prices = np.array([row[2] for row in batch], dtype=np.float32)
                
                if use_gpu_lookup:
                    signal_epochs = np.array([int(ts.timestamp()) for ts in timestamps], dtype=np.int64)
                    results = self.gpu_calc.calculate_batch_from_gpu_cache(
                        signal_epochs, ref_prices, atrs, offset_minutes=60
                    )
                else:
                    prices = self.fetch_klines_batch_from_cache(timestamps, ref_prices)
                    results = self.gpu_calc.calculate_batch(prices, ref_prices, atrs)
                
                barrier_names, barrier_times = self.gpu_calc.determine_first_significant_batch(
                    results, self.config.significant_threshold
                )
                
                if write_future is not None:
                    write_future.result()
                
                write_future = executor.submit(
                    self._save_batch_results,
                    timestamps, atrs, ref_prices, results,
                    barrier_names, barrier_times
                )
                
                processed += len(batch)
            
            if write_future is not None:
                write_future.result()
                
        # 4. Cleanup
        if use_gpu_lookup:
            self.gpu_calc.clear_gpu_cache()
        else:
            # Reset numpy arrays
            self._kline_times = np.array([], dtype=np.int64)
            self._kline_highs = np.array([], dtype=np.float32)
            self._kline_lows = np.array([], dtype=np.float32)
            
        return processed

    def run(self, since: datetime = None, resume: bool = True, overwrite: bool = False, use_gpu_lookup: bool = False):
        """
        Run de backfill met chunked processing (per 6 maanden).
        """
        if resume and since is None:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                since = checkpoint
                self.logger.info(f"Resuming from checkpoint: {since}")
        
        # Bepaal start en eind tijd
        if not since:
            # Haal oudste signaal op
            with get_cursor() as cur:
                cur.execute("SELECT MIN(time_1) FROM kfl.mtf_signals_lead WHERE asset_id = %s", (self.asset_id,))
                since = cur.fetchone()[0] or datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
            
        now = datetime.now(timezone.utc)
        current_start = since
        total_processed = 0
        chunk_months = 6

        # Chunk per 6 maanden
        while current_start < now:
            # Bereken eind van de chunk (6 maanden verder, eerste dag van die maand)
            m = current_start.month - 1 + chunk_months
            next_year = current_start.year + m // 12
            next_month = m % 12 + 1
            chunk_end = current_start.replace(year=next_year, month=next_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            if chunk_end.tzinfo is None and current_start.tzinfo:
                chunk_end = chunk_end.replace(tzinfo=timezone.utc)
            current_end = min(chunk_end, now)
            
            # Verwerk chunk
            processed = self.process_chunk(current_start, current_end, overwrite, use_gpu_lookup)
            total_processed += processed
            
            # Save checkpoint
            if processed > 0:
                self.save_checkpoint(current_end)
            
            current_start = current_end
            
        self.logger.info(f"Backfill complete: {total_processed} timestamps processed")
    
    def _save_batch_results(
        self,
        timestamps, atrs, ref_prices, results,
        barrier_names, barrier_times
    ):
        """
        Sla batch resultaten op via bulk UPSERT met execute_values.
        
        REASON: execute_values is 10-50x sneller dan executemany voor bulk inserts.
        """
        # REASON: execute_values vereist een andere query syntax (zonder VALUES placeholders)
        query = """
            INSERT INTO qbn.barrier_outcomes (
                asset_id, time_1, atr_at_signal, reference_price, max_observation_min,
                time_to_up_025_atr, time_to_up_050_atr, time_to_up_075_atr,
                time_to_up_100_atr, time_to_up_125_atr, time_to_up_150_atr,
                time_to_up_175_atr, time_to_up_200_atr, time_to_up_225_atr,
                time_to_up_250_atr, time_to_up_275_atr, time_to_up_300_atr,
                time_to_down_025_atr, time_to_down_050_atr, time_to_down_075_atr,
                time_to_down_100_atr, time_to_down_125_atr, time_to_down_150_atr,
                time_to_down_175_atr, time_to_down_200_atr, time_to_down_225_atr,
                time_to_down_250_atr, time_to_down_275_atr, time_to_down_300_atr,
                max_up_atr, max_down_atr, time_to_max_up_min, time_to_max_down_min,
                first_significant_barrier, first_significant_time_min, run_id
            ) VALUES %s
            ON CONFLICT (asset_id, time_1) DO UPDATE SET
                atr_at_signal = EXCLUDED.atr_at_signal,
                reference_price = EXCLUDED.reference_price,
                time_to_up_025_atr = EXCLUDED.time_to_up_025_atr,
                time_to_up_050_atr = EXCLUDED.time_to_up_050_atr,
                time_to_up_075_atr = EXCLUDED.time_to_up_075_atr,
                time_to_up_100_atr = EXCLUDED.time_to_up_100_atr,
                time_to_up_125_atr = EXCLUDED.time_to_up_125_atr,
                time_to_up_150_atr = EXCLUDED.time_to_up_150_atr,
                time_to_up_175_atr = EXCLUDED.time_to_up_175_atr,
                time_to_up_200_atr = EXCLUDED.time_to_up_200_atr,
                time_to_up_225_atr = EXCLUDED.time_to_up_225_atr,
                time_to_up_250_atr = EXCLUDED.time_to_up_250_atr,
                time_to_up_275_atr = EXCLUDED.time_to_up_275_atr,
                time_to_up_300_atr = EXCLUDED.time_to_up_300_atr,
                time_to_down_025_atr = EXCLUDED.time_to_down_025_atr,
                time_to_down_050_atr = EXCLUDED.time_to_down_050_atr,
                time_to_down_075_atr = EXCLUDED.time_to_down_075_atr,
                time_to_down_100_atr = EXCLUDED.time_to_down_100_atr,
                time_to_down_125_atr = EXCLUDED.time_to_down_125_atr,
                time_to_down_150_atr = EXCLUDED.time_to_down_150_atr,
                time_to_down_175_atr = EXCLUDED.time_to_down_175_atr,
                time_to_down_200_atr = EXCLUDED.time_to_down_200_atr,
                time_to_down_225_atr = EXCLUDED.time_to_down_225_atr,
                time_to_down_250_atr = EXCLUDED.time_to_down_250_atr,
                time_to_down_275_atr = EXCLUDED.time_to_down_275_atr,
                time_to_down_300_atr = EXCLUDED.time_to_down_300_atr,
                max_up_atr = EXCLUDED.max_up_atr,
                max_down_atr = EXCLUDED.max_down_atr,
                time_to_max_up_min = EXCLUDED.time_to_max_up_min,
                time_to_max_down_min = EXCLUDED.time_to_max_down_min,
                first_significant_barrier = EXCLUDED.first_significant_barrier,
                first_significant_time_min = EXCLUDED.first_significant_time_min,
                run_id = EXCLUDED.run_id,
                updated_at = NOW()
        """
        
        rows = []
        for i in range(len(timestamps)):
            def get_time(key):
                val = results.get(key, np.array([-1]))[i]
                return int(val) if val > 0 else None
            
            row = (
                self.asset_id, timestamps[i], float(atrs[i]), float(ref_prices[i]),
                self.config.max_observation_min,
                get_time('up_025'), get_time('up_050'), get_time('up_075'),
                get_time('up_100'), get_time('up_125'), get_time('up_150'),
                get_time('up_175'), get_time('up_200'), get_time('up_225'),
                get_time('up_250'), get_time('up_275'), get_time('up_300'),
                get_time('down_025'), get_time('down_050'), get_time('down_075'),
                get_time('down_100'), get_time('down_125'), get_time('down_150'),
                get_time('down_175'), get_time('down_200'), get_time('down_225'),
                get_time('down_250'), get_time('down_275'), get_time('down_300'),
                float(results['max_up_atr'][i]), float(results['max_down_atr'][i]),
                int(results['time_to_max_up'][i]), int(results['time_to_max_down'][i]),
                str(barrier_names[i]), int(barrier_times[i]) if barrier_times[i] > 0 else None,
                self.run_id
            )
            rows.append(row)
        
        # REASON: execute_values is 10-50x sneller dan executemany
        with get_cursor(commit=True) as cur:
            execute_values(cur, query, rows, page_size=5000)


def main():
    parser = argparse.ArgumentParser(description='GPU Barrier Backfill')
    parser.add_argument('--asset-id', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=100000)
    parser.add_argument('--since', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='default')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--incremental', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outcomes')
    parser.add_argument('--run-id', type=str, help='Run identifier for traceability')
    parser.add_argument('--status', action='store_true', help='Show backfill status')
    parser.add_argument('--gpu-lookup', action='store_true', 
                        help='Use GPU-side kline lookup (experimental, faster)')
    
    args = parser.parse_args()
    
    logger = setup_logging("barrier_backfill")

    if args.status:
        query = """
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE first_significant_barrier IS NOT NULL) as filled,
                MIN(time_1) as start_time,
                MAX(time_1) as end_time
            FROM qbn.barrier_outcomes
            WHERE asset_id = %s
        """
        with get_cursor() as cur:
            cur.execute(query, [args.asset_id])
            res = cur.fetchone()
            if res and res[0] > 0:
                print(f"\n📊 Status for Asset {args.asset_id}:")
                print(f"   Totaal rijen: {res[0]}")
                print(f"   Ingevuld:     {res[1]} ({res[1]/res[0]*100:.1f}%)")
                print(f"   Periode:      {res[2]} tot {res[3]}")
            else:
                print(f"\n⚠️  Geen data gevonden in qbn.barrier_outcomes voor asset {args.asset_id}")
        return
    
    logger.info(f"Starting barrier backfill for asset {args.asset_id} (run_id: {args.run_id})")
    
    config = BarrierConfig.from_database(args.config)
    
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since)
    
    backfill = BarrierBackfill(
        asset_id=args.asset_id,
        config=config,
        batch_size=args.batch_size,
        run_id=args.run_id
    )
    
    backfill.run(since=since, resume=not args.no_resume, overwrite=args.overwrite, 
                 use_gpu_lookup=args.gpu_lookup)


if __name__ == '__main__':
    main()
