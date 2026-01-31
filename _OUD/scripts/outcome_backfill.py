#!/usr/bin/env python3
"""
DEPRECATED: outcome_backfill.py
================================================================================
Dit script is verouderd en wordt vervangen door barrier_backfill.py.
Legacy point-in-time outcomes worden niet langer ondersteund voor QBN v3.3+.
================================================================================
"""

import sys
# ... (rest of imports)
import os
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import io
import multiprocessing
from multiprocessing import Pool
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import numpy as np
import pandas as pd

# REASON: Fix FutureWarning for downcasting in fillna
pd.set_option('future.no_silent_downcasting', True)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy not available - falling back to CPU mode")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor, close_pool
from inference.target_generator import ATR_THRESHOLDS, HORIZONS

from core.logging_utils import setup_logging

logger = setup_logging("outcome_backfill")

# REASON: MTF tabellen zijn verplaatst naar schema kfl en gesplitst per classificatie
MTF_TABLES = [
    'kfl.mtf_signals_lead',   # LEADING signalen
    'kfl.mtf_signals_coin',   # COINCIDENT signalen
    'kfl.mtf_signals_conf',   # CONFIRMING signalen
]

# REASON: Elke horizon heeft zijn eigen time-anchor en interval.
# Dit voorkomt autocorrelatie en koppelt outcomes aan de juiste signaal-timeframes.
HORIZON_CONFIG = {
    '1h': {
        'time_col': 'time_60',
        'time_close_col': 'time_close_60',
        'interval_min': '60',
        'horizon_minutes': 60,
        'expected_rows_factor': 1.0,  # Baseline
    },
    '4h': {
        'time_col': 'time_240',
        'time_close_col': 'time_close_240',
        'interval_min': '240',
        'horizon_minutes': 240,
        'expected_rows_factor': 0.25,  # 1/4 van 1h
    },
    '1d': {
        'time_col': 'time_d',
        'time_close_col': 'time_close_d',
        'interval_min': 'D',
        'horizon_minutes': 1440,
        'expected_rows_factor': 0.042,  # ~1/24 van 1h
    }
}


class GPUOutcomeBackfill:
    """
    GPU-accelerated outcome backfill met zero lookahead bias.

    Key Features:
    - Vectorized ATR-relative binning op GPU
    - Binary COPY voor bulk updates
    - Lookahead bias prevention via timestamp filtering
    - Resume capability (skips already processed rows)
    - Ondersteunt alle drie MTF tabellen (lead/coin/conf)
    """

    def __init__(self, asset_name: str = "Unknown", batch_size: int = 50000, gpu_batch_size: int = 100000, use_gpu: bool = True):
        """
        Initialize GPU outcome backfill.

        Args:
            asset_name: Name of the asset being processed (for logging)
            batch_size: Database fetch batch size (High performance with GPU-side joins)
            gpu_batch_size: GPU processing batch size
            use_gpu: Use GPU if available
        """
        self.asset_name = asset_name
        self.batch_size = batch_size
        self.gpu_batch_size = gpu_batch_size
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            logger.info("✅ GPU mode enabled (CuPy)")
        else:
            logger.info("ℹ️  CPU mode (GPU not available or disabled)")

        # ATR thresholds from target_generator.py
        self.atr_thresholds = ATR_THRESHOLDS  # [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    def _get_asset_id(self, asset_identifier: str) -> Optional[int]:
        """
        Convert asset name or ID to asset_id via symbols.symbols.
        
        Args:
            asset_identifier: Asset name (e.g., 'BTCUSDT') or ID string (e.g., '1')
            
        Returns:
            Asset ID or None if not found
        """
        if asset_identifier.isdigit():
            return int(asset_identifier)

        query = """
            SELECT id 
            FROM symbols.symbols 
            WHERE UPPER(bybit_symbol) = UPPER(%s)
               OR UPPER(kraken_symbol) = UPPER(%s)
            LIMIT 1
        """
        
        try:
            with get_cursor() as cur:
                cur.execute(query, (asset_identifier, asset_identifier))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Error getting asset_id for {asset_identifier}: {e}")
            return None

    def _fetch_signals(self, asset_id: int, watermark: datetime, cutoff_interval: str, horizon: str) -> pd.DataFrame:
        """Fetch raw signals using horizon-specific time columns.
        
        Args:
            asset_id: Asset ID to fetch
            watermark: Start timestamp (exclusive)
            cutoff_interval: Interval string for lookahead prevention
            horizon: '1h', '4h', or '1d' - determines which time columns to use
        
        Returns:
            DataFrame with columns [time_1, time_close_1, asset_id] where:
            - For 1h: time_1 = time_60, time_close_1 = time_close_60
            - For 4h: time_1 = time_240, time_close_1 = time_close_240
            - For 1d: time_1 = time_d, time_close_1 = time_close_d
        """
        config = HORIZON_CONFIG[horizon]
        time_col = config['time_col']
        time_close_col = config['time_close_col']
        
        # REASON: DISTINCT ON voorkomt duplicaten (meerdere 1-min rijen per interval).
        # Elke horizon gebruikt zijn eigen time anchor.
        query = f"""
            SELECT DISTINCT ON ({time_col}) 
                {time_col}, {time_close_col}, asset_id
            FROM kfl.mtf_signals_lead
            WHERE asset_id = %s AND {time_col} > %s
              AND {time_close_col} < NOW() - INTERVAL '{cutoff_interval}'
            ORDER BY {time_col} LIMIT %s
        """
        try:
            with get_cursor() as cur:
                cur.execute(query, (asset_id, watermark, self.batch_size))
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame()
                # REASON: Rename naar generieke namen voor compatibiliteit met downstream code
                return pd.DataFrame(rows, columns=['time_1', 'time_close_1', 'asset_id'])
        except Exception as e:
            logger.error(f"Error fetching signals for {horizon}: {e}")
            return pd.DataFrame()

    def _fetch_klines_range(self, asset_id: int, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch 1m klines for a given range to use for entry/exit prices."""
        query = """
            SELECT time, close
            FROM kfl.klines_raw
            WHERE asset_id = %s AND interval_min = '1'
              AND time BETWEEN %s AND %s
        """
        try:
            with get_cursor() as cur:
                cur.execute(query, (asset_id, start_time, end_time))
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows, columns=['time', 'close'])
        except Exception as e:
            logger.error(f"Error fetching klines range: {e}")
            return pd.DataFrame()

    def _fetch_indicators_range(self, asset_id: int, interval_min: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch ATR-14 indicators for a given range and interval."""
        query = """
            SELECT time, atr_14
            FROM kfl.indicators
            WHERE asset_id = %s AND interval_min = %s
              AND time BETWEEN %s AND %s
        """
        try:
            with get_cursor() as cur:
                cur.execute(query, (asset_id, interval_min, start_time, end_time))
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows, columns=['time', 'atr_14'])
        except Exception as e:
            logger.error(f"Error fetching indicators range: {e}")
            return pd.DataFrame()

    def fetch_processable_rows(self, asset: str, horizon: str, watermark: datetime) -> Tuple[pd.DataFrame, datetime]:
        """
        Fetch and align data using Python-side joins to offload DB VM.
        """
        asset_id = self._get_asset_id(asset)
        if asset_id is None:
            logger.error(f"Asset '{asset}' not found in symbols.symbols")
            return pd.DataFrame(), watermark, 0
        
        # REASON: Gebruik HORIZON_CONFIG voor horizon-specifieke parameters
        config = HORIZON_CONFIG[horizon]
        horizon_minutes = config['horizon_minutes']
        cutoff_interval = f"{horizon_minutes} minutes"

        # 1. Fetch raw signals using horizon-specific time columns
        signals = self._fetch_signals(asset_id, watermark, cutoff_interval, horizon)
        if signals.empty:
            return pd.DataFrame(), watermark, 0
        
        new_watermark = signals['time_1'].iloc[-1]
        
        # 2. Determine time range for pricing and indicators
        # Range needs to cover [min(time_1), max(time_close_1) + horizon]
        min_time = signals['time_1'].min() - timedelta(days=2) # Margin for indicators
        max_time = signals['time_close_1'].max() + timedelta(minutes=horizon_minutes + 1)
        
        # 3. Fetch data in bulk ranges (sequential scans on DB)
        # REASON: ATR interval komt uit HORIZON_CONFIG voor consistentie
        atr_interval = config['interval_min']
        
        # Fetch 1m prices for entry/exit
        prices = self._fetch_klines_range(asset_id, min_time, max_time)
        
        # Fetch ATR data
        indicators = self._fetch_indicators_range(asset_id, atr_interval, min_time, max_time)
        
        # Fetch 1m ATR for the legacy column
        indicators_1m = self._fetch_indicators_range(asset_id, '1', min_time, max_time)

        if prices.empty or indicators.empty:
            logger.warning(f"No price or indicator data found for {asset} range {min_time} to {max_time}")
            return pd.DataFrame(), new_watermark, len(signals)

        # 4. Alignment in Python (Pandas)
        # ---------------------------------------------------------
        
        # A. Match Entry Prices (Exact match on time_close_1)
        # REASON: Signals for candle [time_1, time_close_1] are only known at time_close_1.
        # Entry must happen at the moment the signal is finalized.
        df = pd.merge(signals, prices, left_on='time_close_1', right_on='time', how='left')
        df = df.rename(columns={'close': 'entry_close'}).drop(columns=['time'])
        
        # B. Match Exit Prices (Exact match on time_close_1 + horizon)
        # REASON: Exit must be horizon minutes after entry.
        df['exit_time'] = df['time_close_1'] + timedelta(minutes=horizon_minutes)
        df = pd.merge(df, prices, left_on='exit_time', right_on='time', how='left')
        df = df.rename(columns={'close': 'exit_close'}).drop(columns=['time'])
        
        # C. Match Horizon ATR (Lookahead-safe merge_asof)
        # Need to match indicators.time_close <= signals.time_close_1
        # Bepaal indicator close times
        interval_delta_map = {
            '60': timedelta(minutes=60),
            '240': timedelta(minutes=240),
            'D': timedelta(days=1),
            '1': timedelta(minutes=1)
        }
        
        indicators['time_close'] = indicators['time'] + interval_delta_map[atr_interval]
        indicators_1m['time_close'] = indicators_1m['time'] + interval_delta_map['1']
        
        # Sort for merge_asof
        df = df.sort_values('time_close_1')
        indicators = indicators.sort_values('time_close')
        indicators_1m = indicators_1m.sort_values('time_close')
        
        # Horizon ATR
        df = pd.merge_asof(
            df, indicators,
            left_on='time_close_1', right_on='time_close',
            direction='backward'
        )
        df = df.rename(columns={'atr_14': f'atr_{horizon}'}).drop(columns=['time', 'time_close'])
        
        # 1m ATR (legacy)
        df = pd.merge_asof(
            df, indicators_1m,
            left_on='time_close_1', right_on='time_close',
            direction='backward'
        )
        df = df.rename(columns={'atr_14': 'atr_at_signal'}).drop(columns=['time', 'time_close'])
        
        # Filter where essential data is missing
        df_final = df.dropna(subset=['entry_close', 'exit_close', f'atr_{horizon}'])
        
        return df_final, new_watermark, len(signals)

    def calculate_outcomes_gpu(self, df: pd.DataFrame, horizon: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate outcomes using GPU vectorization.

        Args:
            df: DataFrame with current data (entry_close, exit_close, atr_horizon)
            horizon: Horizon name ('1h', '4h', '1d')

        Returns:
            Tuple of (outcomes, returns_pct) as numpy arrays
        """
        # Calculate returns: (exit - entry) / entry * 100
        # REASON: Ensure float64 to avoid object dtype issues with CuPy
        entry_close = df['entry_close'].values.astype(np.float64)
        exit_close = df['exit_close'].values.astype(np.float64)
        
        # REASON: Gebruik horizon-specifieke ATR voor binning
        atr_col = f'atr_{horizon}'
        atr_absolute = df[atr_col].values.astype(np.float64)

        # Return percentage
        returns_pct = ((exit_close - entry_close) / entry_close) * 100

        # Handle NaN future prices
        valid_mask = ~np.isnan(exit_close)

        # REASON: Use atr_at_signal (absolute value) and convert to percentage for binning
        # ATR binning is relative: return_pct / (atr_at_signal / entry_close * 100)
        atr_pct = (atr_absolute / entry_close) * 100  # Convert ATR to percentage

        if self.use_gpu:
            # Transfer to GPU
            returns_gpu = cp.array(returns_pct)
            atr_pct_gpu = cp.array(atr_pct)
            valid_mask_gpu = cp.array(valid_mask)

            # ATR-normalize returns (returns_pct / atr_pct)
            atr_normalized = cp.where(atr_pct_gpu > 0, returns_gpu / atr_pct_gpu, 0)

            # Bin to -3 to +3 scale (vectorized)
            # REASON: Gebruik dynamische thresholds voor consistentie
            t = self.atr_thresholds
            outcomes = cp.zeros_like(atr_normalized, dtype=cp.int8)
            outcomes = cp.where(atr_normalized < t[0], -3, outcomes)
            outcomes = cp.where((atr_normalized >= t[0]) & (atr_normalized < t[1]), -2, outcomes)
            outcomes = cp.where((atr_normalized >= t[1]) & (atr_normalized < t[2]), -1, outcomes)
            outcomes = cp.where((atr_normalized >= t[2]) & (atr_normalized < t[3]), 0, outcomes)
            outcomes = cp.where((atr_normalized >= t[3]) & (atr_normalized < t[4]), 1, outcomes)
            outcomes = cp.where((atr_normalized >= t[4]) & (atr_normalized < t[5]), 2, outcomes)
            outcomes = cp.where(atr_normalized >= t[5], 3, outcomes)

            # Set invalid outcomes to None (will be skipped in update)
            outcomes = cp.where(valid_mask_gpu, outcomes, -99)

            # Transfer back to CPU
            outcomes_cpu = cp.asnumpy(outcomes)
        else:
            # CPU fallback
            atr_normalized = np.where(atr_pct > 0, returns_pct / atr_pct, 0)

            t = self.atr_thresholds
            outcomes_cpu = np.zeros_like(atr_normalized, dtype=np.int8)
            outcomes_cpu[atr_normalized < t[0]] = -3
            outcomes_cpu[(atr_normalized >= t[0]) & (atr_normalized < t[1])] = -2
            outcomes_cpu[(atr_normalized >= t[1]) & (atr_normalized < t[2])] = -1
            outcomes_cpu[(atr_normalized >= t[2]) & (atr_normalized < t[3])] = 0
            outcomes_cpu[(atr_normalized >= t[3]) & (atr_normalized < t[4])] = 1
            outcomes_cpu[(atr_normalized >= t[4]) & (atr_normalized < t[5])] = 2
            outcomes_cpu[atr_normalized >= t[5]] = 3

            # Mark invalid
            outcomes_cpu[~valid_mask] = -99

        return outcomes_cpu, returns_pct

    def bulk_update_outcomes(self, updates: List[Tuple], horizon: str):
        """
        Use binary COPY for efficient bulk UPSERT to qbn.signal_outcomes table.

        REASON: Outcomes are normalized into qbn.signal_outcomes.
        
        Args:
            updates: List of (time_1, asset_id, outcome, return_pct, atr_horizon, atr_1m) tuples
            horizon: Horizon name ('1h', '4h', '1d')
        """
        if not updates:
            return

        try:
            with get_cursor(commit=True) as cur:
                # REASON: Fix "tuple decompression limit exceeded" error for large batches
                cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                
                # Create temp table
                temp_table = f"outcome_updates_{horizon.replace('.', '_')}"
                cur.execute(f"""
                CREATE TEMP TABLE {temp_table} (
                    time_1 TIMESTAMPTZ,
                    asset_id INTEGER,
                    outcome_{horizon} SMALLINT,
                    return_{horizon}_pct REAL,
                    atr_{horizon} REAL,
                    atr_at_signal REAL
                ) ON COMMIT DROP
                """)

                # Prepare data for COPY
                # REASON: Using pandas to_csv is much faster than a manual Python loop for buffer filling
                df_updates = pd.DataFrame(updates, columns=[
                    'time_1', 'asset_id', f'outcome_{horizon}', 
                    f'return_{horizon}_pct', f'atr_{horizon}', 'atr_at_signal'
                ])
                buffer = io.StringIO()
                df_updates.to_csv(buffer, sep='\t', header=False, index=False, na_rep='\\N')
                buffer.seek(0)

                # Binary COPY to temp table
                cur.copy_from(
                    buffer,
                    temp_table,
                    columns=['time_1', 'asset_id', f'outcome_{horizon}', f'return_{horizon}_pct', f'atr_{horizon}', 'atr_at_signal'],
                    null='\\N'
                )

                # REASON: UPSERT into qbn.signal_outcomes using (asset_id, time_1) as key
                # Dit slaat zowel de horizon-specifieke ATR als de 1m ATR op.
                cur.execute(f"""
                INSERT INTO qbn.signal_outcomes (
                    asset_id, time_1, 
                    outcome_{horizon}, return_{horizon}_pct, 
                    atr_{horizon}, atr_at_signal, 
                    updated_at
                )
                SELECT asset_id, time_1, outcome_{horizon}, return_{horizon}_pct, atr_{horizon}, atr_at_signal, NOW()
                FROM {temp_table}
                ON CONFLICT (asset_id, time_1) DO UPDATE SET
                    outcome_{horizon} = EXCLUDED.outcome_{horizon},
                    return_{horizon}_pct = EXCLUDED.return_{horizon}_pct,
                    atr_{horizon} = EXCLUDED.atr_{horizon},
                    atr_at_signal = COALESCE(EXCLUDED.atr_at_signal, qbn.signal_outcomes.atr_at_signal),
                    updated_at = NOW()
                """)

                logger.debug(f"Bulk upserted {len(updates)} rows into qbn.signal_outcomes for horizon {horizon}")

        except Exception as e:
            logger.error(f"Error in bulk_update_outcomes: {e}")
            raise

    def run_backfill_for_asset(self, asset: str, horizons: List[str] = ['1h', '4h', '1d']):
        """
        Main backfill loop with resume capability using normalized outcomes.

        Args:
            asset: Asset name
            horizons: List of horizons to process
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting backfill for {asset} (Normalized Outcomes)")
        logger.info(f"{'='*80}")

        for horizon in horizons:
            logger.info(f"\n  [{horizon}] Processing horizon...")

            # Bepaal initiële watermark uit de database
            asset_id = self._get_asset_id(asset)
            with get_cursor() as cur:
                cur.execute(f"""
                    SELECT COALESCE(MAX(time_1), '2020-01-01'::timestamptz)
                    FROM qbn.signal_outcomes
                    WHERE asset_id = %s AND outcome_{horizon} IS NOT NULL
                """, (asset_id,))
                watermark = cur.fetchone()[0]

            total_processed = 0
            batch_num = 0
            empty_batches_in_a_row = 0
            MAX_EMPTY_BATCHES = 1000 # Skip max 100.000 rijen zonder data (nodig voor Daily ATR warm-up)

            while True:
                # Fetch next batch (lookahead-safe, uses local watermark)
                # REASON: Return raw signals count to prevent premature loop break if rows are dropped during join
                df, watermark, signals_fetched = self.fetch_processable_rows(asset, horizon, watermark)

                if df is None or (signals_fetched == 0 and empty_batches_in_a_row > MAX_EMPTY_BATCHES):
                    logger.info(f"  [{horizon}] ✅ Complete - no more rows to process")
                    break
                
                if signals_fetched == 0:
                    empty_batches_in_a_row += 1
                    # Log af en toe dat we nog aan het zoeken zijn
                    if empty_batches_in_a_row % 10 == 0:
                        logger.info(f"  [{horizon}] {self.asset_name} - Skipping empty period (batch {empty_batches_in_a_row})...")
                    continue
                
                # Reset counter als we data vinden
                empty_batches_in_a_row = 0
                batch_num += 1
                
                if len(df) == 0:
                    logger.info(f"  [{horizon}] {self.asset_name} - Batch {batch_num}: 0 processable rows (from {signals_fetched} signals)")
                    # Don't break here, we still need to check if we reached end of signals
                    if signals_fetched < self.batch_size:
                        break
                    continue

                logger.info(f"  [{horizon}] {self.asset_name} - Batch {batch_num}: {len(df)} processable rows (from {signals_fetched} signals)")

                # Calculate outcomes on GPU
                outcomes, returns_pct = self.calculate_outcomes_gpu(df, horizon)
                # ...
                # Prepare updates (skip invalid outcomes)
                # Tuple: (time_1, asset_id, outcome, return_pct, atr_horizon, atr_1m)
                valid_mask = outcomes != -99
                
                # REASON: Vectorized update preparation for massive performance boost over iloc loop
                time_1_arr = df['time_1'].values[valid_mask]
                asset_id_arr = df['asset_id'].values[valid_mask].astype(int)
                outcomes_valid = outcomes[valid_mask].astype(int)
                returns_valid = returns_pct[valid_mask]
                
                atr_col = f'atr_{horizon}'
                atr_h_arr = df[atr_col].values[valid_mask]
                atr_1m_arr = df['atr_at_signal'].values[valid_mask]

                # Bouw updates met zip (veel sneller dan iloc per row)
                updates = list(zip(
                    time_1_arr,
                    asset_id_arr,
                    outcomes_valid,
                    [float(r) if not np.isnan(r) else None for r in returns_valid],
                    [float(a) if not np.isnan(a) else None for a in atr_h_arr],
                    [float(a) if not np.isnan(a) else None for a in atr_1m_arr]
                    ))

                # Bulk update to qbn.signal_outcomes
                if updates:
                    self.bulk_update_outcomes(updates, horizon)

                total_processed += len(updates)
                logger.info(f"  [{horizon}] {self.asset_name} - Batch {batch_num} complete - Total: {total_processed}")

                # Break if we didn't fetch a full batch (end of signal table)
                if signals_fetched < self.batch_size:
                    break

            logger.info(f"  [{horizon}] ✅ Finished - {total_processed} total rows processed")


def get_all_assets_with_mtf_data() -> List[str]:
    """Get all unique asset names from MTF tables via symbols.symbols."""
    with get_cursor() as cur:
        # REASON: Query all three MTF tables for unique assets
        cur.execute("""
            SELECT DISTINCT s.bybit_symbol
            FROM (
                SELECT asset_id FROM kfl.mtf_signals_lead
                UNION
                SELECT asset_id FROM kfl.mtf_signals_coin
                UNION
                SELECT asset_id FROM kfl.mtf_signals_conf
            ) mtf
            JOIN symbols.symbols s ON s.id = mtf.asset_id
            ORDER BY s.bybit_symbol
        """)
        return [row[0] for row in cur.fetchall()]


def show_backfill_status():
    """Show current backfill status for normalized outcomes per horizon."""
    print("\n" + "="*80)
    print("OUTCOME BACKFILL STATUS (NORMALIZED qbn.signal_outcomes)")
    print("="*80)

    with get_cursor() as cur:
        # REASON: Per horizon de unieke signal records tellen
        print("\n   Signal Records per Horizon (LEAD table):")
        for horizon, config in HORIZON_CONFIG.items():
            time_col = config['time_col']
            cur.execute(f"SELECT COUNT(DISTINCT {time_col}) FROM kfl.mtf_signals_lead")
            count = cur.fetchone()[0]
            print(f"     {horizon} (via {time_col}): {count:,}")

        cur.execute("""
            SELECT COUNT(*) FROM qbn.signal_outcomes
        """)
        total_outcomes = cur.fetchone()[0]

        cur.execute("""
            SELECT
                COUNT(outcome_1h) as with_1h,
                COUNT(outcome_4h) as with_4h,
                COUNT(outcome_1d) as with_1d,
                COUNT(atr_at_signal) as with_atr
            FROM qbn.signal_outcomes
        """)
        with_1h, with_4h, with_1d, with_atr = cur.fetchone()

        print(f"\n   Total outcome records: {total_outcomes:,}")
        print(f"\n   Filled Outcomes (verwacht: 1h > 4h > 1d):")
        if total_outcomes > 0:
            print(f"     outcome_1h: {with_1h:,}")
            print(f"     outcome_4h: {with_4h:,}")
            print(f"     outcome_1d: {with_1d:,}")
            print(f"     atr_at_signal: {with_atr:,}")
        
        # Check lookahead bias per horizon (critical)
        print(f"\n   Lookahead Bias Check:")
        for horizon, config in HORIZON_CONFIG.items():
            time_col = config['time_col']
            time_close_col = config['time_close_col']
            horizon_minutes = config['horizon_minutes']
            
            cur.execute(f"""
                SELECT COUNT(*) FROM qbn.signal_outcomes so
                JOIN kfl.mtf_signals_lead mtf ON so.asset_id = mtf.asset_id AND so.time_1 = mtf.{time_col}
                WHERE mtf.{time_close_col} + INTERVAL '{horizon_minutes} minutes' > NOW()
                  AND so.outcome_{horizon} IS NOT NULL
            """)
            violations = cur.fetchone()[0]
            if violations > 0:
                print(f"     ⚠️  {horizon}: LOOKAHEAD BIAS - {violations} violations!")
            else:
                print(f"     ✅ {horizon}: OK")

    print("\n" + "="*80)


def get_selected_assets() -> List[int]:
    """Get all asset IDs where selected_in_current_run = 1."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id FROM symbols.symbols 
            WHERE selected_in_current_run = 1
            ORDER BY id
        """)
        return [row[0] for row in cur.fetchall()]


def run_parallel_backfill(assets: List[str], horizons: List[str], batch_size: int, use_gpu: bool, workers: int):
    """Run backfill in parallel using a process pool."""
    logger.info(f"🚀 Starting parallel backfill with {workers} workers for {len(assets)} assets")
    
    # REASON: Sluit bestaande verbindingen voor het forken om "connection already closed" errors te voorkomen.
    close_pool()
    
    tasks = [(asset, horizons, batch_size, use_gpu) for asset in assets]
    
    with Pool(processes=workers) as pool:
        if tqdm:
            # imap_unordered laat ons tqdm gebruiken voor de voortgang van de assets
            list(tqdm(pool.imap_unordered(worker_task_wrapper, tasks), total=len(assets), desc="Backfilling Assets"))
        else:
            pool.starmap(worker_task, tasks)

def worker_task_wrapper(args):
    """Helper voor imap_unordered."""
    return worker_task(*args)


def worker_task(asset: str, horizons: List[str], batch_size: int, use_gpu: bool):
    """Individual worker function for one asset."""
    process_id = multiprocessing.current_process().name
    try:
        logger.info(f"👷 {process_id} starting work on asset: {asset}")
        # Initialiseer een verse instance per proces (belangrijk voor CUDA/DB)
        backfill = GPUOutcomeBackfill(
            asset_name=asset,
            batch_size=batch_size,
            use_gpu=use_gpu
        )
        backfill.run_backfill_for_asset(asset, horizons)
        logger.info(f"✅ {process_id} finished asset: {asset}")
    except Exception as e:
        logger.error(f"❌ {process_id} error for asset {asset}: {e}")


def run_validation(asset_name: Optional[str] = None, selected: bool = False, all_assets: bool = False):
    """Run validation and analysis script and save report."""
    logger.info("\n" + "="*80)
    logger.info("Running validation & analysis...")
    logger.info("="*80)
    
    import subprocess
    from datetime import datetime
    
    cmd = [sys.executable, 'scripts/validate_outcome_backfill.py']
    # REASON: Gebruik de volledige analyse (niet --quick) voor het eindrapport
    scope_desc = "single"
    
    if all_assets:
        cmd.append('--all')
        scope_desc = "all"
    elif selected:
        cmd.append('--selected')
        scope_desc = "selected"
    elif asset_name:
        cmd.extend(['--asset', str(asset_name)])
        scope_desc = f"asset_{asset_name}"
    else:
        # Default fallback
        cmd.append('--all')
        scope_desc = "all"

    try:
        # Run validation and capture output for the report
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        output = result.stdout
        
        # Toon op console
        print(output)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Sla op in _validation/ map (conform docker-menu.py)
        validation_dir = Path('_validation')
        validation_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_filename = validation_dir / f'outcome_analysis_{scope_desc}_{timestamp}.md'
        
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Outcome Analyse Rapport\n\n")
            f.write(f"**Scope:** {scope_desc}\n")
            f.write(f"**Timestamp:** {datetime.now().isoformat()}\n")
            f.write(f"**Status:** {'✅ OK' if result.returncode == 0 else '⚠️ Warnings/Errors'}\n\n")
            f.write("---\n\n")
            f.write("## Output\n\n")
            f.write("```\n")
            f.write(output)
            f.write("\n```\n")
            if result.stderr:
                f.write("\n## Errors\n\n")
                f.write("```\n")
                f.write(result.stderr)
                f.write("\n```\n")
            
        logger.info(f"✅ Analysis complete. Report saved to: {md_filename}")
        
    except Exception as e:
        logger.error(f"❌ Error during validation/analysis: {e}")


def main():
    """Main CLI interface."""
    import time
    print("\n" + "!"*80)
    print("! DEPRECATED: Dit script (outcome_backfill.py) is verouderd.")
    print("! Gebruik scripts/barrier_backfill.py voor QBN v3.3+.")
    print("!"*80 + "\n")
    time.sleep(2)
    
    parser = argparse.ArgumentParser(description='GPU-accelerated outcome backfill for QBN_v3')
    parser.add_argument('--asset', help='Specific asset name or ID to backfill (e.g., BTCUSDT or 1)')
    parser.add_argument('--all', action='store_true', help='Backfill all assets')
    parser.add_argument('--selected', action='store_true', help='Backfill only selected assets (selected_in_current_run=1)')
    parser.add_argument('--horizon', choices=['1h', '4h', '1d', 'all'], default='all',
                       help='Specific horizon (default: all)')
    parser.add_argument('--batch-size', type=int, default=50000, help='DB batch size (Default: 50000, optimized for GPU joins)')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers (Default: 2, max 4 for high DB load)')
    parser.add_argument('--status', action='store_true', help='Show backfill status')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU (CPU only)')

    args = parser.parse_args()

    # Status check
    if args.status:
        show_backfill_status()
        return 0

    # Initialize backfill (for single asset mode or metadata checks)
    backfill = GPUOutcomeBackfill(
        asset_name=args.asset or "Bulk",
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu
    )

    # Determine assets to process
    if args.all:
        assets = get_all_assets_with_mtf_data()
        logger.info(f"Processing all {len(assets)} assets")
    elif args.selected:
        assets = [str(aid) for aid in get_selected_assets()]
        logger.info(f"Processing {len(assets)} selected assets")
    elif args.asset:
        assets = [args.asset]
    else:
        parser.print_help()
        return 1

    # REASON: Limit parallel workers to prevent DB connection pressure on weaker servers
    if args.workers > 2:
        logger.warning(f"⚠️  {args.workers} workers can put significant load on the DB server. Consider --workers 2 for remote databases.")

    # Determine horizons
    horizons = [args.horizon] if args.horizon != 'all' else ['1h', '4h', '1d']

    # Run backfill (Parallel or Single)
    start_time = datetime.now()
    
    if len(assets) > 1 and args.workers > 1:
        run_parallel_backfill(assets, horizons, args.batch_size, not args.no_gpu, args.workers)
    else:
        # Single asset mode
        for asset in assets:
            try:
                # REASON: Verse instance per asset met de juiste naam voor logging
                backfill = GPUOutcomeBackfill(
                    asset_name=asset,
                    batch_size=args.batch_size, 
                    use_gpu=not args.no_gpu
                )
                backfill.run_backfill_for_asset(asset, horizons)
            except Exception as e:
                logger.error(f"Error processing {asset}: {e}")
                continue

    elapsed = datetime.now() - start_time
    logger.info(f"\n✅ Backfill complete - Elapsed time: {elapsed}")

    # REASON: Altijd validatie en analyse uitvoeren aan het eind van de run (op verzoek van gebruiker)
    # Dit genereert ook de .md rapporten in _validation/ zodat analyse altijd beschikbaar is.
    run_validation(
        asset_name=args.asset if not args.all and not args.selected else None,
        selected=args.selected,
        all_assets=args.all
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
