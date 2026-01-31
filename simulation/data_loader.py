import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from database.db import get_cursor
from inference.node_types import BarrierOutcomeState

logger = logging.getLogger(__name__)

class BacktestDataLoader:
    """
    Efficiënte data loader voor backtesting.
    Haalt signalen, prijzen en ATR parallel op.
    """
    
    def __init__(self, asset_id: int):
        self.asset_id = asset_id
        
    def _fetch_table_data(self, query: str, params: tuple) -> pd.DataFrame:
        """Helper om data op te halen in een aparte thread."""
        try:
            with get_cursor() as cur:
                cur.execute(query, params)
                if not cur.description:
                    return pd.DataFrame()
                return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        except Exception as e:
            logger.error(f"Error fetching data with query: {query[:100]}... Error: {e}")
            return pd.DataFrame()

    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Haal alle data op die nodig is voor backtesting (Signalen + OHLC + ATR).
        Paralleliseert nu over zowel tabellen als tijds-chunks voor maximale snelheid.
        """
        logger.info(f"📥 Fetching backtest data for asset {self.asset_id} ({start_date.date()} to {end_date.date()})...")
        
        # REASON: Splits de periode op in chunks van ± 3 maanden om DB parallelisatie te benutten.
        # EXPL: Een 8700G met 12 vCPUs kan veel meer parallelle queries aan dan 1 enkele stream.
        chunk_days = 90
        intervals = []
        curr = start_date
        while curr < end_date:
            next_curr = min(curr + pd.Timedelta(days=chunk_days), end_date)
            intervals.append((curr, next_curr))
            curr = next_curr

        all_queries = []
        
        # 1. Definieer de tabellen en hun specifieke query templates
        table_configs = [
            {
                'name': 'market',
                'query': "SELECT time as time_1, open, high, low, close, atr_14 FROM kfl.indicators WHERE asset_id = %s AND interval_min = '60' AND time BETWEEN %s AND %s",
                'time_col': 'time_1'
            },
            {
                'name': 'kfl.mtf_signals_lead',
                'query': "SELECT * FROM kfl.mtf_signals_lead WHERE asset_id = %s AND time_1 BETWEEN %s AND %s",
                'time_col': 'time_1'
            },
            {
                'name': 'kfl.mtf_signals_coin',
                'query': "SELECT * FROM kfl.mtf_signals_coin WHERE asset_id = %s AND time_1 BETWEEN %s AND %s",
                'time_col': 'time_1'
            },
            {
                'name': 'kfl.mtf_signals_conf',
                'query': "SELECT * FROM kfl.mtf_signals_conf WHERE asset_id = %s AND time_1 BETWEEN %s AND %s",
                'time_col': 'time_1'
            },
            {
                'name': 'outcomes',
                'query': "SELECT time_1, first_significant_barrier, first_significant_time_min FROM qbn.barrier_outcomes WHERE asset_id = %s AND time_1 BETWEEN %s AND %s",
                'time_col': 'time_1'
            }
        ]

        # 2. Bouw de lijst van alle (Tabel, Interval) combinaties
        query_tasks = []
        for config in table_configs:
            for start, end in intervals:
                query_tasks.append((config['name'], config['query'], (self.asset_id, start, end)))

        # 3. Voer alle tasks parallel uit
        # We gebruiken max_workers=15 om de 12 vCPUs van de DB goed te belasten zonder te overbelasten
        raw_results = []
        logger.info(f"🚀 Executing {len(query_tasks)} parallel data-fetch tasks (chunks of {chunk_days} days)...")
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_task = {executor.submit(self._fetch_table_data, q, p): (name, i) for i, (name, q, p) in enumerate(query_tasks)}
            for future in future_to_task:
                name, task_id = future_to_task[future]
                df_chunk = future.result()
                if not df_chunk.empty:
                    raw_results.append((name, df_chunk))

        # 4. Groepeer en concat chunks per tabel
        table_dfs = {}
        for name, df_chunk in raw_results:
            if name not in table_dfs:
                table_dfs[name] = []
            table_dfs[name].append(df_chunk)

        final_table_results = {}
        for name, chunks in table_dfs.items():
            combined_df = pd.concat(chunks).drop_duplicates(subset=['time_1']).sort_values('time_1')
            final_table_results[name] = combined_df

        # 5. Merge Logic (identiek aan origineel, maar met geconcatte resultaten)
        df = final_table_results.get('market')
        if df is None or df.empty:
            logger.error("No market data found!")
            return pd.DataFrame()
            
        df.set_index('time_1', inplace=True)
        
        # Merge signals
        for table in ['kfl.mtf_signals_lead', 'kfl.mtf_signals_coin', 'kfl.mtf_signals_conf']:
            sig_df = final_table_results.get(table)
            if sig_df is not None and not sig_df.empty:
                sig_df.set_index('time_1', inplace=True)
                if 'asset_id' in sig_df.columns:
                    sig_df.drop(columns=['asset_id'], inplace=True)
                
                cols_to_use = sig_df.columns.difference(df.columns)
                df = df.join(sig_df[cols_to_use], how='left')
        
        # Merge outcomes
        out_df = final_table_results.get('outcomes')
        if out_df is not None and not out_df.empty:
            out_df.set_index('time_1', inplace=True)
            df = df.join(out_df, how='left')
            
        df.reset_index(inplace=True)
        df.sort_values('time_1', inplace=True)
        
        # --- Post-processing (Fill NaNs etc) ---
        sig_cols = [c for c in df.columns if c.endswith(('_60', '_240', '_d'))]
        if sig_cols:
            df[sig_cols] = df[sig_cols].fillna(0)
            
        if 'atr_14' in df.columns:
            df['atr_14'] = df['atr_14'].ffill()
            df = df.rename(columns={'atr_14': 'atr'})  # v3.4 compatibility
            
        logger.info(f"📊 Parallel fetch complete: {len(df)} rows, {len(df.columns)} columns")
        return df
