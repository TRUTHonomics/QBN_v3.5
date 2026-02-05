"""
GPU Combination Data Loader - Laadt data uit DRIE MTF-tabellen en merget op GPU.

ARCHITECTUUR NOOT:
- GEEN zware JOINs op de Database VM (10.10.10.3)
- Vier aparte simpele SELECT queries (lead, coin, conf, outcomes)
- Merge via GPU sorted merge (cp.searchsorted) - 50x sneller dan SQL JOIN

SEMANTISCHE CLASSIFICATIE:
- Leading signalen: kfl.mtf_signals_lead
- Coincident signalen: kfl.mtf_signals_coin  
- Confirming signalen: kfl.mtf_signals_conf

De classificatie Leading/Coincident/Confirming is per SIGNAALTYPE (niet per timeframe)
en is vastgelegd in qbn.signal_classification.

Gebruik:
    loader = GPUCombinationDataLoader(gpu_config)
    loader.load_and_cache(asset_id=1, lookback_days=365)
    data = loader.get_cached_data()
"""

import logging
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from config.gpu_config import GPUConfig
from database.db import get_cursor

logger = logging.getLogger(__name__)


# REASON: Horizon-specifieke suffix mapping conform MTF architectuur
HORIZON_CONFIG = {
    '1h': {'suffix': '60', 'outcome_col': 'outcome_1h'},
    '4h': {'suffix': '240', 'outcome_col': 'outcome_4h'},
    '1d': {'suffix': 'd', 'outcome_col': 'outcome_1d'},
}


@dataclass
class CachedCombinationData:
    """
    Container voor gecachte combinatie data op GPU/CPU.
    
    ARCHITECTUUR:
    - Leading/Coincident/Confirming per timeframe suffix
    - Suffixes: _d (daily), _240 (4h), _60 (1h), _1 (1m)
    - Outcomes per horizon: 1h, 4h, 1d
    """
    
    time: Any  # cp.ndarray of np.ndarray
    
    # Leading composite scores (uit mtf_signals_lead)
    leading_d: Any
    leading_240: Any
    leading_60: Any
    
    # Coincident composite scores (uit mtf_signals_coin)
    coincident_d: Any
    coincident_240: Any
    coincident_60: Any
    
    # Confirming composite scores (uit mtf_signals_conf)
    confirming_d: Any
    confirming_240: Any
    confirming_60: Any
    
    # Outcomes per horizon
    outcome_1h: Any
    outcome_4h: Any
    outcome_1d: Any
    
    # Uniqueness weights (1/N per physical event)
    uniqueness_weight: Any
    
    # Metadata
    asset_id: int
    n_rows: int
    cached_at: datetime
    is_gpu: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Converteer naar dictionary voor verdere verwerking."""
        return {
            'time': self.time,
            # Leading (per timeframe)
            'leading_d': self.leading_d,
            'leading_240': self.leading_240,
            'leading_60': self.leading_60,
            # Coincident (per timeframe)
            'coincident_d': self.coincident_d,
            'coincident_240': self.coincident_240,
            'coincident_60': self.coincident_60,
            # Confirming (per timeframe)
            'confirming_d': self.confirming_d,
            'confirming_240': self.confirming_240,
            'confirming_60': self.confirming_60,
            # Outcomes
            'outcome_1h': self.outcome_1h,
            'outcome_4h': self.outcome_4h,
            'outcome_1d': self.outcome_1d,
            # Uniqueness
            'uniqueness_weight': self.uniqueness_weight
        }
    
    def get_composites_for_horizon(self, horizon: str) -> Tuple[Any, Any, Any]:
        """
        Haal de juiste Leading/Coincident/Confirming op voor een horizon.
        
        Args:
            horizon: '1h', '4h', of '1d'
            
        Returns:
            (leading, coincident, confirming) arrays voor de gegeven horizon
        """
        suffix = HORIZON_CONFIG[horizon]['suffix']
        
        return (
            getattr(self, f'leading_{suffix}'),
            getattr(self, f'coincident_{suffix}'),
            getattr(self, f'confirming_{suffix}')
        )
    
    def get_outcome_for_horizon(self, horizon: str) -> Any:
        """
        Haal de juiste outcome kolom op voor een horizon.
        
        Args:
            horizon: '1h', '4h', of '1d'
            
        Returns:
            outcome array voor de gegeven horizon
        """
        outcome_col = HORIZON_CONFIG[horizon]['outcome_col']
        return getattr(self, outcome_col)


class GPUCombinationDataLoader:
    """
    Laadt data in batches van DB en cachet in GPU memory.
    
    KRITIEK: Geen zware JOINs in de database!
    Alle merging gebeurt op de GPU Node (10.10.10.2).
    
    Data komt uit DRIE MTF-tabellen:
    - kfl.mtf_signals_lead (Leading signalen)
    - kfl.mtf_signals_coin (Coincident signalen)
    - kfl.mtf_signals_conf (Confirming signalen)
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        """
        Initialize GPU Combination Data Loader.
        
        Args:
            config: GPU configuration (uses default if None)
        """
        self.config = config or GPUConfig.from_env()
        self._cached_data: Optional[CachedCombinationData] = None
        
        # Determine if GPU is available
        self.use_gpu = self.config.use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu:
            try:
                cp.cuda.Device(self.config.device_id).use()
                logger.info(f"GPUCombinationDataLoader initialized on GPU device {self.config.device_id}")
            except Exception as e:
                if self.config.auto_fallback_on_error:
                    logger.warning(f"GPU init failed: {e}, falling back to CPU")
                    self.use_gpu = False
                else:
                    raise
        else:
            logger.info("GPUCombinationDataLoader using CPU mode")
    
    @property
    def xp(self):
        """Get array library (CuPy if GPU, NumPy if CPU)."""
        return cp if self.use_gpu else np
    
    def load_and_cache(
        self,
        asset_id: int,
        lookback_days: Optional[int] = None,
        end_time: Optional[datetime] = None
    ) -> CachedCombinationData:
        """
        Laad alle benodigde data in één keer en cache op GPU.
        
        Strategie:
        1. Fetch LEADING signals in batch (geen JOIN)
        2. Fetch COINCIDENT signals in batch (geen JOIN)
        3. Fetch CONFIRMING signals in batch (geen JOIN)
        4. Fetch outcomes in batch (geen JOIN)
        5. Merge op GPU (veel sneller dan SQL JOIN)
        
        Args:
            asset_id: Asset ID to load
            lookback_days: Number of days to look back (None = all data)
            end_time: End time for data (default: now)
            
        Returns:
            CachedCombinationData with merged data
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        # REASON: None = all data (geen lookback limiet)
        if lookback_days is None:
            start_time = datetime(2000, 1, 1, tzinfo=timezone.utc)  # Effectief "all data"
            lookback_desc = "all data"
        else:
            start_time = end_time - timedelta(days=lookback_days)
            lookback_desc = f"{lookback_days} days"
        
        logger.info(f"Loading data for asset {asset_id}, lookback: {lookback_desc}")
        
        # Fetch alle vier datasets parallel (max efficiency)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_lead = executor.submit(
                self._fetch_mtf_signals, 'lead', asset_id, start_time, end_time
            )
            future_coin = executor.submit(
                self._fetch_mtf_signals, 'coin', asset_id, start_time, end_time
            )
            future_conf = executor.submit(
                self._fetch_mtf_signals, 'conf', asset_id, start_time, end_time
            )
            future_outcomes = executor.submit(
                self._fetch_outcomes, asset_id, start_time, end_time
            )
            
            df_lead = future_lead.result()
            df_coin = future_coin.result()
            df_conf = future_conf.result()
            df_outcomes = future_outcomes.result()
        
        logger.info(
            f"Fetched: {len(df_lead)} lead, {len(df_coin)} coin, "
            f"{len(df_conf)} conf, {len(df_outcomes)} outcomes"
        )
        
        # Validatie
        if len(df_lead) == 0:
            raise ValueError(f"No LEADING signals found for asset {asset_id}")
        if len(df_coin) == 0:
            raise ValueError(f"No COINCIDENT signals found for asset {asset_id}")
        if len(df_conf) == 0:
            raise ValueError(f"No CONFIRMING signals found for asset {asset_id}")
        if len(df_outcomes) == 0:
            raise ValueError(f"No outcomes found for asset {asset_id}")
        
        # Merge op GPU (inner join op time_1 over alle vier datasets)
        self._cached_data = self._gpu_merge_four(
            df_lead, df_coin, df_conf, df_outcomes, asset_id
        )
        
        logger.info(
            f"Cached {self._cached_data.n_rows} merged rows on "
            f"{'GPU' if self.use_gpu else 'CPU'}"
        )
        
        return self._cached_data
    
    def _fetch_mtf_signals(
        self,
        signal_type: str,
        asset_id: int,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch MTF signals - simpele SELECT, GEEN JOIN.
        
        REASON: Minimaliseer database load, alle zware operaties op GPU.
        
        Args:
            signal_type: 'lead', 'coin', of 'conf'
            asset_id: Asset ID
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame met time_1 en concordance_score_* kolommen
        """
        table_map = {
            'lead': 'kfl.mtf_signals_lead',
            'coin': 'kfl.mtf_signals_coin',
            'conf': 'kfl.mtf_signals_conf',
        }
        
        table = table_map[signal_type]
        
        # REASON: Simpele query zonder JOINs
        # EXPL: Haal alle concordance_scores op voor d, 240, 60 timeframes
        query = f"""
        SELECT 
            time_1,
            concordance_score_d,
            concordance_score_240,
            concordance_score_60
        FROM {table}
        WHERE asset_id = %s
          AND time_1 >= %s
          AND time_1 < %s
        ORDER BY time_1
        """
        
        with get_cursor() as cur:
            cur.execute(query, (asset_id, start_time, end_time))
            rows = cur.fetchall()
        
        if not rows:
            return pd.DataFrame(columns=[
                'time_1', 'concordance_score_d', 
                'concordance_score_240', 'concordance_score_60'
            ])
        
        df = pd.DataFrame(rows, columns=[
            'time_1', 'concordance_score_d', 
            'concordance_score_240', 'concordance_score_60'
        ])
        # REASON: Normaliseer time_1 naar datetime64[ns, UTC] zodat merge en downstream
        # geen "Invalid value ... for dtype 'datetime64[ns]'" geven (DB/driver kan [us] leveren).
        df['time_1'] = pd.to_datetime(df['time_1'], utc=True, errors='coerce')
        return df
    
    def _fetch_outcomes(
        self,
        asset_id: int,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch outcomes uit qbn.barrier_outcomes en berekent uniqueness gewichten.
        
        REASON: Migratie naar First-Touch Barriers en Uniqueness weighting.
        """
        query = """
        SELECT 
            time_1,
            first_significant_barrier,
            first_significant_time_min
        FROM qbn.barrier_outcomes
        WHERE asset_id = %s
          AND time_1 >= %s
          AND time_1 < %s
        ORDER BY time_1
        """
        
        with get_cursor() as cur:
            cur.execute(query, (asset_id, start_time, end_time))
            rows = cur.fetchall()
        
        if not rows:
            return pd.DataFrame(columns=['time_1', 'outcome_1h', 'outcome_4h', 'outcome_1d', 'uniqueness_weight'])
        
        df = pd.DataFrame(rows, columns=['time_1', 'first_significant_barrier', 'first_significant_time_min'])
        # REASON: Zelfde normalisatie als MTF fetches: datetime64[ns, UTC] voor consistente merge.
        df['time_1'] = pd.to_datetime(df['time_1'], utc=True, errors='coerce')
        
        # 1. Bepaal binary outcomes voor de drie horizons
        for horizon in ['1h', '4h', '1d']:
            h_min = 60 if horizon == '1h' else (240 if horizon == '4h' else 1440)
            col = f'outcome_{horizon}'
            df[col] = 0
            
            is_up = df['first_significant_barrier'].str.startswith('up', na=False)
            is_down = df['first_significant_barrier'].str.startswith('down', na=False)
            is_within = df['first_significant_time_min'] <= h_min
            
            df.loc[is_up & is_within, col] = 1
            df.loc[is_down & is_within, col] = -1

        # 2. Bereken Uniqueness gewicht (1/N)
        # N = aantal signalen die dezelfde fysieke barrier hit claimen
        # REASON: first_significant_time_min kan NaN zijn (bijv. barrier 'none'),
        # wat RuntimeWarning kan geven in pandas bij to_timedelta cast.
        # REASON: Initialiseer als datetime64[ns, UTC] om dtype mismatch met time_1 te voorkomen (Pandas 2.x).
        df['hit_timestamp'] = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')
        mask_valid_time = df['time_1'].notna()
        mask_valid_min = df['first_significant_time_min'].notna()
        mask_valid = mask_valid_time & mask_valid_min
        if mask_valid.any():
            df.loc[mask_valid, 'hit_timestamp'] = (
                df.loc[mask_valid, 'time_1']
                + pd.to_timedelta(df.loc[mask_valid, 'first_significant_time_min'], unit='m')
            )
        
        mask_hit = (df['first_significant_barrier'] != 'none') & (df['first_significant_barrier'].notna())
        df['uniqueness_weight'] = 1.0
        
        if mask_hit.any():
            counts = df[mask_hit].groupby(['first_significant_barrier', 'hit_timestamp']).size().reset_index(name='cluster_n')
            df = df.merge(counts, on=['first_significant_barrier', 'hit_timestamp'], how='left')
            df.loc[mask_hit, 'uniqueness_weight'] = 1.0 / df.loc[mask_hit, 'cluster_n']
            df.drop(columns=['cluster_n'], inplace=True)
        
        # Drop tijdelijke kolommen om geheugen op GPU te sparen
        return df[['time_1', 'outcome_1h', 'outcome_4h', 'outcome_1d', 'uniqueness_weight']]
    
    def _gpu_merge_four(
        self,
        df_lead: pd.DataFrame,
        df_coin: pd.DataFrame,
        df_conf: pd.DataFrame,
        df_outcomes: pd.DataFrame,
        asset_id: int
    ) -> CachedCombinationData:
        """
        Merge vier DataFrames op GPU via sorted merge.
        
        REASON: GPU merge is ~50x sneller dan PostgreSQL JOIN voor grote datasets.
        Gebruikt cp.searchsorted voor O(n log n) sorted merge.
        
        Strategie:
        1. Vind gemeenschappelijke timestamps in alle vier datasets
        2. Extract only matching rows
        """
        xp = self.xp
        
        # REASON: Convert timestamps to int64 for comparison.
        # .to_numpy().view('int64') werkt voor zowel DatetimeArray als numpy.ndarray.
        times_lead = df_lead['time_1'].to_numpy().view('int64')
        times_coin = df_coin['time_1'].to_numpy().view('int64')
        times_conf = df_conf['time_1'].to_numpy().view('int64')
        times_out = df_outcomes['time_1'].to_numpy().view('int64')
        
        if self.use_gpu:
            result = self._merge_gpu(
                times_lead, times_coin, times_conf, times_out,
                df_lead, df_coin, df_conf, df_outcomes
            )
        else:
            result = self._merge_cpu(
                times_lead, times_coin, times_conf, times_out,
                df_lead, df_coin, df_conf, df_outcomes
            )
        
        # Unpack result
        (time_result, n_rows,
         lead_d, lead_240, lead_60,
         coin_d, coin_240, coin_60,
         conf_d, conf_240, conf_60,
         outcome_1h, outcome_4h, outcome_1d,
         uniqueness_weight) = result
        
        return CachedCombinationData(
            time=time_result,
            # Leading
            leading_d=lead_d,
            leading_240=lead_240,
            leading_60=lead_60,
            # Coincident
            coincident_d=coin_d,
            coincident_240=coin_240,
            coincident_60=coin_60,
            # Confirming
            confirming_d=conf_d,
            confirming_240=conf_240,
            confirming_60=conf_60,
            # Outcomes
            outcome_1h=outcome_1h,
            outcome_4h=outcome_4h,
            outcome_1d=outcome_1d,
            # Uniqueness
            uniqueness_weight=uniqueness_weight,
            # Metadata
            asset_id=asset_id,
            n_rows=n_rows,
            cached_at=datetime.now(timezone.utc),
            is_gpu=self.use_gpu
        )
    
    def _merge_gpu(
        self,
        times_lead: np.ndarray,
        times_coin: np.ndarray,
        times_conf: np.ndarray,
        times_out: np.ndarray,
        df_lead: pd.DataFrame,
        df_coin: pd.DataFrame,
        df_conf: pd.DataFrame,
        df_outcomes: pd.DataFrame
    ) -> Tuple:
        """GPU-accelerated merge van vier datasets."""
        
        # Transfer naar GPU
        t_lead = cp.asarray(times_lead)
        t_coin = cp.asarray(times_coin)
        t_conf = cp.asarray(times_conf)
        t_out = cp.asarray(times_out)
        
        # Stap 1: Vind gemeenschappelijke timestamps
        # We gebruiken outcomes als basis en zoeken matches in de andere drie
        
        # Find indices waar outcomes matchen met lead
        idx_out_in_lead = cp.searchsorted(t_lead, t_out)
        valid_lead = idx_out_in_lead < len(t_lead)
        matching_lead_times = t_lead[cp.minimum(idx_out_in_lead, len(t_lead) - 1)]
        match_lead = valid_lead & (matching_lead_times == t_out)
        
        # Find indices waar outcomes matchen met coin
        idx_out_in_coin = cp.searchsorted(t_coin, t_out)
        valid_coin = idx_out_in_coin < len(t_coin)
        matching_coin_times = t_coin[cp.minimum(idx_out_in_coin, len(t_coin) - 1)]
        match_coin = valid_coin & (matching_coin_times == t_out)
        
        # Find indices waar outcomes matchen met conf
        idx_out_in_conf = cp.searchsorted(t_conf, t_out)
        valid_conf = idx_out_in_conf < len(t_conf)
        matching_conf_times = t_conf[cp.minimum(idx_out_in_conf, len(t_conf) - 1)]
        match_conf = valid_conf & (matching_conf_times == t_out)
        
        # Final mask: alle vier moeten matchen
        final_mask = match_lead & match_coin & match_conf
        
        # Extract matched indices
        matched_out_idx = cp.where(final_mask)[0]
        matched_lead_idx = idx_out_in_lead[final_mask]
        matched_coin_idx = idx_out_in_coin[final_mask]
        matched_conf_idx = idx_out_in_conf[final_mask]
        
        n_rows = int(cp.sum(final_mask))
        
        # Extract data
        # Leading
        lead_d = cp.asarray(
            df_lead['concordance_score_d'].fillna(0).values.astype(np.float32)
        )[matched_lead_idx]
        lead_240 = cp.asarray(
            df_lead['concordance_score_240'].fillna(0).values.astype(np.float32)
        )[matched_lead_idx]
        lead_60 = cp.asarray(
            df_lead['concordance_score_60'].fillna(0).values.astype(np.float32)
        )[matched_lead_idx]
        
        # Coincident
        coin_d = cp.asarray(
            df_coin['concordance_score_d'].fillna(0).values.astype(np.float32)
        )[matched_coin_idx]
        coin_240 = cp.asarray(
            df_coin['concordance_score_240'].fillna(0).values.astype(np.float32)
        )[matched_coin_idx]
        coin_60 = cp.asarray(
            df_coin['concordance_score_60'].fillna(0).values.astype(np.float32)
        )[matched_coin_idx]
        
        # Confirming
        conf_d = cp.asarray(
            df_conf['concordance_score_d'].fillna(0).values.astype(np.float32)
        )[matched_conf_idx]
        conf_240 = cp.asarray(
            df_conf['concordance_score_240'].fillna(0).values.astype(np.float32)
        )[matched_conf_idx]
        conf_60 = cp.asarray(
            df_conf['concordance_score_60'].fillna(0).values.astype(np.float32)
        )[matched_conf_idx]
        
        # Outcomes - handle NaN as -999
        outcome_1h = cp.asarray(
            df_outcomes['outcome_1h'].fillna(-999).values.astype(np.float32)
        )[matched_out_idx]
        outcome_4h = cp.asarray(
            df_outcomes['outcome_4h'].fillna(-999).values.astype(np.float32)
        )[matched_out_idx]
        outcome_1d = cp.asarray(
            df_outcomes['outcome_1d'].fillna(-999).values.astype(np.float32)
        )[matched_out_idx]
        
        # Uniqueness weight
        uniqueness_weight = cp.asarray(
            df_outcomes['uniqueness_weight'].fillna(1.0).values.astype(np.float32)
        )[matched_out_idx]
        
        time_result = t_out[matched_out_idx]
        
        return (
            time_result, n_rows,
            lead_d, lead_240, lead_60,
            coin_d, coin_240, coin_60,
            conf_d, conf_240, conf_60,
            outcome_1h, outcome_4h, outcome_1d,
            uniqueness_weight
        )
    
    def _merge_cpu(
        self,
        times_lead: np.ndarray,
        times_coin: np.ndarray,
        times_conf: np.ndarray,
        times_out: np.ndarray,
        df_lead: pd.DataFrame,
        df_coin: pd.DataFrame,
        df_conf: pd.DataFrame,
        df_outcomes: pd.DataFrame
    ) -> Tuple:
        """CPU fallback merge van vier datasets."""
        
        # Stap 1: Vind gemeenschappelijke timestamps
        idx_out_in_lead = np.searchsorted(times_lead, times_out)
        valid_lead = idx_out_in_lead < len(times_lead)
        matching_lead_times = times_lead[np.minimum(idx_out_in_lead, len(times_lead) - 1)]
        match_lead = valid_lead & (matching_lead_times == times_out)
        
        idx_out_in_coin = np.searchsorted(times_coin, times_out)
        valid_coin = idx_out_in_coin < len(times_coin)
        matching_coin_times = times_coin[np.minimum(idx_out_in_coin, len(times_coin) - 1)]
        match_coin = valid_coin & (matching_coin_times == times_out)
        
        idx_out_in_conf = np.searchsorted(times_conf, times_out)
        valid_conf = idx_out_in_conf < len(times_conf)
        matching_conf_times = times_conf[np.minimum(idx_out_in_conf, len(times_conf) - 1)]
        match_conf = valid_conf & (matching_conf_times == times_out)
        
        # Final mask
        final_mask = match_lead & match_coin & match_conf
        
        # Extract matched indices
        matched_out_idx = np.where(final_mask)[0]
        matched_lead_idx = idx_out_in_lead[final_mask]
        matched_coin_idx = idx_out_in_coin[final_mask]
        matched_conf_idx = idx_out_in_conf[final_mask]
        
        n_rows = int(np.sum(final_mask))
        
        # Extract data
        # Leading
        lead_d = df_lead['concordance_score_d'].fillna(0).values.astype(np.float32)[matched_lead_idx]
        lead_240 = df_lead['concordance_score_240'].fillna(0).values.astype(np.float32)[matched_lead_idx]
        lead_60 = df_lead['concordance_score_60'].fillna(0).values.astype(np.float32)[matched_lead_idx]
        
        # Coincident
        coin_d = df_coin['concordance_score_d'].fillna(0).values.astype(np.float32)[matched_coin_idx]
        coin_240 = df_coin['concordance_score_240'].fillna(0).values.astype(np.float32)[matched_coin_idx]
        coin_60 = df_coin['concordance_score_60'].fillna(0).values.astype(np.float32)[matched_coin_idx]
        
        # Confirming
        conf_d = df_conf['concordance_score_d'].fillna(0).values.astype(np.float32)[matched_conf_idx]
        conf_240 = df_conf['concordance_score_240'].fillna(0).values.astype(np.float32)[matched_conf_idx]
        conf_60 = df_conf['concordance_score_60'].fillna(0).values.astype(np.float32)[matched_conf_idx]
        
        # Outcomes
        outcome_1h = df_outcomes['outcome_1h'].fillna(-999).values.astype(np.float32)[matched_out_idx]
        outcome_4h = df_outcomes['outcome_4h'].fillna(-999).values.astype(np.float32)[matched_out_idx]
        outcome_1d = df_outcomes['outcome_1d'].fillna(-999).values.astype(np.float32)[matched_out_idx]
        
        # Uniqueness weight
        uniqueness_weight = df_outcomes['uniqueness_weight'].fillna(1.0).values.astype(np.float32)[matched_out_idx]
        
        time_result = times_out[matched_out_idx]
        
        return (
            time_result, n_rows,
            lead_d, lead_240, lead_60,
            coin_d, coin_240, coin_60,
            conf_d, conf_240, conf_60,
            outcome_1h, outcome_4h, outcome_1d,
            uniqueness_weight
        )
    
    def get_cached_data(self) -> Optional[CachedCombinationData]:
        """Get currently cached data."""
        return self._cached_data
    
    def clear_cache(self):
        """Clear cached data and free GPU memory."""
        if self._cached_data is not None:
            self._cached_data = None
            
            if self.use_gpu:
                # Free GPU memory
                cp.get_default_memory_pool().free_all_blocks()
                logger.debug("GPU cache cleared")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        info = {
            'cached': self._cached_data is not None,
            'n_rows': self._cached_data.n_rows if self._cached_data else 0,
            'is_gpu': self.use_gpu
        }
        
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                info['gpu_used_bytes'] = mempool.used_bytes()
                info['gpu_total_bytes'] = mempool.total_bytes()
            except Exception:
                pass
        
        return info


def create_combination_loader(config: Optional[GPUConfig] = None) -> GPUCombinationDataLoader:
    """Factory function voor GPUCombinationDataLoader."""
    return GPUCombinationDataLoader(config)
