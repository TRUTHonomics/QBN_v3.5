import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from database.db import get_cursor
from .barrier_config import BarrierConfig, BarrierOutcomeResult

logger = logging.getLogger(__name__)


class BarrierOutcomeGenerator:
    """
    Generator voor first-touch barrier outcomes.
    
    Berekent voor elke timestamp de tijd tot elke ATR-barrier wordt geraakt,
    plus de extremen binnen het observation window.
    """
    
    def __init__(self, config: Optional[BarrierConfig] = None):
        """
        Initialiseer de generator.
        
        Args:
            config: BarrierConfig instance (default: laad 'default' uit DB)
        """
        self.config = config or BarrierConfig.from_database("default")
        self._kline_cache: Dict[Tuple[int, datetime, datetime], pd.DataFrame] = {}
        
        logger.info(
            f"BarrierOutcomeGenerator initialized: "
            f"up_barriers={self.config.up_barriers}, "
            f"down_barriers={self.config.down_barriers}, "
            f"max_obs={self.config.max_observation_min}min"
        )
    
    def calculate_for_timestamp(
        self,
        asset_id: int,
        time_1: datetime,
        atr: float,
        ref_price: float
    ) -> BarrierOutcomeResult:
        """
        Bereken barrier outcomes voor één timestamp.
        
        Args:
            asset_id: Asset identifier
            time_1: Signaal timestamp
            atr: ATR waarde op dat moment
            ref_price: Referentie prijs (close op time_1)
            
        Returns:
            BarrierOutcomeResult met alle berekende waarden
            
        Raises:
            ValueError: Bij ongeldige inputs
        """
        # Valideer inputs
        self._validate_inputs(atr, ref_price)
        
        # Bepaal time range voor klines
        end_time = time_1 + timedelta(minutes=self.config.max_observation_min)
        
        # Haal klines op
        klines = self._fetch_klines(asset_id, time_1, end_time)
        
        if klines.empty:
            logger.warning(f"Geen klines gevonden voor {asset_id} @ {time_1}")
            return self._empty_result(asset_id, time_1, atr, ref_price)
        
        # Bereken barriers
        up_times = self._calculate_barriers(
            klines, ref_price, atr, self.config.up_barriers, direction='up'
        )
        down_times = self._calculate_barriers(
            klines, ref_price, atr, self.config.down_barriers, direction='down'
        )
        
        # Bereken extremen
        extremes = self._calculate_extremes(klines, ref_price, atr)
        
        # Bepaal first significant
        first_sig, first_time = self.determine_first_significant(
            up_times, down_times, self.config.significant_threshold
        )
        
        return BarrierOutcomeResult(
            asset_id=asset_id,
            time_1=time_1,
            atr_at_signal=atr,
            reference_price=ref_price,
            max_observation_min=self.config.max_observation_min,
            time_to_up_barriers=up_times,
            time_to_down_barriers=down_times,
            max_up_atr=extremes['max_up_atr'],
            max_down_atr=extremes['max_down_atr'],
            time_to_max_up_min=extremes['time_to_max_up_min'],
            time_to_max_down_min=extremes['time_to_max_down_min'],
            first_significant_barrier=first_sig,
            first_significant_time_min=first_time
        )
    
    def batch_calculate(
        self,
        asset_id: int,
        timestamps: List[Tuple[datetime, float, float]]
    ) -> List[BarrierOutcomeResult]:
        """
        Bereken barriers voor meerdere timestamps (CPU).
        
        Args:
            asset_id: Asset identifier
            timestamps: List van (time_1, atr, ref_price) tuples
            
        Returns:
            List van BarrierOutcomeResult
        """
        results = []
        for time_1, atr, ref_price in timestamps:
            try:
                result = self.calculate_for_timestamp(asset_id, time_1, atr, ref_price)
                results.append(result)
            except Exception as e:
                logger.error(f"Error calculating barriers for {time_1}: {e}")
                results.append(self._empty_result(asset_id, time_1, atr, ref_price))
        return results
    
    def determine_first_significant(
        self,
        up_times: Dict[str, Optional[int]],
        down_times: Dict[str, Optional[int]],
        threshold: float
    ) -> Tuple[str, Optional[int]]:
        """
        Bepaal de first significant barrier outcome.
        
        Logica:
        1. Zoek welke richting (up/down) de 'threshold' barrier als eerste raakt.
        2. In die winnende richting, bepaal de HOOGSTE barrier die geraakt is
           VOORDAT de tegenovergestelde threshold barrier geraakt werd (of einde window).
        """
        # Vind barrier level key voor threshold
        threshold_key = f"{int(threshold * 100):03d}"
        
        up_threshold_time = up_times.get(threshold_key)
        down_threshold_time = down_times.get(threshold_key)
        
        if up_threshold_time is None and down_threshold_time is None:
            return ("none", None)
            
        # Bepaal winnende richting gebaseerd op threshold
        # Winnaar is UP als up_threshold_time kleiner is (of als down_threshold_time None is)
        if down_threshold_time is None or (up_threshold_time is not None and up_threshold_time <= down_threshold_time):
            winner_dir = 'up'
            winner_times = up_times
            opposite_time = down_threshold_time or 99999
            winning_time = up_threshold_time
        else:
            winner_dir = 'down'
            winner_times = down_times
            opposite_time = up_threshold_time or 99999
            winning_time = down_threshold_time
            
        # Zoek verste barrier in winnende richting geraakt vóór opposite_time
        best_key = threshold_key
        best_time = winning_time
        
        # Sorteer keys om hoogste te vinden
        for key in sorted(winner_times.keys(), reverse=True):
            t = winner_times[key]
            if t is not None and t <= opposite_time:
                # Gevonden! Omdat we reverse sorteren is dit de hoogste.
                if int(key) >= int(threshold_key):
                    return (f"{winner_dir}_{key}", t)
        
        return (f"{winner_dir}_{best_key}", best_time)
    
    def save_to_database(self, result: BarrierOutcomeResult) -> bool:
        """
        Sla één resultaat op in de database.
        
        Args:
            result: BarrierOutcomeResult
            
        Returns:
            True bij succes
        """
        return self.save_batch([result]) == 1
    
    def save_batch(self, results: List[BarrierOutcomeResult]) -> int:
        """
        Sla batch resultaten op via UPSERT.
        
        Args:
            results: List van BarrierOutcomeResult
            
        Returns:
            Aantal opgeslagen records
        """
        if not results:
            return 0
        
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
                first_significant_barrier, first_significant_time_min,
                updated_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                NOW()
            )
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
                updated_at = NOW()
        """
        
        rows = [r.to_db_row() for r in results]
        
        with get_cursor(commit=True) as cur:
            cur.executemany(query, rows)
            
        return len(results)
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _validate_inputs(self, atr: float, ref_price: float) -> None:
        """Valideer input parameters."""
        if atr is None or atr <= 0:
            raise ValueError(f"ATR moet positief zijn, kreeg: {atr}")
        if ref_price is None or ref_price <= 0:
            raise ValueError(f"Reference price moet positief zijn, kreeg: {ref_price}")
    
    def _fetch_klines(
        self, 
        asset_id: int, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """
        Haal 1-minuut klines op uit database.
        
        Returns:
            DataFrame met columns: time, high, low
        """
        cache_key = (asset_id, start, end)
        if cache_key in self._kline_cache:
            return self._kline_cache[cache_key]
        
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
            cur.execute(query, (asset_id, start, end))
            rows = cur.fetchall()
        
        df = pd.DataFrame(rows, columns=['time', 'high', 'low'])
        
        # Cache alleen als niet te groot
        if len(df) < 10000:
            self._kline_cache[cache_key] = df
            
        return df
    
    def _calculate_barriers(
        self,
        klines: pd.DataFrame,
        ref_price: float,
        atr: float,
        barriers: List[float],
        direction: str
    ) -> Dict[str, Optional[int]]:
        """
        Bereken tijd tot elke barrier.
        
        Args:
            klines: DataFrame met high/low kolommen
            ref_price: Referentie prijs
            atr: ATR waarde
            barriers: Lijst van barrier levels (in ATR units)
            direction: 'up' of 'down'
            
        Returns:
            Dict met barrier_key -> minuten (of None)
        """
        results = {}
        
        for level in barriers:
            key = f"{int(level * 100):03d}"
            
            if direction == 'up':
                target_price = ref_price + (level * atr)
                # Zoek eerste kline waar high >= target
                mask = klines['high'] >= target_price
            else:
                target_price = ref_price - (level * atr)
                # Zoek eerste kline waar low <= target
                mask = klines['low'] <= target_price
            
            if mask.any():
                # Index is 0-based, +1 voor minuten sinds start
                first_idx = mask.idxmax()
                results[key] = int(first_idx) + 1
            else:
                results[key] = None
                
        return results
    
    def _calculate_extremes(
        self,
        klines: pd.DataFrame,
        ref_price: float,
        atr: float
    ) -> Dict[str, any]:
        """
        Bereken prijsextremen in ATR units.
        
        Returns:
            Dict met max_up_atr, max_down_atr, time_to_max_*
        """
        if klines.empty:
            return {
                'max_up_atr': 0.0,
                'max_down_atr': 0.0,
                'time_to_max_up_min': None,
                'time_to_max_down_min': None
            }
        
        # ATR-normalized prices
        highs_atr = (klines['high'] - ref_price) / atr
        lows_atr = (klines['low'] - ref_price) / atr
        
        max_up_idx = highs_atr.idxmax()
        max_down_idx = lows_atr.idxmin()
        
        return {
            'max_up_atr': float(highs_atr.max()),
            'max_down_atr': float(lows_atr.min()),
            'time_to_max_up_min': int(max_up_idx) + 1 if pd.notna(max_up_idx) else None,
            'time_to_max_down_min': int(max_down_idx) + 1 if pd.notna(max_down_idx) else None
        }
    
    def _empty_result(
        self,
        asset_id: int,
        time_1: datetime,
        atr: float,
        ref_price: float
    ) -> BarrierOutcomeResult:
        """Maak een leeg resultaat voor error cases."""
        return BarrierOutcomeResult(
            asset_id=asset_id,
            time_1=time_1,
            atr_at_signal=atr,
            reference_price=ref_price,
            max_observation_min=self.config.max_observation_min,
            first_significant_barrier="none"
        )
