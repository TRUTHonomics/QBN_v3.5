
#!/usr/bin/env python3
"""
OutcomeTargetGenerator - Berekent koersuitkomsten voor CPT training

ARCHITECTUUR NOOT:
- Outcomes worden berekend uit kfl.klines_raw
- Discretisatie naar 7 ATR-relatieve bins (-3 tot +3)
- Drie horizons: 1h, 4h, 1d

TARGET STATES:
  -3: Strong_Bearish  (return < -1.25 * ATR)
  -2: Bearish         (-1.25*ATR <= return < -0.75*ATR)
  -1: Slight_Bearish  (-0.75*ATR <= return < -0.25*ATR)
   0: Neutral         (-0.25*ATR <= return < +0.25*ATR)
  +1: Slight_Bullish  (+0.25*ATR <= return < +0.75*ATR)
  +2: Bullish         (+0.75*ATR <= return < +1.25*ATR)
  +3: Strong_Bullish  (return >= +1.25*ATR)
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from database.db import get_cursor

logger = logging.getLogger(__name__)


# ATR bin thresholds (in ATR units)
ATR_THRESHOLDS = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]

# Horizon definitions in minutes
HORIZONS = {
    '1h': 60,
    '4h': 240,
    '1d': 1440
}


class OutcomeTargetGenerator:
    """
    Berekent koersuitkomsten voor CPT training.
    
    Gebruikt kfl.klines_raw om returns te berekenen en discretiseert
    naar 7 ATR-relatieve bins voor robuuste probabilistische voorspellingen.
    """
    
    def __init__(self):
        """Initialize OutcomeTargetGenerator."""
        logger.info("OutcomeTargetGenerator initialized")
    
    def discretize_return_to_atr_bins(self, return_pct: float, atr_pct: float) -> int:
        """
        Discretiseer return naar 7 ATR-relatieve bins.
        
        Args:
            return_pct: Return in percentage (bijv. 1.5 voor +1.5%)
            atr_pct: ATR in percentage (bijv. 2.0 voor 2%)
            
        Returns:
            State ID van -3 tot +3
        """
        if atr_pct is None or atr_pct <= 0:
            # REASON: Fallback naar Neutral bij ontbrekende ATR
            return 0
        
        # Normaliseer return naar ATR units
        atr_normalized = return_pct / atr_pct
        
        # REASON: Gebruik dynamische thresholds voor flexibiliteit
        # EXPL: We loopen door de thresholds om de juiste bin te bepalen.
        # [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]
        # < -1.25 -> -3
        # < -0.75 -> -2
        # < -0.25 -> -1
        # < 0.25  -> 0
        # < 0.75  -> 1
        # < 1.25  -> 2
        # >= 1.25 -> 3
        
        if atr_normalized < ATR_THRESHOLDS[0]:
            return -3
        elif atr_normalized < ATR_THRESHOLDS[1]:
            return -2
        elif atr_normalized < ATR_THRESHOLDS[2]:
            return -1
        elif atr_normalized < ATR_THRESHOLDS[3]:
            return 0
        elif atr_normalized < ATR_THRESHOLDS[4]:
            return 1
        elif atr_normalized < ATR_THRESHOLDS[5]:
            return 2
        else:
            return 3
    
    def state_id_to_name(self, state_id: int) -> str:
        """Convert state ID naar naam."""
        names = {
            -3: 'Strong_Bearish',
            -2: 'Bearish',
            -1: 'Slight_Bearish',
            0: 'Neutral',
            1: 'Slight_Bullish',
            2: 'Bullish',
            3: 'Strong_Bullish'
        }
        return names.get(state_id, 'Unknown')
    
    def state_id_to_atr_midpoint(self, state_id: int) -> float:
        """
        Get midpoint of ATR bin for expected value calculation.
        
        Returns:
            Midpoint in ATR units
        """
        midpoints = {
            -3: -2.5,   # Strong_Bearish: assume -2.5 ATR
            -2: -1.5,   # Bearish
            -1: -0.75,  # Slight_Bearish
            0: 0.0,     # Neutral
            1: 0.75,    # Slight_Bullish
            2: 1.5,     # Bullish
            3: 2.5      # Strong_Bullish: assume +2.5 ATR
        }
        return midpoints.get(state_id, 0.0)
    
    def calculate_expected_atr_move(self, probabilities: Dict[int, float]) -> float:
        """
        Bereken verwachte ATR beweging uit probability distribution.
        
        Args:
            probabilities: Dict van state_id -> probability
            
        Returns:
            Expected ATR move (gewogen gemiddelde)
        """
        expected = 0.0
        for state_id, prob in probabilities.items():
            expected += prob * self.state_id_to_atr_midpoint(state_id)
        return expected
    
    def calculate_outcomes_for_asset(self, asset_id: int, 
                                     start_time: Optional[datetime] = None,
                                     end_time: Optional[datetime] = None,
                                     batch_size: int = 10000) -> int:
        """
        [DISABLED] Bereken en update outcomes voor een asset in qbn.signal_outcomes.
        
        REASON: qbn.signal_outcomes is verwijderd in v3.1. Gebruik qbn.barrier_outcomes.
        """
        logger.warning(f"calculate_outcomes_for_asset called for asset {asset_id}, but qbn.signal_outcomes is deprecated/removed.")
        return 0
    
    def _calculate_horizon_outcomes(self, asset_id: int, 
                                    horizon_name: str,
                                    horizon_minutes: int,
                                    start_time: Optional[datetime],
                                    end_time: Optional[datetime],
                                    batch_size: int) -> int:
        """
        [DISABLED] Bereken outcomes voor specifieke horizon.
        """
        return 0
    
    def calculate_atr_for_signals(self, asset_id: int, 
                                  batch_size: int = 10000) -> int:
        """
        [DISABLED] Bereken ATR voor signals waar deze ontbreekt.
        """
        logger.warning(f"calculate_atr_for_signals called for asset {asset_id}, but qbn.signal_outcomes is deprecated/removed.")
        return 0
    
    def verify_training_data_ready(self, asset_id: int) -> Dict[str, Any]:
        """
        Check of historische data klaar is voor CPT training.
        
        REASON: We filteren op 60m boundaries (via join met kfl.indicators) 
        omdat dit de dataset is die daadwerkelijk voor training wordt gebruikt.
        """
        # REASON: In v3.1 gebruiken we uitsluitend qbn.barrier_outcomes.
        # qbn.signal_outcomes is legacy en wordt niet meer ondersteund.
        with get_cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'qbn' AND table_name = 'barrier_outcomes')")
            has_barrier = cur.fetchone()[0]

        if not has_barrier:
            return {
                'ready': False,
                'total_rows': 0,
                'recommendation': 'no_tables_found',
                'error': 'Table qbn.barrier_outcomes not found'
            }

        # v3.1 Barrier-based status (Preferred)
        query = """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(CASE WHEN bo.first_significant_time_min <= 60 OR bo.time_1 <= NOW() - INTERVAL '60 minutes' THEN 1 END) as ready_1h,
            COUNT(CASE WHEN bo.first_significant_time_min <= 240 OR bo.time_1 <= NOW() - INTERVAL '240 minutes' THEN 1 END) as ready_4h,
            COUNT(CASE WHEN bo.first_significant_time_min <= 1440 OR bo.time_1 <= NOW() - INTERVAL '1440 minutes' THEN 1 END) as ready_1d,
            COUNT(bo.atr_at_signal) as rows_with_atr
        FROM kfl.mtf_signals_lead mtf
        JOIN kfl.indicators ind 
            ON ind.asset_id = mtf.asset_id 
            AND ind.time = mtf.time_1
            AND ind.interval_min = '60'
        LEFT JOIN qbn.barrier_outcomes bo
            ON bo.asset_id = mtf.asset_id
            AND bo.time_1 = mtf.time_1
        WHERE mtf.asset_id = %s
        """
        
        try:
            with get_cursor() as cur:
                cur.execute(query, (asset_id,))
                row = cur.fetchone()
            
            if not row or row[0] == 0:
                return {
                    'ready': False,
                    'total_rows': 0,
                    'recommendation': 'no_data'
                }
            
            total = row[0]
            
            result = {
                'ready': True,
                'total_rows': total,
                'rows_with_outcome_1h': row[1],
                'rows_with_outcome_4h': row[2],
                'rows_with_outcome_1d': row[3],
                'rows_with_atr_1m': row[4],
                'coverage_1h': row[1] / total if total > 0 else 0,
                'coverage_4h': row[2] / total if total > 0 else 0,
                'coverage_1d': row[3] / total if total > 0 else 0,
                'coverage_atr_1h': row[4] / total if total > 0 else 0,
                'coverage_atr_4h': row[4] / total if total > 0 else 0,
                'coverage_atr_1d': row[4] / total if total > 0 else 0,
                'table_used': 'barrier_outcomes'
            }
            
            # Determine recommendation
            min_coverage = min(result['coverage_1h'], result['coverage_4h'], result['coverage_1d'])
            
            if min_coverage >= 0.95:
                result['recommendation'] = 'ready'
            elif min_coverage >= 0.5:
                result['recommendation'] = 'run_backfill'
                result['ready'] = False
            else:
                result['recommendation'] = 'wait_for_horizon'
                result['ready'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying training data for asset {asset_id}: {e}")
            return {
                'ready': False,
                'total_rows': 0,
                'error': str(e),
                'recommendation': 'error'
            }
    
    def run_full_backfill(self, asset_ids: Optional[List[int]] = None,
                         max_iterations: int = 100) -> Dict[str, int]:
        """
        Run volledige outcome backfill voor alle assets.
        
        Args:
            asset_ids: Specifieke assets (None = alle met MTF data)
            max_iterations: Maximum iteraties per asset (voor batching)
            
        Returns:
            Dict met statistieken
        """
        if asset_ids is None:
            # Haal alle assets op met MTF data
            with get_cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT asset_id 
                    FROM kfl.mtf_signals_lead 
                    ORDER BY asset_id
                """)
                asset_ids = [row[0] for row in cur.fetchall()]
        
        logger.info(f"Running full outcome backfill for {len(asset_ids)} assets")
        
        stats = {
            'total_assets': len(asset_ids),
            'total_outcomes_updated': 0,
            'total_atr_updated': 0,
            'assets_completed': 0
        }
        
        for asset_id in asset_ids:
            # First update ATR where missing
            atr_updated = 0
            for _ in range(max_iterations):
                updated = self.calculate_atr_for_signals(asset_id)
                if updated == 0:
                    break
                atr_updated += updated
            
            # Then calculate outcomes
            outcomes_updated = 0
            for _ in range(max_iterations):
                updated = self.calculate_outcomes_for_asset(asset_id)
                if updated == 0:
                    break
                outcomes_updated += updated
            
            stats['total_atr_updated'] += atr_updated
            stats['total_outcomes_updated'] += outcomes_updated
            stats['assets_completed'] += 1
            
            logger.info(f"Asset {asset_id}: ATR={atr_updated}, Outcomes={outcomes_updated}")
        
        return stats


def create_target_generator() -> OutcomeTargetGenerator:
    """Factory function voor OutcomeTargetGenerator."""
    return OutcomeTargetGenerator()
