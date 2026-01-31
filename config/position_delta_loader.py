"""
PositionDeltaThresholdLoader - Database-driven delta threshold configuration voor QBN v3.2.

Laadt delta thresholds voor Position_Confidence en Position_Prediction uit 
qbn.position_delta_threshold_config.

ARCHITECTUUR:
- Delta thresholds zijn asset-specifiek
- Fallback naar defaults als database entry ontbreekt
- Logging van geladen waarden voor traceerbaarheid

DELTA STATES:
- deteriorating: delta < -threshold (situatie verslechtert voor positie)
- stable: -threshold <= delta <= +threshold
- improving: delta > +threshold (situatie verbetert voor positie)

USAGE:
    from config.position_delta_loader import PositionDeltaThresholdLoader
    
    loader = PositionDeltaThresholdLoader(asset_id=1)
    coinc_thresh = loader.get_threshold('cumulative', 'coincident')
    conf_thresh = loader.get_threshold('cumulative', 'confirming')
    
    # Of direct discretiseren
    state = loader.discretize_delta(delta_value, 'cumulative', 'coincident')
"""

import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal

from core.config_defaults import (
    DEFAULT_DELTA_THRESHOLD_COINCIDENT,
    DEFAULT_DELTA_THRESHOLD_CONFIRMING
)
from core.config_warnings import warn_fallback_active

logger = logging.getLogger(__name__)


# Delta states voor discretisatie
DELTA_STATES = ['deteriorating', 'stable', 'improving']


@dataclass
class DeltaThresholdConfig:
    """Container voor delta threshold waarden."""
    cumulative_coincident: float
    cumulative_confirming: float
    instantaneous_coincident: float
    instantaneous_confirming: float
    source: str  # 'database' of 'fallback'
    mi_scores: Dict[str, float]  # MI scores per threshold


class PositionDeltaThresholdLoader:
    """
    Laadt delta threshold configuratie uit qbn.position_delta_threshold_config.
    
    De tabel heeft de volgende structuur:
    - asset_id: int
    - delta_type: str ('cumulative', 'instantaneous')
    - score_type: str ('coincident', 'confirming')
    - threshold: decimal
    - mi_score: decimal (optioneel)
    - source_method: str ('MI Grid Search', 'Manual', etc.)
    """
    
    def __init__(self, asset_id: int):
        """
        Initialiseer PositionDeltaThresholdLoader voor een specifiek asset.
        
        Args:
            asset_id: Asset ID waarvoor thresholds worden geladen
        """
        self.asset_id = asset_id
        self._config: Optional[DeltaThresholdConfig] = None
        self._load_from_db()
    
    def _load_from_db(self):
        """Laad thresholds uit qbn.position_delta_threshold_config."""
        try:
            from database.db import get_cursor
            
            with get_cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ON (delta_type, score_type) 
                        delta_type, score_type, threshold, mi_score, source_method
                    FROM qbn.position_delta_threshold_config
                    WHERE asset_id = %s
                    ORDER BY delta_type, score_type, updated_at DESC
                """, (self.asset_id,))
                
                rows = cur.fetchall()
            
            if not rows:
                self._use_fallback_defaults()
                return
            
            # Parse resultaten
            data = {
                ('cumulative', 'coincident'): DEFAULT_DELTA_THRESHOLD_COINCIDENT,
                ('cumulative', 'confirming'): DEFAULT_DELTA_THRESHOLD_CONFIRMING,
                ('instantaneous', 'coincident'): DEFAULT_DELTA_THRESHOLD_COINCIDENT,
                ('instantaneous', 'confirming'): DEFAULT_DELTA_THRESHOLD_CONFIRMING,
            }
            mi_scores = {}
            sources = set()
            
            for delta_type, score_type, threshold, mi_score, source_method in rows:
                val = float(threshold) if isinstance(threshold, Decimal) else threshold
                data[(delta_type, score_type)] = val
                
                if mi_score is not None:
                    mi_key = f"{delta_type}_{score_type}"
                    mi_scores[mi_key] = float(mi_score) if isinstance(mi_score, Decimal) else mi_score
                
                if source_method:
                    sources.add(source_method)
            
            self._config = DeltaThresholdConfig(
                cumulative_coincident=data[('cumulative', 'coincident')],
                cumulative_confirming=data[('cumulative', 'confirming')],
                instantaneous_coincident=data[('instantaneous', 'coincident')],
                instantaneous_confirming=data[('instantaneous', 'confirming')],
                source=f"database ({', '.join(sources)})" if sources else "database",
                mi_scores=mi_scores
            )
            
            logger.info(
                f"✅ Delta thresholds geladen voor asset {self.asset_id} "
                f"(coinc={self._config.cumulative_coincident:.3f}, "
                f"conf={self._config.cumulative_confirming:.3f}, "
                f"bron: {self._config.source})"
            )
            
        except Exception as e:
            logger.error(f"❌ Fout bij laden delta thresholds: {e}")
            self._use_fallback_defaults()
    
    def _use_fallback_defaults(self):
        """Stel fallback defaults in met DUIDELIJKE WARNING."""
        fallback_vals = {
            'cumulative_coincident': DEFAULT_DELTA_THRESHOLD_COINCIDENT,
            'cumulative_confirming': DEFAULT_DELTA_THRESHOLD_CONFIRMING,
        }
        
        warn_fallback_active(
            component="PositionDeltaThresholdLoader",
            config_name=f"asset_{self.asset_id}_delta_thresholds",
            fallback_values=fallback_vals,
            reason="Geen delta thresholds gevonden in qbn.position_delta_threshold_config",
            fix_command="Draai 'Position Delta Threshold Analyse' in training menu"
        )
        
        self._config = DeltaThresholdConfig(
            cumulative_coincident=DEFAULT_DELTA_THRESHOLD_COINCIDENT,
            cumulative_confirming=DEFAULT_DELTA_THRESHOLD_CONFIRMING,
            instantaneous_coincident=DEFAULT_DELTA_THRESHOLD_COINCIDENT,
            instantaneous_confirming=DEFAULT_DELTA_THRESHOLD_CONFIRMING,
            source="fallback",
            mi_scores={}
        )
    
    # =========================================================================
    # Threshold Access Methods
    # =========================================================================
    
    def get_threshold(self, delta_type: str, score_type: str) -> float:
        """
        Haal threshold voor specifieke delta/score combinatie.
        
        Args:
            delta_type: 'cumulative' of 'instantaneous'
            score_type: 'coincident' of 'confirming'
            
        Returns:
            Threshold waarde
        """
        key = f"{delta_type}_{score_type}"
        return getattr(self._config, key, DEFAULT_DELTA_THRESHOLD_COINCIDENT)
    
    @property
    def cumulative_coincident_threshold(self) -> float:
        """Threshold voor cumulatieve coincident delta."""
        return self._config.cumulative_coincident
    
    @property
    def cumulative_confirming_threshold(self) -> float:
        """Threshold voor cumulatieve confirming delta."""
        return self._config.cumulative_confirming
    
    @property
    def source(self) -> str:
        """Bron van de geladen thresholds ('database' of 'fallback')."""
        return self._config.source
    
    @property
    def is_from_database(self) -> bool:
        """True als thresholds uit de database komen."""
        return self._config.source.startswith("database")
    
    # =========================================================================
    # Discretization Methods
    # =========================================================================
    
    def discretize_delta(
        self, 
        delta: float, 
        delta_type: str = 'cumulative',
        score_type: str = 'coincident'
    ) -> str:
        """
        Discretiseer een delta waarde naar state.
        
        Args:
            delta: Delta waarde (direction-aware, positief = gunstig)
            delta_type: 'cumulative' of 'instantaneous'
            score_type: 'coincident' of 'confirming'
            
        Returns:
            State: 'deteriorating', 'stable', of 'improving'
        """
        threshold = self.get_threshold(delta_type, score_type)
        
        if delta < -threshold:
            return 'deteriorating'
        elif delta > threshold:
            return 'improving'
        return 'stable'
    
    def discretize_deltas(
        self,
        delta_coinc: float,
        delta_conf: float,
        delta_type: str = 'cumulative'
    ) -> Tuple[str, str]:
        """
        Discretiseer beide delta waarden tegelijk.
        
        Args:
            delta_coinc: Coincident delta waarde
            delta_conf: Confirming delta waarde
            delta_type: 'cumulative' of 'instantaneous'
            
        Returns:
            Tuple van (coinc_state, conf_state)
        """
        coinc_state = self.discretize_delta(delta_coinc, delta_type, 'coincident')
        conf_state = self.discretize_delta(delta_conf, delta_type, 'confirming')
        return coinc_state, conf_state
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def check_database_availability(cls, asset_id: int = 1) -> Dict[str, Any]:
        """
        Check of position_delta_threshold_config tabel data bevat.
        
        Args:
            asset_id: Asset ID om te checken (default: 1)
            
        Returns:
            Dict met status informatie
        """
        try:
            from database.db import get_cursor
            
            with get_cursor() as cur:
                # Check totaal aantal entries
                cur.execute("SELECT COUNT(*) FROM qbn.position_delta_threshold_config")
                total_count = cur.fetchone()[0]
                
                # Check entries voor specifiek asset
                cur.execute(
                    "SELECT COUNT(*) FROM qbn.position_delta_threshold_config WHERE asset_id = %s",
                    (asset_id,)
                )
                asset_count = cur.fetchone()[0]
                
                # Check welke delta/score types aanwezig zijn
                cur.execute(
                    """SELECT DISTINCT delta_type, score_type 
                       FROM qbn.position_delta_threshold_config 
                       WHERE asset_id = %s""",
                    (asset_id,)
                )
                types = [(row[0], row[1]) for row in cur.fetchall()]
            
            return {
                'available': total_count > 0,
                'total_entries': total_count,
                'asset_id': asset_id,
                'asset_entries': asset_count,
                'types': types,
                'status': 'ok' if asset_count >= 2 else 'incomplete'  # 2 = min (coinc + conf)
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'status': 'error'
            }
    
    def __repr__(self) -> str:
        return (
            f"PositionDeltaThresholdLoader(asset_id={self.asset_id}, "
            f"coinc={self._config.cumulative_coincident:.4f}, "
            f"conf={self._config.cumulative_confirming:.4f}, "
            f"source='{self.source}')"
        )
