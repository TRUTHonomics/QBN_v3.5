"""
ThresholdLoader - Database-driven threshold configuration voor QBN v3.

Laadt composite en alignment thresholds uit qbn.composite_threshold_config.
Vervangt hardcoded waarden in network_config.py.

ARCHITECTUUR:
- Thresholds zijn asset- en horizon-specifiek
- Fallback naar defaults als database entry ontbreekt
- Logging van geladen waarden voor traceerbaarheid

USAGE:
    from config.threshold_loader import ThresholdLoader
    
    # Per-horizon thresholds laden
    loader = ThresholdLoader(asset_id=1, horizon='1h')
    neutral_band = loader.composite_neutral_band
    strong_threshold = loader.composite_strong_threshold
    
    # Of met factory method
    loaders = ThresholdLoader.load_all_horizons(asset_id=1)
    for horizon, loader in loaders.items():
        print(f"{horizon}: {loader.composite_strong_threshold}")
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from decimal import Decimal

# REASON: Import centrale defaults en warning utility
from core.config_defaults import (
    DEFAULT_COMPOSITE_NEUTRAL_BAND,
    DEFAULT_COMPOSITE_STRONG_THRESHOLD,
    DEFAULT_ALIGNMENT_HIGH_THRESHOLD,
    DEFAULT_ALIGNMENT_LOW_THRESHOLD
)
from core.config_warnings import warn_fallback_active

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Container voor threshold waarden per semantic class."""
    # Composite thresholds (neutral_band, strong_threshold)
    leading_composite: Dict[str, float]
    coincident_composite: Dict[str, float]
    confirming_composite: Dict[str, float]
    
    # Alignment thresholds (high_threshold, low_threshold)
    leading_alignment: Dict[str, float]
    coincident_alignment: Dict[str, float]
    confirming_alignment: Dict[str, float]
    
    source: str  # 'database' of 'fallback'
    
    # REASON: Backwards compatibility helpers
    @property
    def composite_neutral_band(self) -> float:
        return self.leading_composite.get('neutral_band', DEFAULT_COMPOSITE_NEUTRAL_BAND)
    
    @property
    def composite_strong_threshold(self) -> float:
        return self.leading_composite.get('strong_threshold', DEFAULT_COMPOSITE_STRONG_THRESHOLD)
    
    @property
    def alignment_high_threshold(self) -> float:
        return self.leading_alignment.get('high_threshold', DEFAULT_ALIGNMENT_HIGH_THRESHOLD)
    
    @property
    def alignment_low_threshold(self) -> float:
        return self.leading_alignment.get('low_threshold', DEFAULT_ALIGNMENT_LOW_THRESHOLD)


class ThresholdLoader:
    """
    Laadt threshold configuratie uit qbn.composite_threshold_config.
    
    De tabel heeft de volgende structuur:
    - asset_id: int
    - horizon: str ('1h', '4h', '1d')
    - config_type: str ('leading_composite', 'coincident_composite', etc.)
    - param_name: str ('neutral_band', 'strong_threshold', 'high_threshold', 'low_threshold')
    - param_value: decimal
    - source_method: str ('MI Grid Search', 'CART Analysis', etc.)
    """
    
    def __init__(self, asset_id: int, horizon: str):
        """
        Initialiseer ThresholdLoader voor een specifiek asset en horizon.
        
        Args:
            asset_id: Asset ID waarvoor thresholds worden geladen
            horizon: Tijdshorizon ('1h', '4h', '1d')
        """
        self.asset_id = asset_id
        self.horizon = horizon
        self._config: Optional[ThresholdConfig] = None
        self._load_from_db()
    
    def _load_from_db(self):
        """Laad thresholds uit qbn.composite_threshold_config."""
        try:
            from database.db import get_cursor
            
            with get_cursor() as cur:
                # REASON: Gebruik DISTINCT ON om alleen de LAATSTE waarden per parameter op te halen.
                # Zonder dit kunnen oude runs met foute defaults (zoals 0.5) de boel vervuilen.
                cur.execute("""
                    SELECT DISTINCT ON (config_type, param_name) 
                        config_type, param_name, param_value, source_method
                    FROM qbn.composite_threshold_config
                    WHERE asset_id = %s AND horizon = %s
                    ORDER BY config_type, param_name, updated_at DESC
                """, (self.asset_id, self.horizon))
                
                rows = cur.fetchall()
            
            if not rows:
                self._use_fallback_defaults()
                return
            
            # Parse resultaten in geneste structuur
            data: Dict[str, Dict[str, float]] = {
                'leading_composite': {
                    'neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                    # v3.5: asymmetrisch (bull/bear)
                    'bullish_neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'bullish_strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                    'bearish_neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'bearish_strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                },
                'coincident_composite': {
                    'neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                    'bullish_neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'bullish_strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                    'bearish_neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'bearish_strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                },
                'confirming_composite': {
                    'neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                    'bullish_neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'bullish_strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                    'bearish_neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                    'bearish_strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                },
                'leading_alignment': {'high_threshold': DEFAULT_ALIGNMENT_HIGH_THRESHOLD, 'low_threshold': DEFAULT_ALIGNMENT_LOW_THRESHOLD},
                'coincident_alignment': {'high_threshold': DEFAULT_ALIGNMENT_HIGH_THRESHOLD, 'low_threshold': DEFAULT_ALIGNMENT_LOW_THRESHOLD},
                'confirming_alignment': {'high_threshold': DEFAULT_ALIGNMENT_HIGH_THRESHOLD, 'low_threshold': DEFAULT_ALIGNMENT_LOW_THRESHOLD}
            }
            
            # REASON: Support legacy 'composite' and 'alignment' keys for backward compatibility
            legacy_mapping = {
                'composite': ['leading_composite', 'coincident_composite', 'confirming_composite'],
                'alignment': ['leading_alignment', 'coincident_alignment', 'confirming_alignment']
            }
            
            sources: set = set()
            seen_params: Dict[str, set] = {}
            
            for config_type, param_name, param_value, source_method in rows:
                val = float(param_value) if isinstance(param_value, Decimal) else param_value
                if source_method:
                    sources.add(source_method)
                seen_params.setdefault(config_type, set()).add(param_name)
                
                if config_type in data:
                    data[config_type][param_name] = val
                elif config_type in legacy_mapping:
                    # Vul alle targets met de legacy waarde
                    for target_type in legacy_mapping[config_type]:
                        data[target_type][param_name] = val

            # v3.5 compat: als alleen legacy (symmetrisch) aanwezig is, spiegel naar bull/bear keys
            for comp_key in ['leading_composite', 'coincident_composite', 'confirming_composite']:
                nb = data[comp_key].get('neutral_band', DEFAULT_COMPOSITE_NEUTRAL_BAND)
                st = data[comp_key].get('strong_threshold', DEFAULT_COMPOSITE_STRONG_THRESHOLD)
                # Alleen overschrijven als deze params niet expliciet in DB stonden
                if 'bullish_neutral_band' not in seen_params.get(comp_key, set()):
                    data[comp_key]['bullish_neutral_band'] = nb
                if 'bearish_neutral_band' not in seen_params.get(comp_key, set()):
                    data[comp_key]['bearish_neutral_band'] = nb
                if 'bullish_strong_threshold' not in seen_params.get(comp_key, set()):
                    data[comp_key]['bullish_strong_threshold'] = st
                if 'bearish_strong_threshold' not in seen_params.get(comp_key, set()):
                    data[comp_key]['bearish_strong_threshold'] = st
            
            self._config = ThresholdConfig(
                leading_composite=data['leading_composite'],
                coincident_composite=data['coincident_composite'],
                confirming_composite=data['confirming_composite'],
                leading_alignment=data['leading_alignment'],
                coincident_alignment=data['coincident_alignment'],
                confirming_alignment=data['confirming_alignment'],
                source=f"database ({', '.join(sources)})" if sources else "database"
            )
            
            logger.info(
                f"✅ Granulaire thresholds geladen voor asset {self.asset_id}, horizon {self.horizon} "
                f"(bron: {self._config.source})"
            )
            
        except Exception as e:
            logger.error(f"❌ Kritieke fout bij laden thresholds: {e}")
            self._use_fallback_defaults()
    
    def _use_fallback_defaults(self):
        """Stel fallback defaults in met DUIDELIJKE WARNING."""
        fallback_vals = {
            'neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
            'strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
            'high_threshold': DEFAULT_ALIGNMENT_HIGH_THRESHOLD,
            'low_threshold': DEFAULT_ALIGNMENT_LOW_THRESHOLD
        }
        
        warn_fallback_active(
            component="ThresholdLoader",
            config_name=f"asset_{self.asset_id}_{self.horizon}",
            fallback_values=fallback_vals,
            reason="Geen thresholds gevonden in qbn.composite_threshold_config",
            fix_command="Draai menu optie 7 (Threshold Optimalisatie)"
        )
        
        def_comp = {
            'neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
            'strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
            'bullish_neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
            'bullish_strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
            'bearish_neutral_band': DEFAULT_COMPOSITE_NEUTRAL_BAND,
            'bearish_strong_threshold': DEFAULT_COMPOSITE_STRONG_THRESHOLD,
        }
        def_align = {'high_threshold': DEFAULT_ALIGNMENT_HIGH_THRESHOLD, 'low_threshold': DEFAULT_ALIGNMENT_LOW_THRESHOLD}
        
        self._config = ThresholdConfig(
            leading_composite=def_comp.copy(),
            coincident_composite=def_comp.copy(),
            confirming_composite=def_comp.copy(),
            leading_alignment=def_align.copy(),
            coincident_alignment=def_align.copy(),
            confirming_alignment=def_align.copy(),
            source="fallback"
        )
    
    # =========================================================================
    # Granulaire toegangsmethoden
    # =========================================================================
    
    @property
    def composite_neutral_band(self) -> float:
        """Shortcut naar leading composite neutral band."""
        return self._config.composite_neutral_band

    @property
    def composite_strong_threshold(self) -> float:
        """Shortcut naar leading composite strong threshold."""
        return self._config.composite_strong_threshold

    @property
    def alignment_high_threshold(self) -> float:
        """Shortcut naar leading alignment high threshold."""
        return self._config.alignment_high_threshold

    @property
    def alignment_low_threshold(self) -> float:
        """Shortcut naar leading alignment low threshold."""
        return self._config.alignment_low_threshold
    def get_composite_params(self, semantic_class: str) -> Dict[str, float]:
        """Haal (neutral_band, strong_threshold) voor een specifieke laag."""
        key = f"{semantic_class.lower()}_composite"
        return getattr(self._config, key, self._config.leading_composite)
    
    def get_alignment_params(self, semantic_class: str) -> Dict[str, float]:
        """Haal (high_threshold, low_threshold) voor een specifieke laag."""
        key = f"{semantic_class.lower()}_alignment"
        return getattr(self._config, key, self._config.leading_alignment)
    
    @property
    def source(self) -> str:
        """Bron van de geladen thresholds ('database' of 'fallback')."""
        return self._config.source
    
    @property
    def is_from_database(self) -> bool:
        """True als thresholds uit de database komen."""
        return self._config.source.startswith("database")
    
    # =========================================================================
    # Composite thresholds als dict (voor SignalAggregator compatibiliteit)
    # =========================================================================
    
    def get_composite_thresholds(self) -> Dict[str, float]:
        """
        Retourneer thresholds als dict voor SignalAggregator.
        
        Returns:
            Dict met threshold waarden voor composite state mapping.
        """
        # v3.5: asymmetrische thresholds. Neutral band is niet langer symmetrisch per definitie.
        params = self._config.leading_composite
        bull_nb = float(params.get('bullish_neutral_band', params.get('neutral_band', DEFAULT_COMPOSITE_NEUTRAL_BAND)))
        bear_nb = float(params.get('bearish_neutral_band', params.get('neutral_band', DEFAULT_COMPOSITE_NEUTRAL_BAND)))
        bull_st = float(params.get('bullish_strong_threshold', params.get('strong_threshold', DEFAULT_COMPOSITE_STRONG_THRESHOLD)))
        bear_st = float(params.get('bearish_strong_threshold', params.get('strong_threshold', DEFAULT_COMPOSITE_STRONG_THRESHOLD)))
        
        return {
            'strong_bearish': -bear_st,
            'bearish': -bear_nb,
            'neutral_low': -bear_nb,
            'neutral_high': bull_nb,
            'bullish': bull_nb,
            'strong_bullish': bull_st
        }
    
    # =========================================================================
    # Factory methods
    # =========================================================================
    
    @classmethod
    def load_all_horizons(cls, asset_id: int) -> Dict[str, 'ThresholdLoader']:
        """
        Laad thresholds voor alle standaard horizons.
        
        Args:
            asset_id: Asset ID waarvoor thresholds worden geladen
            
        Returns:
            Dict met horizon als key en ThresholdLoader als value
        """
        horizons = ['1h', '4h', '1d']
        return {h: cls(asset_id, h) for h in horizons}
    
    @classmethod
    def check_database_availability(cls, asset_id: int = 1) -> Dict[str, Any]:
        """
        Check of composite_threshold_config tabel data bevat.
        
        Args:
            asset_id: Asset ID om te checken (default: 1)
            
        Returns:
            Dict met status informatie
        """
        try:
            from database.db import get_cursor
            
            with get_cursor() as cur:
                # Check totaal aantal entries
                cur.execute("SELECT COUNT(*) FROM qbn.composite_threshold_config")
                total_count = cur.fetchone()[0]
                
                # Check entries voor specifiek asset
                cur.execute(
                    "SELECT COUNT(*) FROM qbn.composite_threshold_config WHERE asset_id = %s",
                    (asset_id,)
                )
                asset_count = cur.fetchone()[0]
                
                # Check horizons voor asset
                cur.execute(
                    "SELECT DISTINCT horizon FROM qbn.composite_threshold_config WHERE asset_id = %s",
                    (asset_id,)
                )
                horizons = [row[0] for row in cur.fetchall()]
            
            return {
                'available': total_count > 0,
                'total_entries': total_count,
                'asset_id': asset_id,
                'asset_entries': asset_count,
                'horizons': horizons,
                'status': 'ok' if asset_count >= 4 else 'incomplete'  # 4 = min params per horizon
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'status': 'error'
            }
    
    def __repr__(self) -> str:
        return (
            f"ThresholdLoader(asset_id={self.asset_id}, horizon='{self.horizon}', "
            f"neutral_band={self.composite_neutral_band:.4f}, "
            f"strong_threshold={self.composite_strong_threshold:.4f}, "
            f"source='{self.source}')"
        )

