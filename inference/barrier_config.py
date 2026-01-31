from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class BarrierConfig:
    """Configuratie voor barrier berekeningen."""
    
    # Barrier levels in ATR units
    up_barriers: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
    )
    down_barriers: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
    )
    
    # Threshold voor "significant" barrier
    significant_threshold: float = 0.75
    
    # Maximale observation window in minuten
    max_observation_min: int = 1440  # 24 uur (voorheen 48u)
    
    # Configuratie naam (voor database reference)
    config_name: str = "default"
    
    def __post_init__(self):
        """Valideer configuratie na initialisatie."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Valideer configuratie parameters.
        
        Raises:
            ValueError: Bij ongeldige configuratie
        """
        if not self.up_barriers:
            raise ValueError("up_barriers mag niet leeg zijn")
        if not self.down_barriers:
            raise ValueError("down_barriers mag niet leeg zijn")
        if not all(b > 0 for b in self.up_barriers):
            raise ValueError("Alle up_barriers moeten positief zijn")
        if not all(b > 0 for b in self.down_barriers):
            raise ValueError("Alle down_barriers moeten positief zijn")
        if self.significant_threshold <= 0:
            raise ValueError("significant_threshold moet positief zijn")
        if self.max_observation_min < 60:
            raise ValueError("max_observation_min moet minimaal 60 zijn")
        return True
    
    @classmethod
    def from_database(cls, config_name: str = "default") -> 'BarrierConfig':
        """
        Laad configuratie uit qbn.barrier_config.
        
        Args:
            config_name: Naam van de configuratie
            
        Returns:
            BarrierConfig instance
        """
        from database.db import get_cursor
        
        with get_cursor() as cur:
            cur.execute("""
                SELECT up_barriers, down_barriers, significant_threshold,
                       max_observation_min, config_name
                FROM qbn.barrier_config
                WHERE config_name = %s AND is_active = TRUE
            """, (config_name,))
            row = cur.fetchone()
            
        if not row:
            raise ValueError(f"Config '{config_name}' niet gevonden of niet actief")
        
        return cls(
            up_barriers=list(row[0]),
            down_barriers=list(row[1]),
            significant_threshold=float(row[2]),
            max_observation_min=int(row[3]),
            config_name=row[4]
        )
    
    def save_to_database(self, notes: str = None):
        """Sla huidige configuratie op in de database."""
        from database.db import get_cursor
        
        query = """
            INSERT INTO qbn.barrier_config (
                config_name, up_barriers, down_barriers,
                significant_threshold, max_observation_min, notes
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (config_name) DO UPDATE SET
                up_barriers = EXCLUDED.up_barriers,
                down_barriers = EXCLUDED.down_barriers,
                significant_threshold = EXCLUDED.significant_threshold,
                max_observation_min = EXCLUDED.max_observation_min,
                notes = COALESCE(EXCLUDED.notes, qbn.barrier_config.notes),
                updated_at = NOW()
        """
        
        with get_cursor(commit=True) as cur:
            cur.execute(query, (
                self.config_name,
                self.up_barriers,
                self.down_barriers,
                self.significant_threshold,
                self.max_observation_min,
                notes
            ))
        logger.info(f"Saved BarrierConfig '{self.config_name}' to database")

    @classmethod
    def from_yaml(cls, path: str) -> 'BarrierConfig':
        """Laad configuratie uit YAML bestand."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


class BarrierConfigLoader:
    """Unified config loader met prioriteit: DB > YAML > Defaults."""
    
    # Pad relatief aan de root van het project (f:/Containers/QBN_v3)
    # De loader bevindt zich in inference/, dus we gaan een map omhoog
    DEFAULT_YAML_PATH = Path(__file__).parent.parent / 'config' / 'barrier_config.yaml'
    
    @classmethod
    def load(
        cls,
        config_name: str = 'default',
        asset_id: Optional[int] = None,
        prefer_database: bool = True
    ) -> 'BarrierConfig':
        """
        Laad configuratie met fallback chain.
        
        Args:
            config_name: Configuratie naam
            asset_id: Optioneel asset ID voor overrides
            prefer_database: Probeer eerst DB
        """
        config = None
        
        # 1. Probeer database
        if prefer_database:
            try:
                config = BarrierConfig.from_database(config_name)
                logger.debug(f"Loaded config '{config_name}' from database")
            except Exception as e:
                logger.warning(f"DB config load failed for '{config_name}': {e}")
        
        # 2. Fallback naar YAML
        if config is None:
            try:
                config = cls._load_from_yaml(config_name)
                logger.debug(f"Loaded config '{config_name}' from YAML")
            except Exception as e:
                logger.warning(f"YAML config load failed for '{config_name}': {e}")
        
        # 3. Fallback naar defaults
        if config is None:
            config = BarrierConfig(config_name=config_name)
            logger.info(f"Using default BarrierConfig for '{config_name}'")
        
        # 4. Apply asset overrides uit YAML (indien beschikbaar)
        if asset_id:
            config = cls._apply_asset_overrides(config, asset_id)
        
        return config
    
    @classmethod
    def _load_from_yaml(cls, config_name: str) -> 'BarrierConfig':
        """Laad specifieke config uit YAML."""
        if not cls.DEFAULT_YAML_PATH.exists():
            raise FileNotFoundError(f"YAML config not found: {cls.DEFAULT_YAML_PATH}")
        
        with open(cls.DEFAULT_YAML_PATH, 'r') as f:
            data = yaml.safe_load(f)
        
        if config_name not in data:
            raise KeyError(f"Config '{config_name}' not in YAML")
        
        cfg_data = data[config_name]
        return BarrierConfig(
            up_barriers=cfg_data.get('up_barriers', [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]),
            down_barriers=cfg_data.get('down_barriers', [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]),
            significant_threshold=cfg_data.get('significant_threshold', 0.75),
            max_observation_min=cfg_data.get('max_observation_min', 2880),
            config_name=config_name
        )
    
    @classmethod
    def _apply_asset_overrides(cls, config: 'BarrierConfig', asset_id: int) -> 'BarrierConfig':
        """Pas asset-specifieke overrides toe uit YAML."""
        if not cls.DEFAULT_YAML_PATH.exists():
            return config
            
        try:
            with open(cls.DEFAULT_YAML_PATH, 'r') as f:
                data = yaml.safe_load(f)
            
            overrides = data.get('asset_overrides', {}).get(asset_id, {})
            if overrides:
                # Als er een andere config wordt gespecificeerd voor dit asset, laad die eerst
                if 'config' in overrides and overrides['config'] != config.config_name:
                    config = cls._load_from_yaml(overrides['config'])
                
                # Pas specifieke velden aan
                if 'significant_threshold' in overrides:
                    config.significant_threshold = overrides['significant_threshold']
                
                logger.debug(f"Applied overrides for asset {asset_id}")
        except Exception as e:
            logger.warning(f"Failed to apply asset overrides for {asset_id}: {e}")
        
        return config
    
    @classmethod
    def sync_yaml_to_db(cls):
        """Synchroniseer alle YAML configs naar database."""
        if not cls.DEFAULT_YAML_PATH.exists():
            logger.error(f"Cannot sync: YAML config not found at {cls.DEFAULT_YAML_PATH}")
            return

        with open(cls.DEFAULT_YAML_PATH, 'r') as f:
            data = yaml.safe_load(f)
        
        synced_count = 0
        for name, cfg in data.items():
            if name == 'asset_overrides' or not isinstance(cfg, dict):
                continue
            
            try:
                barrier_cfg = BarrierConfig(
                    up_barriers=cfg.get('up_barriers', [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]),
                    down_barriers=cfg.get('down_barriers', [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]),
                    significant_threshold=cfg.get('significant_threshold', 0.75),
                    max_observation_min=cfg.get('max_observation_min', 2880),
                    config_name=name
                )
                barrier_cfg.save_to_database(notes="Synced from YAML")
                synced_count += 1
            except Exception as e:
                logger.error(f"Failed to sync config '{name}': {e}")
        
        logger.info(f"Synced {synced_count} configs to database")


def validate_barrier_config(config: BarrierConfig) -> List[str]:
    """
    Valideer config en return lijst van waarschuwingen (niet blokkerend).
    """
    warnings = []
    
    # Check barrier ranges
    if max(config.up_barriers) > 3.0:
        warnings.append("Up barriers > 3.0 ATR zijn extreem hoog")
    
    if config.significant_threshold not in config.up_barriers and \
       not any(abs(config.significant_threshold - b) < 0.001 for b in config.up_barriers):
        warnings.append(f"significant_threshold {config.significant_threshold} staat niet in up_barriers")
    
    # Check observation window
    if config.max_observation_min < 60:
        warnings.append("max_observation_min < 60 is zeer kort voor barrier analyse")
    
    if config.max_observation_min > 10080:  # 7 dagen
        warnings.append("max_observation_min > 7 dagen kan performance problemen geven")
    
    return warnings


@dataclass
class BarrierOutcomeResult:
    """Resultaat van barrier berekening voor één timestamp."""
    
    # Identifiers
    asset_id: int
    time_1: datetime
    
    # Context
    atr_at_signal: float
    reference_price: float
    max_observation_min: int
    
    # Tijd tot barriers (None = niet bereikt)
    time_to_up_barriers: Dict[str, Optional[int]] = field(default_factory=dict)
    time_to_down_barriers: Dict[str, Optional[int]] = field(default_factory=dict)
    
    # Extremen
    max_up_atr: float = 0.0
    max_down_atr: float = 0.0
    time_to_max_up_min: Optional[int] = None
    time_to_max_down_min: Optional[int] = None
    
    # First significant
    first_significant_barrier: str = "none"
    first_significant_time_min: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Converteer naar dictionary."""
        result = {
            'asset_id': self.asset_id,
            'time_1': self.time_1,
            'atr_at_signal': self.atr_at_signal,
            'reference_price': self.reference_price,
            'max_observation_min': self.max_observation_min,
            'max_up_atr': self.max_up_atr,
            'max_down_atr': self.max_down_atr,
            'time_to_max_up_min': self.time_to_max_up_min,
            'time_to_max_down_min': self.time_to_max_down_min,
            'first_significant_barrier': self.first_significant_barrier,
            'first_significant_time_min': self.first_significant_time_min,
        }
        
        # Flatten barrier dicts
        for key, val in self.time_to_up_barriers.items():
            result[f'time_to_up_{key}_atr'] = val
        for key, val in self.time_to_down_barriers.items():
            result[f'time_to_down_{key}_atr'] = val
            
        return result
    
    def to_db_row(self) -> Tuple:
        """Converteer naar database row tuple."""
        d = self.to_dict()
        return (
            d['asset_id'], d['time_1'], d['atr_at_signal'], d['reference_price'],
            d['max_observation_min'],
            d.get('time_to_up_025_atr'), d.get('time_to_up_050_atr'),
            d.get('time_to_up_075_atr'), d.get('time_to_up_100_atr'),
            d.get('time_to_up_125_atr'), d.get('time_to_up_150_atr'),
            d.get('time_to_up_175_atr'), d.get('time_to_up_200_atr'),
            d.get('time_to_up_225_atr'), d.get('time_to_up_250_atr'),
            d.get('time_to_up_275_atr'), d.get('time_to_up_300_atr'),
            d.get('time_to_down_025_atr'), d.get('time_to_down_050_atr'),
            d.get('time_to_down_075_atr'), d.get('time_to_down_100_atr'),
            d.get('time_to_down_125_atr'), d.get('time_to_down_150_atr'),
            d.get('time_to_down_175_atr'), d.get('time_to_down_200_atr'),
            d.get('time_to_down_225_atr'), d.get('time_to_down_250_atr'),
            d.get('time_to_down_275_atr'), d.get('time_to_down_300_atr'),
            d['max_up_atr'], d['max_down_atr'],
            d['time_to_max_up_min'], d['time_to_max_down_min'],
            d['first_significant_barrier'], d['first_significant_time_min']
        )

@dataclass
class BarrierPrediction:
    """Voorspelling voor één time window."""
    
    window_minutes: int
    
    # Probability distribution over barrier states
    p_up_strong: float
    p_up_weak: float
    p_neutral: float
    p_down_weak: float
    p_down_strong: float
    
    @property
    def most_probable_state(self) -> str:
        """De state met de hoogste probabiliteit."""
        probs = {
            'up_strong': self.p_up_strong,
            'up_weak': self.p_up_weak,
            'neutral': self.p_neutral,
            'down_weak': self.p_down_weak,
            'down_strong': self.p_down_strong
        }
        return max(probs, key=probs.get)

    @property
    def expected_direction(self) -> str:
        """Meest waarschijnlijke richting."""
        p_up = self.p_up_strong + self.p_up_weak
        p_down = self.p_down_strong + self.p_down_weak
        if p_up > p_down and p_up > self.p_neutral:
            return 'up'
        elif p_down > p_up and p_down > self.p_neutral:
            return 'down'
        return 'neutral'
    
    @property
    def directional_confidence(self) -> float:
        """Verschil tussen up en down probabiliteit."""
        p_up = self.p_up_strong + self.p_up_weak
        p_down = self.p_down_strong + self.p_down_weak
        return p_up - p_down
    
    @property
    def estimated_win_rate(self) -> float:
        """Geschatte win rate (up vs down, excl. neutral)."""
        p_up = self.p_up_strong + self.p_up_weak
        p_down = self.p_down_strong + self.p_down_weak
        total = p_up + p_down
        return p_up / total if total > 0 else 0.5
    
    @property
    def strength_ratio(self) -> float:
        """Ratio strong moves vs weak moves."""
        strong = self.p_up_strong + self.p_down_strong
        weak = self.p_up_weak + self.p_down_weak
        return strong / (strong + weak) if (strong + weak) > 0 else 0.5
    
    def to_dict(self) -> Dict:
        return {
            'window_minutes': self.window_minutes,
            'p_up_strong': round(self.p_up_strong, 4),
            'p_up_weak': round(self.p_up_weak, 4),
            'p_neutral': round(self.p_neutral, 4),
            'p_down_weak': round(self.p_down_weak, 4),
            'p_down_strong': round(self.p_down_strong, 4),
            'expected_direction': self.expected_direction,
            'directional_confidence': round(self.directional_confidence, 4),
            'estimated_win_rate': round(self.estimated_win_rate, 4),
            'strength_ratio': round(self.strength_ratio, 4)
        }

@dataclass
class BarrierInferenceResult:
    """
    Volledig resultaat van barrier-based inference (v3.1).

    v3.1 CHANGES:
    - entry_confidence is VOLLEDIG VERWIJDERD
    - position_confidence bevat nu Coincident + Confirming aggregatie
    """

    asset_id: int
    timestamp: datetime

    # Context
    regime: str
    leading_composite: str
    coincident_composite: str
    confirming_composite: str

    trade_hypothesis: str

    # Barrier predictions per window
    predictions: Dict[str, BarrierPrediction]  # key = '1h', '4h', '1d'
    position_confidence: str = "neutral"

    @property
    def distributions(self) -> Dict[str, Dict[str, float]]:
        """Backwards compatibility voor raw distributions."""
        return {
            h: {
                'up_strong': p.p_up_strong,
                'up_weak': p.p_up_weak,
                'neutral': p.p_neutral,
                'down_weak': p.p_down_weak,
                'down_strong': p.p_down_strong
            } for h, p in self.predictions.items()
        }

    @property
    def prediction_states(self) -> Dict[str, str]:
        """Backwards compatibility voor prediction states as strings."""
        return {h: p.most_probable_state for h, p in self.predictions.items()}

    # Metadata
    inference_time_ms: float = 0.0
    model_version: str = "3.1-barrier"
    
    def to_dict(self) -> Dict:
        return {
            'asset_id': self.asset_id,
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'regime': self.regime,
            'leading_composite': self.leading_composite,
            'coincident_composite': self.coincident_composite,
            'confirming_composite': self.confirming_composite,
            'trade_hypothesis': self.trade_hypothesis,
            'position_confidence': self.position_confidence,
            'predictions': {k: v.to_dict() for k, v in self.predictions.items()},
            'inference_time_ms': round(self.inference_time_ms, 2),
            'model_version': self.model_version
        }
