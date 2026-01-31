from enum import Enum, IntEnum
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from config.bayesian_config import SignalState

class NodeType(Enum):
    """Types van nodes in het Bayesian Network."""
    ROOT = "root"              # HTF_Regime
    COMPOSITE = "composite"    # Leading/Coincident/Confirming composites
    ENTRY = "entry"            # Trade_Hypothesis
    TIMING = "timing"          # Entry_Confidence (legacy)
    MANAGEMENT = "management"  # Position_Confidence (v3.2 legacy, verwijderd in v3.4)
    LATENT = "latent"          # Legacy Entry_Timing (houd voor compatibiliteit)
    TARGET = "target"          # Prediction nodes
    
    # v3.4: Direct Sub-Predictions Architecture
    POSITION_SUBPREDICTION = "position_subprediction"  # Momentum_Prediction, Volatility_Regime, Exit_Timing

class SemanticClass(Enum):
    """Semantische classificatie van signalen."""
    LEADING = "LEADING"
    COINCIDENT = "COINCIDENT"
    CONFIRMING = "CONFIRMING"

class OutcomeState(IntEnum):
    """7-state outcome representatie (-3 tot +3)."""
    STRONG_BEARISH = -3
    BEARISH = -2
    SLIGHT_BEARISH = -1
    NEUTRAL = 0
    SLIGHT_BULLISH = 1
    BULLISH = 2
    STRONG_BULLISH = 3
    
    @classmethod
    def all_states(cls) -> List[int]:
        return [-3, -2, -1, 0, 1, 2, 3]
    
    @classmethod
    def state_names(cls) -> List[str]:
        return [
            "Strong_Bearish", "Bearish", "Slight_Bearish",
            "Neutral",
            "Slight_Bullish", "Bullish", "Strong_Bullish"
        ]

    @classmethod
    def to_string(cls, value: int) -> str:
        """Convert integer to string label."""
        # REASON: Mapping voor 7-state outcomes (-3 tot +3)
        # EXPL: value + 3 mapt -3 naar index 0, 0 naar index 3, etc.
        idx = int(value) + 3
        names = cls.state_names()
        if 0 <= idx < len(names):
            return names[idx]
        return "Neutral"

class BarrierOutcomeState(Enum):
    """Discrete outcome states gebaseerd op first-touch barriers."""
    
    UP_STRONG = "up_strong"      # First touch >= 1.00 ATR up
    UP_WEAK = "up_weak"          # First touch 0.50-0.99 ATR up
    NEUTRAL = "neutral"          # Geen significant barrier binnen window
    DOWN_WEAK = "down_weak"      # First touch 0.50-0.99 ATR down
    DOWN_STRONG = "down_strong"  # First touch >= 1.00 ATR down
    
    @classmethod
    def state_names(cls) -> List[str]:
        return [s.value for s in cls]
    
    @classmethod
    def from_barrier(cls, barrier_name: str, time_min: int, window_min: int) -> 'BarrierOutcomeState':
        """
        Map barrier naam naar discrete state.
        
        Args:
            barrier_name: 'up_075', 'down_100', 'none', etc.
            time_min: Tijd tot barrier in minuten
            window_min: Maximum window voor deze voorspelling
        """
        if barrier_name is None or not isinstance(barrier_name, str) or barrier_name == 'none' or time_min is None or time_min > window_min:
            return cls.NEUTRAL
        
        if barrier_name.startswith('up_'):
            try:
                level = int(barrier_name.split('_')[1]) / 100
                return cls.UP_STRONG if level >= 1.0 else cls.UP_WEAK
            except (IndexError, ValueError):
                return cls.NEUTRAL
        
        if barrier_name.startswith('down_'):
            try:
                level = int(barrier_name.split('_')[1]) / 100
                return cls.DOWN_STRONG if level >= 1.0 else cls.DOWN_WEAK
            except (IndexError, ValueError):
                return cls.NEUTRAL
        
        return cls.NEUTRAL

class RegimeState(Enum):
    """
    HTF Regime states (11-state model).
    Covers all 25 combinations of Daily and 4H ADX signals.
    """
    SYNC_STRONG_BULLISH = "sync_strong_bullish"
    SYNC_BULLISH = "sync_bullish"
    BULLISH_CONSOLIDATING = "bullish_consolidating"
    BULLISH_RETRACING = "bullish_retracing"
    BULLISH_EMERGING = "bullish_emerging"
    MACRO_RANGING = "macro_ranging"
    BEARISH_EMERGING = "bearish_emerging"
    BEARISH_RETRACING = "bearish_retracing"
    BEARISH_CONSOLIDATING = "bearish_consolidating"
    SYNC_BEARISH = "sync_bearish"
    SYNC_STRONG_BEARISH = "sync_strong_bearish"

    @classmethod
    def all_states(cls) -> List[str]:
        """
        Geeft de lijst van actieve states terug op basis van configuratie.
        REASON: Support voor 5-state reduction bij kleine datasets.
        """
        # REASON: Import hier om circulaire dependencies te voorkomen
        from config.bayesian_config import QBNv2Config
        from .state_reduction import REDUCED_REGIME_STATES
        
        config = QBNv2Config()
        if config.use_reduced_regimes:
            return REDUCED_REGIME_STATES
        return [s.value for s in cls]

class CompositeState(Enum):
    """Composite signal states (aggregated from individual signals)."""
    STRONG_BEARISH = "strong_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    STRONG_BULLISH = "strong_bullish"

@dataclass
class NodeDefinition:
    """Definitie van een BN node."""
    name: str
    node_type: NodeType
    states: List[str]
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    description: str = ""
    
    @property
    def num_states(self) -> int:
        return len(self.states)
    
    @property
    def is_root(self) -> bool:
        return len(self.parents) == 0
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

