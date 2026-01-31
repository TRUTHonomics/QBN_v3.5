"""
State Reduction Module voor HTF_Regime.
Mapt de 11 gedetailleerde macro-regimes naar 5 robuustere states
om de "Neutral Trap" bij kleine datasets te voorkomen.
"""

import numpy as np
from typing import Dict, Union, List
from enum import Enum

class StateReductionLevel(Enum):
    FULL = "full"       # 11 states
    REDUCED = "reduced" # 5 states

# Mapping definitie van 11 -> 5 states
REGIME_MAPPING_11_TO_5 = {
    # Full Bullish
    "sync_strong_bullish": "full_bullish",
    "sync_bullish": "full_bullish",
    
    # Bullish Transition
    "bullish_consolidating": "bullish_transition",
    "bullish_retracing": "bullish_transition",
    "bullish_emerging": "bullish_transition",
    
    # Macro Ranging
    "macro_ranging": "macro_ranging",
    
    # Bearish Transition
    "bearish_consolidating": "bearish_transition",
    "bearish_retracing": "bearish_transition",
    "bearish_emerging": "bearish_transition",
    
    # Full Bearish
    "sync_strong_bearish": "full_bearish",
    "sync_bearish": "full_bearish"
}

# Lijst van de 5 gereduceerde states in volgorde van Bearish naar Bullish (0 tot 4)
REDUCED_REGIME_STATES = [
    "full_bearish",        # 0
    "bearish_transition",  # 1
    "macro_ranging",       # 2
    "bullish_transition",  # 3
    "full_bullish"         # 4
]

# REASON: Map van 11-state names naar reduced state names voor CPT loading
# EXPL: De CPT generator gebruikt soms de volle 11-state names, de inference engine
# werkt met de 5 reduced states. Deze map converteert bij het laden van CPTs.
FULL_TO_REDUCED_REGIME_MAP = {
    # Full Bullish
    "sync_strong_bullish": "full_bullish",
    "sync_bullish": "full_bullish",
    
    # Bullish Transition
    "bullish_consolidating": "bullish_transition",
    "bullish_retracing": "bullish_transition",
    "bullish_emerging": "bullish_transition",
    
    # Macro Ranging
    "macro_ranging": "macro_ranging",
    
    # Bearish Transition
    "bearish_consolidating": "bearish_transition",
    "bearish_retracing": "bearish_transition",
    "bearish_emerging": "bearish_transition",
    
    # Full Bearish
    "sync_strong_bearish": "full_bearish",
    "sync_bearish": "full_bearish",
    
    # Also include reduced states mapping to themselves for consistency
    "full_bullish": "full_bullish",
    "bullish_transition": "bullish_transition",
    "bearish_transition": "bearish_transition",
    "full_bearish": "full_bearish"
}

def get_state_mapping(level: StateReductionLevel = StateReductionLevel.REDUCED) -> Dict[str, str]:
    """Geeft de mapping dictionary terug voor het opgegeven niveau."""
    if level == StateReductionLevel.REDUCED:
        return REGIME_MAPPING_11_TO_5
    return {s: s for s in REGIME_MAPPING_11_TO_5.keys()} # Identity mapping for FULL

def recommend_reduction_level(asset_count: int, days: int) -> StateReductionLevel:
    """Aanbeveling op basis van dataset grootte."""
    if asset_count < 10 or days < 365:
        return StateReductionLevel.REDUCED
    return StateReductionLevel.FULL

def reduce_regime_state(state: str) -> str:
    """
    Reduceert een enkele 11-state regime string naar een 5-state string.
    """
    if not state:
        return "macro_ranging"
    return REGIME_MAPPING_11_TO_5.get(state.lower(), "macro_ranging")

def reduce_regime_array(states: Union[np.ndarray, list]) -> np.ndarray:
    """
    Vectorized mapping van een array met 11-state strings naar 5-state strings.
    """
    return np.array([reduce_regime_state(s) for s in states])
