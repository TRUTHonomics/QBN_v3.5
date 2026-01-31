"""
Signal Aggregator voor QBN v3.

Aggregeert individuele signalen naar composite states (strong_bearish tot strong_bullish).
Thresholds worden geladen uit de database via ThresholdLoader.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
import logging
import numpy as np
from .node_types import SemanticClass, CompositeState
from config.bayesian_config import SignalState

if TYPE_CHECKING:
    from config.threshold_loader import ThresholdLoader

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Aggregeert individuele signalen naar composite states.
    
    Neemt signalen uit een semantische klasse en berekent
    een aggregate state (strong_bearish tot strong_bullish).
    
    ARCHITECTUUR:
    - Thresholds worden geïnjecteerd via ThresholdLoader (aanbevolen)
    - Fallback naar config.network_config constanten voor backward compatibility
    """
    
    # Mapping van signal polarity naar numerieke waarde
    POLARITY_VALUES = {
        SignalState.BULLISH: 1,
        SignalState.STRONG_BULLISH: 2,
        SignalState.BEARISH: -1,
        SignalState.STRONG_BEARISH: -2,
        SignalState.NEUTRAL: 0
    }
    
    def __init__(
        self,
        signal_classification: Dict[str, Dict],
        threshold_loader: Optional['ThresholdLoader'] = None
    ):
        """
        Args:
            signal_classification: Dict met signal_name -> {semantic_class, polarity, weights}
            threshold_loader: Optional ThresholdLoader voor DB-driven thresholds.
                              Als None wordt gebruikt, worden fallback defaults geladen.
        """
        self.signal_classification = signal_classification
        self._threshold_loader = threshold_loader
        self._thresholds: Dict[str, float] = {}
        
        # Initialiseer thresholds
        self._init_thresholds()
        
        # Bouw class mapping
        self._build_class_mapping()
    
    def _init_thresholds(self):
        """Initialiseer composite thresholds uit ThresholdLoader of fallback."""
        if self._threshold_loader is not None:
            # REASON: De thresholds zijn nu per-laag beschikbaar. 
            # We laden ze hier lui per aggregatie of cache ze per klasse.
            pass
        else:
            # REASON: Fallback naar centrale defaults met duidelijke warning
            from core.config_defaults import DEFAULT_COMPOSITE_NEUTRAL_BAND, DEFAULT_COMPOSITE_STRONG_THRESHOLD
            from core.config_warnings import warn_fallback_active
            
            warn_fallback_active(
                component="SignalAggregator",
                config_name="composite_thresholds",
                fallback_values={
                    'nb': DEFAULT_COMPOSITE_NEUTRAL_BAND, 
                    'st': DEFAULT_COMPOSITE_STRONG_THRESHOLD
                },
                reason="Geen ThresholdLoader aanwezig",
                fix_command="Zorg dat een ThresholdLoader wordt geïnjecteerd"
            )
            
            self._thresholds = {
                'strong_bearish': -DEFAULT_COMPOSITE_STRONG_THRESHOLD,
                'bearish': -DEFAULT_COMPOSITE_NEUTRAL_BAND,
                'neutral_low': -DEFAULT_COMPOSITE_NEUTRAL_BAND,
                'neutral_high': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                'bullish': DEFAULT_COMPOSITE_NEUTRAL_BAND,
                'strong_bullish': DEFAULT_COMPOSITE_STRONG_THRESHOLD
            }
    
    def update_thresholds(self, threshold_loader: 'ThresholdLoader'):
        """
        Update thresholds met een nieuwe ThresholdLoader.
        
        Args:
            threshold_loader: Nieuwe ThresholdLoader instance
        """
        self._threshold_loader = threshold_loader
        logger.debug(f"Thresholds updated to {threshold_loader.horizon}")
    
    def _build_class_mapping(self):
        """Bouw mapping van semantic class naar signalen."""
        self.class_signals: Dict[SemanticClass, List[str]] = {
            SemanticClass.LEADING: [],
            SemanticClass.COINCIDENT: [],
            SemanticClass.CONFIRMING: []
        }
        
        for signal_name, info in self.signal_classification.items():
            try:
                sem_class = SemanticClass(info['semantic_class'])
                self.class_signals[sem_class].append(signal_name)
            except (ValueError, KeyError):
                continue
    
    def _get_thresholds_for_class(self, semantic_class: SemanticClass) -> Dict[str, float]:
        """Haal thresholds op voor een specifieke klasse."""
        if self._threshold_loader:
            params = self._threshold_loader.get_composite_params(semantic_class.value)
            # v3.5: asymmetrisch indien beschikbaar, anders fallback naar legacy symmetrisch.
            nb = float(params.get('neutral_band', 0.0))
            st = float(params.get('strong_threshold', 0.0))
            bull_nb = float(params.get('bullish_neutral_band', nb))
            bear_nb = float(params.get('bearish_neutral_band', nb))
            bull_st = float(params.get('bullish_strong_threshold', st))
            bear_st = float(params.get('bearish_strong_threshold', st))
            return {
                'strong_bearish': -bear_st,
                'bearish': -bear_nb,
                'bullish': bull_nb,
                'strong_bullish': bull_st
            }
        return self._thresholds

    def aggregate_signals(
        self,
        active_signals: Dict[str, int],
        semantic_class: SemanticClass,
        horizon: str = '1h'
    ) -> CompositeState:
        """
        Aggregeer signalen van één semantische klasse naar composite state.
        
        Args:
            active_signals: Dict van signal_name -> SignalState (int value)
            semantic_class: De semantische klasse om te aggregeren
            horizon: De voospellingshorizon voor gewichtsbepaling ('1h', '4h', '1d')
            
        Returns:
            CompositeState (strong_bearish tot strong_bullish)
        """
        signals_in_class = self.class_signals[semantic_class]
        
        if not signals_in_class:
            return CompositeState.NEUTRAL
        
        # Bereken score
        total_score = 0.0
        active_count = 0
        
        for signal_name in signals_in_class:
            if signal_name not in active_signals:
                continue
                
            signal_val = active_signals[signal_name]
            try:
                state = SignalState(signal_val)
            except ValueError:
                state = SignalState.NEUTRAL
                
            polarity_value = self.POLARITY_VALUES.get(state, 0)
            
            sig_info = self.signal_classification.get(signal_name, {})
            
            # REASON: Haal polariteit op uit classificatie (bullish=1, bearish=-1, neutral=0)
            # KFL signalen zijn vaak 1 wanneer actief, ook voor bearish signalen.
            raw_polarity = sig_info.get('polarity', 1)
            if isinstance(raw_polarity, str):
                pol_lower = raw_polarity.lower()
                polarity = 1 if pol_lower == 'bullish' else (-1 if pol_lower == 'bearish' else 0)
            else:
                polarity = int(raw_polarity) if raw_polarity is not None else 1
            
            weights = sig_info.get('weights', {})
            weight = weights.get(horizon, sig_info.get('weight', 1.0))
            
            # Score bijdrage: signal_waarde * polariteit * weight
            # signal_waarde (polarity_value) is meestal 1 of -1
            total_score += polarity_value * polarity * weight
            active_count += weight
        
        if active_count == 0:
            normalized_score = 0.0
        else:
            normalized_score = total_score / (active_count * 2.0)
        
        # REASON: Gebruik klasse-specifieke thresholds
        thresholds = self._get_thresholds_for_class(semantic_class)
        return self._score_to_state(normalized_score, thresholds)
    
    def _score_to_state(self, score: float, thresholds: Dict[str, float]) -> CompositeState:
        """Map normalized score naar CompositeState met specifieke thresholds."""
        if score < thresholds['strong_bearish']:
            return CompositeState.STRONG_BEARISH
        elif score < thresholds['bearish']:
            return CompositeState.BEARISH
        elif score >= thresholds['strong_bullish']:
            return CompositeState.STRONG_BULLISH
        elif score >= thresholds['bullish']:
            return CompositeState.BULLISH
        else:
            return CompositeState.NEUTRAL
    
    def aggregate_all_classes(
        self,
        active_signals: Dict[str, int],
        horizon: str = '1h'
    ) -> Dict[SemanticClass, CompositeState]:
        """
        Aggregeer alle semantische klassen tegelijk.
        
        Returns:
            Dict van SemanticClass -> CompositeState
        """
        return {
            sem_class: self.aggregate_signals(active_signals, sem_class, horizon)
            for sem_class in SemanticClass
        }
