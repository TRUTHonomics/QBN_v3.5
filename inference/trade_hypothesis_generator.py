"""
Trade_Hypothesis Generator voor QBN v3

Genereert CPTs voor de Trade_Hypothesis node op basis van:
- Leading_Composite state
- Historische outcomes

REASON: Voorstel 2 - Leading-First Cascade
"""

from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

from .node_types import CompositeState


class TradeHypothesisGenerator:
    """
    Generator voor Trade_Hypothesis CPTs.
    
    Maps Leading_Composite → Trade_Hypothesis met outcome-based training.
    """
    
    HYPOTHESIS_STATES = [
        "no_setup", 
        "weak_long", 
        "strong_long", 
        "weak_short", 
        "strong_short"
    ]
    
    # Deterministische mapping Leading → Hypothesis
    # EXPL: Leading composites zijn de primaire bron voor de trade hypothese.
    # REASON: Uitgebreid met integers en capitalized strings voor robuustheid.
    LEADING_TO_HYPOTHESIS = {
        # Lowercase strings
        'strong_bullish': 'strong_long',
        'bullish': 'weak_long',
        'neutral': 'no_setup',
        'bearish': 'weak_short',
        'strong_bearish': 'strong_short',
        
        # Capitalized strings (van SignalState)
        'Strong_Bullish': 'strong_long',
        'Bullish': 'weak_long',
        'Neutral': 'no_setup',
        'Bearish': 'weak_short',
        'Strong_Bearish': 'strong_short',
        
        # Integers (van GPU aggregatie)
        2: 'strong_long',
        1: 'weak_long',
        0: 'no_setup',
        -1: 'weak_short',
        -2: 'strong_short'
    }
    
    def __init__(self, laplace_alpha: float = 1.0):
        self.laplace_alpha = laplace_alpha
    
    def generate_cpt(
        self,
        training_data: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Genereer Trade_Hypothesis CPT.
        
        Args:
            training_data: List van {leading_composite, outcome} dicts
            
        Returns:
            CPT: Dict[leading_state] -> Dict[hypothesis_state] -> probability
        """
        # Count observations
        counts = defaultdict(lambda: defaultdict(int))
        
        for record in training_data:
            leading = record.get('leading_composite', 'neutral')
            
            # Derive hypothesis van leading
            hypothesis = self.derive_hypothesis(leading)
            
            counts[leading][hypothesis] += 1
        
        # Normalize met Laplace smoothing
        cpt = {}
        composite_states = [s.value for s in CompositeState]
        
        for leading_state in composite_states:
            state_counts = counts[leading_state]
            total = sum(state_counts.values()) + self.laplace_alpha * len(self.HYPOTHESIS_STATES)
            
            cpt[leading_state] = {}
            for hyp_state in self.HYPOTHESIS_STATES:
                count = state_counts.get(hyp_state, 0)
                cpt[leading_state][hyp_state] = (count + self.laplace_alpha) / total
        
        return cpt
    
    def derive_hypothesis(self, leading_composite: str) -> str:
        """
        Derive Trade_Hypothesis state van Leading_Composite.
        
        Args:
            leading_composite: CompositeState value
            
        Returns:
            Hypothesis state
        """
        return self.LEADING_TO_HYPOTHESIS.get(leading_composite, 'no_setup')

