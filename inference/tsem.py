"""
TSEM (Trade Signal Execution Manager) Decision Logic Module.

Combineert 4 position-side signalen tot actionable trade decisions:
1. Momentum_Prediction (bearish/neutral/bullish)
2. Volatility_Regime (low_vol/normal/high_vol)
3. Exit_Timing (exit_now/hold/extend)
4. Position_Prediction (target_hit/stoploss_hit/timeout)

Zoals gespecificeerd in QBN v3.4 spec (260124_QBN_v3.4_node_structure.md):
- Exit_Timing heeft hoogste prioriteit
- Momentum als secondary check
- Volatility als sizing/confidence modifier
- Position_Prediction als tiebreaker

Usage:
    from inference.tsem import TSEMDecisionEngine, TSEMSignals, TSEMDecision
    
    engine = TSEMDecisionEngine()
    signals = TSEMSignals(
        momentum_prediction='bearish',
        volatility_regime='high_vol',
        exit_timing='hold',
        position_prediction='stoploss_hit'
    )
    decision = engine.evaluate(signals)
    print(f"Action: {decision.action}, Confidence: {decision.confidence}, Size Mult: {decision.size_multiplier}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TSEMAction(Enum):
    """Trade action decisions."""
    CLOSE = "close"
    HOLD = "hold"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


@dataclass
class TSEMSignals:
    """
    Input signalen van QBN v3.4 Position-side nodes.
    """
    momentum_prediction: str  # bearish/neutral/bullish
    volatility_regime: str  # low_vol/normal/high_vol
    exit_timing: str  # exit_now/hold/extend
    position_prediction: str  # target_hit/stoploss_hit/timeout
    
    # Optional metadata
    momentum_confidence: Optional[float] = None  # P(predicted state)
    volatility_confidence: Optional[float] = None
    exit_timing_confidence: Optional[float] = None
    position_prediction_confidence: Optional[float] = None


@dataclass
class TSEMDecision:
    """
    Output decision van TSEM engine.
    """
    action: TSEMAction
    confidence: float  # 0.0-1.0
    size_multiplier: float  # Positie grootte aanpassing (1.0 = normaal, 0.5 = half, 1.2 = 20% meer)
    reasoning: str  # Uitleg van decision logic
    priority_signal: str  # Welk signaal de decision triggerde


class TSEMDecisionEngine:
    """
    TSEM Decision Engine - combineert 4 position-side signalen.
    
    Decision Hierarchy (spec v3.4):
    1. Exit_Timing = "exit_now" → CLOSE (hoogste prioriteit)
    2. Momentum = "bearish" + Position_Prediction ∈ {"stoploss_hit", "timeout"} → CLOSE
    3. Volatility → Size modifier (niet direct action)
    4. Position_Prediction = "target_hit" + Momentum ≠ "bearish" → HOLD (met confidence)
    
    Size Modifiers:
    - low_vol: 1.2x (meer vertrouwen in stabiele markt)
    - normal: 1.0x (baseline)
    - high_vol: 0.5x (risk-off)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TSEM engine met optionele config.
        
        Args:
            config: Dict met:
                - exit_timing_confidence_threshold: float (default 0.5)
                - momentum_reversal_confidence_threshold: float (default 0.6)
                - low_vol_size_mult: float (default 1.2)
                - normal_vol_size_mult: float (default 1.0)
                - high_vol_size_mult: float (default 0.5)
        """
        self.config = config or {}
        
        # Confidence thresholds
        self.exit_timing_threshold = self.config.get('exit_timing_confidence_threshold', 0.5)
        self.momentum_threshold = self.config.get('momentum_reversal_confidence_threshold', 0.6)
        
        # Volatility size multipliers
        self.size_mult = {
            'low_vol': self.config.get('low_vol_size_mult', 1.2),
            'normal': self.config.get('normal_vol_size_mult', 1.0),
            'high_vol': self.config.get('high_vol_size_mult', 0.5),
        }
        
        logger.info(f"TSEM Decision Engine initialized with config: {self.config}")
    
    def evaluate(self, signals: TSEMSignals) -> TSEMDecision:
        """
        Evalueer position signals en return actionable decision.
        
        Args:
            signals: TSEMSignals object met 4 node outputs
            
        Returns:
            TSEMDecision met action, confidence, size_multiplier, reasoning
        """
        # Bepaal volatility size multiplier (toepassen op ALLE actions)
        size_mult = self.size_mult.get(signals.volatility_regime, 1.0)
        
        # RULE 1: Exit_Timing heeft hoogste prioriteit
        if signals.exit_timing == "exit_now":
            # Check confidence threshold als beschikbaar
            if signals.exit_timing_confidence is not None and signals.exit_timing_confidence < self.exit_timing_threshold:
                logger.debug(f"Exit_Timing='exit_now' maar confidence={signals.exit_timing_confidence:.2f} < threshold={self.exit_timing_threshold}")
            else:
                return TSEMDecision(
                    action=TSEMAction.CLOSE,
                    confidence=0.9,
                    size_multiplier=size_mult,
                    reasoning="Exit_Timing='exit_now' (hoogste prioriteit)",
                    priority_signal="exit_timing"
                )
        
        # RULE 2: Momentum reversal + negatieve Position_Prediction
        if signals.momentum_prediction == "bearish" and signals.position_prediction in ["stoploss_hit", "timeout"]:
            # Check confidence
            if signals.momentum_confidence is not None and signals.momentum_confidence < self.momentum_threshold:
                logger.debug(f"Momentum='bearish' maar confidence={signals.momentum_confidence:.2f} < threshold={self.momentum_threshold}")
            else:
                return TSEMDecision(
                    action=TSEMAction.CLOSE,
                    confidence=0.8,
                    size_multiplier=size_mult,
                    reasoning=f"Momentum='bearish' + Position_Prediction='{signals.position_prediction}'",
                    priority_signal="momentum_prediction"
                )
        
        # RULE 3: Target hit scenario (positief signaal)
        if signals.position_prediction == "target_hit" and signals.momentum_prediction != "bearish":
            # Dit is een HOLD scenario — we verwachten target te halen
            return TSEMDecision(
                action=TSEMAction.HOLD,
                confidence=0.7,
                size_multiplier=size_mult,
                reasoning=f"Position_Prediction='target_hit', Momentum='{signals.momentum_prediction}' (niet bearish)",
                priority_signal="position_prediction"
            )
        
        # RULE 4: Bullish momentum + extend signal (scale-in opportunity)
        if signals.momentum_prediction == "bullish" and signals.exit_timing == "extend":
            return TSEMDecision(
                action=TSEMAction.SCALE_IN,
                confidence=0.6,
                size_multiplier=size_mult * 0.5,  # Conservative scale-in (half position)
                reasoning="Momentum='bullish' + Exit_Timing='extend' (scale-in opportunity)",
                priority_signal="momentum_prediction"
            )
        
        # RULE 5: Bearish momentum maar nog geen exit signaal (scale-out)
        if signals.momentum_prediction == "bearish" and signals.exit_timing == "hold":
            return TSEMDecision(
                action=TSEMAction.SCALE_OUT,
                confidence=0.6,
                size_multiplier=size_mult * 0.5,  # Take partial profit
                reasoning="Momentum='bearish' maar Exit_Timing='hold' (partial exit)",
                priority_signal="momentum_prediction"
            )
        
        # DEFAULT: Hold (geen duidelijk signaal)
        return TSEMDecision(
            action=TSEMAction.HOLD,
            confidence=0.5,
            size_multiplier=size_mult,
            reasoning=f"Default HOLD: Momentum='{signals.momentum_prediction}', Exit='{signals.exit_timing}', PP='{signals.position_prediction}'",
            priority_signal="none"
        )
    
    def evaluate_dict(self, signals_dict: Dict) -> TSEMDecision:
        """
        Convenience method voor dict input (bijv. uit inference result).
        
        Args:
            signals_dict: Dict met keys: momentum_prediction, volatility_regime, exit_timing, position_prediction
            
        Returns:
            TSEMDecision
        """
        signals = TSEMSignals(
            momentum_prediction=signals_dict.get('momentum_prediction', 'neutral'),
            volatility_regime=signals_dict.get('volatility_regime', 'normal'),
            exit_timing=signals_dict.get('exit_timing', 'hold'),
            position_prediction=signals_dict.get('position_prediction', 'timeout'),
            momentum_confidence=signals_dict.get('momentum_confidence'),
            volatility_confidence=signals_dict.get('volatility_confidence'),
            exit_timing_confidence=signals_dict.get('exit_timing_confidence'),
            position_prediction_confidence=signals_dict.get('position_prediction_confidence'),
        )
        return self.evaluate(signals)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Initialize engine
    engine = TSEMDecisionEngine()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Exit Now Signal',
            'signals': TSEMSignals('neutral', 'normal', 'exit_now', 'timeout')
        },
        {
            'name': 'Bearish Momentum + Stoploss',
            'signals': TSEMSignals('bearish', 'high_vol', 'hold', 'stoploss_hit')
        },
        {
            'name': 'Target Hit + Bullish',
            'signals': TSEMSignals('bullish', 'low_vol', 'hold', 'target_hit')
        },
        {
            'name': 'Scale-In Opportunity',
            'signals': TSEMSignals('bullish', 'normal', 'extend', 'target_hit')
        },
        {
            'name': 'Partial Exit (bearish but no exit signal)',
            'signals': TSEMSignals('bearish', 'normal', 'hold', 'timeout')
        },
        {
            'name': 'Default Hold (neutral everything)',
            'signals': TSEMSignals('neutral', 'normal', 'hold', 'timeout')
        },
    ]
    
    print("="*80)
    print("TSEM DECISION ENGINE - TEST SCENARIOS")
    print("="*80)
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Input: MP={scenario['signals'].momentum_prediction}, "
              f"VR={scenario['signals'].volatility_regime}, "
              f"ET={scenario['signals'].exit_timing}, "
              f"PP={scenario['signals'].position_prediction}")
        
        decision = engine.evaluate(scenario['signals'])
        
        print(f"  → Action: {decision.action.value}")
        print(f"  → Confidence: {decision.confidence:.1f}")
        print(f"  → Size Multiplier: {decision.size_multiplier:.2f}x")
        print(f"  → Reasoning: {decision.reasoning}")
        print(f"  → Priority Signal: {decision.priority_signal}")
    
    print("\n" + "="*80)
    print("Test complete. TSEM module is ready for integration.")
    print("="*80)
