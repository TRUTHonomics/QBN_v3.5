from typing import Dict, Optional
from .node_types import RegimeState

class HTFRegimeDetector:
    """
    Detecteert het Higher Timeframe regime op basis van D/240 signalen.
    
    Gebruikt ADX voor trend strength en directional indicators
    voor trend richting.
    """
    
    def __init__(self, adx_trending_threshold: float = 25.0, adx_strong_threshold: float = 30.0):
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_strong_threshold = adx_strong_threshold
    
    def detect_regime(
        self,
        adx_signal_d: Optional[int],
        adx_signal_240: Optional[int]
    ) -> RegimeState:
        """
        Detecteer HTF regime op basis van gediscretiseerde ADX signalen (D en 4H).
        
        Args:
            adx_signal_d: ADX signaal (-2 tot +2) op Daily timeframe
            adx_signal_240: ADX signaal (-2 tot +2) op 4H timeframe
            
        Returns:
            RegimeState (één van de 11 states)
            
        REASON: Gebruik centrale ADX signalen ipv ruwe indicators voor consistentie met pipeline.
        """
        # Default naar ranging bij ontbrekende data
        if adx_signal_d is None or adx_signal_240 is None:
            return RegimeState.MACRO_RANGING
        
        # 1. Daily Bullish
        if adx_signal_d > 0:
            if adx_signal_240 > 0:
                # Beiden bullish: check of één van beiden sterk is (+2)
                if adx_signal_d == 2 or adx_signal_240 == 2:
                    return RegimeState.SYNC_STRONG_BULLISH
                return RegimeState.SYNC_BULLISH
            elif adx_signal_240 < 0:
                return RegimeState.BULLISH_RETRACING
            else:
                return RegimeState.BULLISH_CONSOLIDATING
                
        # 2. Daily Bearish
        if adx_signal_d < 0:
            if adx_signal_240 < 0:
                # Beiden bearish: check of één van beiden sterk is (-2)
                if adx_signal_d == -2 or adx_signal_240 == -2:
                    return RegimeState.SYNC_STRONG_BEARISH
                return RegimeState.SYNC_BEARISH
            elif adx_signal_240 > 0:
                return RegimeState.BEARISH_RETRACING
            else:
                return RegimeState.BEARISH_CONSOLIDATING
                
        # 3. Daily Ranging
        if adx_signal_240 > 0:
            return RegimeState.BULLISH_EMERGING
        elif adx_signal_240 < 0:
            return RegimeState.BEARISH_EMERGING
            
        return RegimeState.MACRO_RANGING

