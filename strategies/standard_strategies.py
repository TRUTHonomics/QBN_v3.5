from typing import Dict, Optional, Tuple
from .base_strategy import QBNStrategy, TradeSignal

class ProbabilityThresholdStrategy(QBNStrategy):
    """
    Strategie die handelt op basis van waarschijnlijkheidsdrempels.
    
    Regels:
    - Long als P(Up) > min_prob
    - Short als P(Down) > min_prob
    - Exit als P(Contra) > exit_prob OF P(Neutral) > neutral_threshold (optioneel)
    """
    
    def __init__(
        self, 
        min_prob: float = 0.60, 
        horizon: str = '1h',
        atr_tp: float = 2.0,
        atr_sl: float = 1.5,
        required_regimes: Optional[list] = None
    ):
        super().__init__(f"ProbThreshold_{horizon}_{min_prob}")
        self.min_prob = min_prob
        self.horizon = horizon
        self.atr_tp = atr_tp
        self.atr_sl = atr_sl
        self.required_regimes = required_regimes
        
    def on_data(self, 
                timestamp: str,
                predictions: Dict[str, Dict], 
                market_data: Dict[str, float],
                current_position: Optional[str] = None) -> TradeSignal:
        
        # 1. Haal voorspelling op voor gekozen horizon
        pred_key = f'prediction_{self.horizon}'
        if pred_key not in predictions:
            return TradeSignal('HOLD', 0.0, "No prediction")
            
        dist = predictions[pred_key] # Dict met {state: prob}
        
        # 2. Check Regime Filter
        if self.required_regimes:
            current_regime = predictions.get('regime')
            if current_regime not in self.required_regimes:
                return TradeSignal('HOLD', 0.0, f"Regime {current_regime} not allowed")
        
        # 3. Bereken Bullish/Bearish kansen
        p_up = sum(prob for state, prob in dist.items() if 'up' in state.lower() or 'bullish' in state.lower())
        p_down = sum(prob for state, prob in dist.items() if 'down' in state.lower() or 'bearish' in state.lower())
        
        # 4. Entry Logica
        if current_position is None:
            meta = {
                'p_up': p_up,
                'p_down': p_down,
                'regime': predictions.get('regime', ""),
                'hypothesis': predictions.get('trade_hypothesis', "")
            }
            if p_up >= self.min_prob:
                return TradeSignal(
                    'LONG', 
                    p_up, 
                    f"Bullish Prob {p_up:.2f} > {self.min_prob}",
                    tp_atr_mult=self.atr_tp,
                    sl_atr_mult=self.atr_sl,
                    meta=meta
                )
            elif p_down >= self.min_prob:
                return TradeSignal(
                    'SHORT', 
                    p_down, 
                    f"Bearish Prob {p_down:.2f} > {self.min_prob}",
                    tp_atr_mult=self.atr_tp,
                    sl_atr_mult=self.atr_sl,
                    meta=meta
                )
                
        # 5. Exit Logica (Dynamic) - als signaal omdraait
        elif current_position == 'LONG':
            if p_down > 0.50: # Voorbeeld: Sluit long als bearish kans > 50%
                return TradeSignal('EXIT', p_down, "Signal reversal (Bearish dominance)")
                
        elif current_position == 'SHORT':
            if p_up > 0.50:
                return TradeSignal('EXIT', p_up, "Signal reversal (Bullish dominance)")
                
        return TradeSignal('HOLD', 0.0, "No signal")

class RelativeEdgeStrategy(QBNStrategy):
    """
    Strategie die kijkt naar de relatieve verhouding tussen P(Up) en P(Down).
    
    Regels:
    - Long als P(Up) > P(Down) * edge_factor
    - Short als P(Down) > P(Up) * edge_factor
    - Filter: Minimale gezamenlijke kans (bijv. P(Up)+P(Down) > 0.05) om ruis te voorkomen.
    """
    
    def __init__(
        self, 
        edge_factor: float = 1.5, 
        min_total_signal: float = 0.02,
        horizon: str = '1h',
        atr_tp: float = 2.0,
        atr_sl: float = 1.5
    ):
        super().__init__(f"RelativeEdge_{horizon}_x{edge_factor}")
        self.edge_factor = edge_factor
        self.min_total_signal = min_total_signal
        self.horizon = horizon
        self.atr_tp = atr_tp
        self.atr_sl = atr_sl
        
    def on_data(self, 
                timestamp: str,
                predictions: Dict[str, Dict], 
                market_data: Dict[str, float],
                current_position: Optional[str] = None) -> TradeSignal:
        
        pred_key = f'prediction_{self.horizon}'
        if pred_key not in predictions:
            return TradeSignal('HOLD', 0.0, "No prediction")
            
        dist = predictions[pred_key]
        
        # Bereken Bullish/Bearish kansen
        p_up = sum(prob for state, prob in dist.items() if 'up' in state.lower() or 'bullish' in state.lower())
        p_down = sum(prob for state, prob in dist.items() if 'down' in state.lower() or 'bearish' in state.lower())
        
        total_signal = p_up + p_down
        
        if current_position is None:
            meta = {
                'p_up': p_up,
                'p_down': p_down,
                'regime': predictions.get('regime', ""),
                'hypothesis': predictions.get('trade_hypothesis', "")
            }
            # Check of we genoeg "signaal" hebben t.o.v. ruis
            if total_signal < self.min_total_signal:
                return TradeSignal('HOLD', 0.0, f"Signal too weak ({total_signal:.4f} < {self.min_total_signal})")
            
            # Long logica: P(Up) moet x keer groter zijn dan P(Down)
            if p_up > p_down * self.edge_factor:
                return TradeSignal(
                    'LONG', 
                    p_up / (p_up + p_down), # Genormaliseerde sterkte
                    f"Bullish Edge: {p_up:.4f} vs {p_down:.4f} (Ratio {(p_up/max(0.0001, p_down)):.2f})",
                    tp_atr_mult=self.atr_tp,
                    sl_atr_mult=self.atr_sl,
                    meta=meta
                )
            
            # Short logica
            elif p_down > p_up * self.edge_factor:
                return TradeSignal(
                    'SHORT', 
                    p_down / (p_up + p_down),
                    f"Bearish Edge: {p_down:.4f} vs {p_up:.4f} (Ratio {(p_down/max(0.0001, p_up)):.2f})",
                    tp_atr_mult=self.atr_tp,
                    sl_atr_mult=self.atr_sl,
                    meta=meta
                )
                
        # Exit Logica: Sluit als de ratio omdraait tegen onze positie
        elif current_position == 'LONG':
            if p_down > p_up: # Zodra Bearish waarschijnlijker is dan Bullish
                return TradeSignal('EXIT', p_down, "Edge lost (P_down > P_up)")
                
        elif current_position == 'SHORT':
            if p_up > p_down:
                return TradeSignal('EXIT', p_up, "Edge lost (P_up > P_down)")
                
        return TradeSignal('HOLD', 0.0, "No signal")
