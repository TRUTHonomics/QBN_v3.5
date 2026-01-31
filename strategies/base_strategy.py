from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple

@dataclass
class TradeSignal:
    """
    Resultaat van een strategie-beslissing.
    """
    action: Literal['LONG', 'SHORT', 'HOLD', 'EXIT']
    confidence: float
    reason: str
    
    # Risk Management (Optioneel, kan ook door strategy berekend worden)
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    
    # Risk parameters (Multipliers voor ATR)
    tp_atr_mult: Optional[float] = None
    sl_atr_mult: Optional[float] = None

    # REASON: Metadata voor analyse (bijv. BN distributie op moment van entry)
    meta: Dict = None

class QBNStrategy(ABC):
    """
    Abstracte base class voor QBN trading strategieën.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def on_data(self, 
                timestamp: str,
                predictions: Dict[str, Dict], 
                market_data: Dict[str, float],
                current_position: Optional[str] = None) -> TradeSignal:
        """
        Beslis over trade actie op basis van nieuwe data.
        
        Args:
            timestamp: Huidige tijd
            predictions: Dictionary met predictions (1h, 4h, 1d) en hun distributies
            market_data: Dictionary met prijsdata (close, atr, etc.)
            current_position: Huidige positie ('LONG', 'SHORT' of None)
            
        Returns:
            TradeSignal object
        """
        pass
    
    def calculate_exits(self, entry_price: float, atr: float, direction: str) -> Tuple[float, float]:
        """
        Helper voor ATR-based exits.
        Moet geïmplementeerd worden door subklassen of gebruikt via exit params.
        """
        pass
