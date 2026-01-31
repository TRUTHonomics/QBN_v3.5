"""
Backtest Configuration voor QBN Trade Simulator.

Deze module definieert alle configureerbare parameters voor het simuleren
van trades op basis van QBN signalen over historische data.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class BacktestConfig:
    """
    Configuratie voor backtest simulatie.
    
    Deze parameters bepalen hoe QBN signalen worden vertaald naar trade entries/exits
    en hoe de trade lifecycle wordt gesimuleerd (Kraken Futures compatible).
    """
    
    # ========== BACKTEST WINDOW ==========
    start_date: datetime
    end_date: datetime
    asset_id: int
    train_window_days: int = 90
    retrain_interval_days: Optional[int] = None  # None = geen retraining tijdens backtest
    
    # ========== CAPITAL MANAGEMENT ==========
    initial_capital_usd: float = 10000.0
    
    # ========== ENTRY PARAMETERS ==========
    order_type: str = 'market'  # 'market' of 'limit'
    leverage: float = 1.0  # 1-100x (Kraken Futures max depends op symbol)
    
    # Position sizing (kies één methode)
    position_size_pct: Optional[float] = 2.0  # % van wallet risico per trade
    position_size_usd: Optional[float] = None  # Vast USD bedrag (overschrijft pct als gezet)
    
    slippage_pct: float = 0.05  # Verwachte slippage bij market orders (%)
    
    # ========== EXIT PARAMETERS ==========
    # ATR-based exits (dynamisch)
    stop_loss_atr_mult: float = 1.0  # SL = entry_price ± (atr * mult)
    take_profit_atr_mult: float = 1.5  # TP = entry_price ± (atr * mult)
    
    # Fixed % exits (alternatief)
    stop_loss_pct: Optional[float] = None  # Overschrijft ATR als gezet
    take_profit_pct: Optional[float] = None  # Overschrijft ATR als gezet
    
    use_atr_based_exits: bool = True  # True = ATR, False = fixed %
    
    # ========== POSITION MANAGEMENT ==========
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 50.0  # % van peak profit (50 = half-back)
    trailing_activation_pct: float = 0.5  # Minimum winst voor activatie (%)
    
    max_holding_time_hours: Optional[int] = None  # Timeout exit (None = geen limit)
    
    # QBN Signal-Driven Exits
    use_qbn_exit_timing: bool = True  # Exit als Exit_Timing = 'exit_now'
    exit_on_momentum_reversal: bool = False  # Exit als Momentum_Prediction omkeert
    
    # ========== QBN ENTRY FILTERS ==========
    # Trigger op Trade_Hypothesis met strength >= threshold (BEIDE long en short)
    # Valid values: 'weak' (trade op weak_* EN strong_*) of 'strong' (alleen strong_*)
    entry_strength_threshold: str = 'weak'
    
    # Regime filter (empty list = geen filter)
    # Reduced states: full_bearish, bearish_transition, macro_ranging, bullish_transition, full_bullish
    regime_filter: List[str] = field(default_factory=list)
    
    # ========== QBN MANAGEMENT SIGNALS (4 Position-Side Inputs) ==========
    # 1. Volatility_Regime (low_vol/normal/high_vol)
    volatility_position_sizing: bool = False  # Pas positiegrootte aan: low_vol=1.2x, normal=1.0x, high_vol=0.5x
    
    # 2. Position_Prediction (target_hit/stoploss_hit/timeout)
    use_position_prediction_exit: bool = False  # Exit als PP = stoploss_hit of timeout
    
    # ========== FEES & COSTS ==========
    maker_fee_pct: float = 0.02  # Kraken Futures maker fee
    taker_fee_pct: float = 0.05  # Kraken Futures taker fee
    funding_rate_enabled: bool = False  # Simuleer funding costs (vereist 8h checks)
    
    # ========== METADATA ==========
    user_id: str = 'default'
    description: str = ''
    
    def __post_init__(self):
        """Valideer configuratie na initialisatie."""
        # REASON: Forceer numerieke types naar float omdat ze als decimal.Decimal uit de DB kunnen komen
        # EXPL: Voorkomt 'TypeError: unsupported operand type(s) for /: decimal.Decimal and float'
        float_fields = [
            'leverage', 'position_size_pct', 'position_size_usd', 'slippage_pct',
            'stop_loss_atr_mult', 'stop_loss_pct', 'take_profit_atr_mult', 'take_profit_pct',
            'trailing_stop_pct', 'trailing_activation_pct', 'maker_fee_pct', 'taker_fee_pct',
            'initial_capital_usd'
        ]
        for field_name in float_fields:
            val = getattr(self, field_name)
            if val is not None:
                setattr(self, field_name, float(val))

        # REASON: Zorg dat start_date en end_date timezone-aware zijn
        if self.start_date.tzinfo is None:
            from datetime import timezone
            self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        if self.end_date.tzinfo is None:
            from datetime import timezone
            self.end_date = self.end_date.replace(tzinfo=timezone.utc)
        
        # Validaties
        if self.end_date <= self.start_date:
            raise ValueError("end_date moet na start_date zijn")
        
        if self.leverage < 1 or self.leverage > 100:
            raise ValueError("leverage moet tussen 1 en 100 zijn")
        
        if self.position_size_pct and (self.position_size_pct <= 0 or self.position_size_pct > 100):
            raise ValueError("position_size_pct moet tussen 0 en 100 zijn")
        
        if self.initial_capital_usd <= 0:
            raise ValueError("initial_capital_usd moet positief zijn")
    
    def to_dict(self) -> dict:
        """Converteer naar dictionary voor database opslag."""
        return {
            'user_id': self.user_id,
            'asset_id': self.asset_id,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'train_window_days': self.train_window_days,
            'retrain_interval_days': self.retrain_interval_days,
            'order_type': self.order_type,
            'leverage': self.leverage,
            'position_size_pct': self.position_size_pct,
            'position_size_usd': self.position_size_usd,
            'slippage_pct': self.slippage_pct,
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'take_profit_pct': self.take_profit_pct,
            'use_atr_based_exits': self.use_atr_based_exits,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trailing_stop_pct': self.trailing_stop_pct,
            'trailing_activation_pct': self.trailing_activation_pct,
            'max_holding_time_hours': self.max_holding_time_hours,
            'use_qbn_exit_timing': self.use_qbn_exit_timing,
            'entry_strength_threshold': self.entry_strength_threshold,
            'regime_filter': self.regime_filter,
            'exit_on_momentum_reversal': self.exit_on_momentum_reversal,
            'volatility_position_sizing': self.volatility_position_sizing,
            'use_position_prediction_exit': self.use_position_prediction_exit,
            'maker_fee_pct': self.maker_fee_pct,
            'taker_fee_pct': self.taker_fee_pct,
            'funding_rate_enabled': self.funding_rate_enabled,
            'initial_capital_usd': self.initial_capital_usd,
        }
