"""
QuantBayes Nexus Configuration Management
==========================================

Centralized configuration for Bayesian Network components.

ARCHITECTUUR NOOT:
- Signal processing en MTF building gebeurt in KFL_backend_v3
- QBN focust op Bayesian inference en leest uit KFL tabellen
- Data bronnen: kfl.mtf_signals_current_lead, kfl.mtf_signals_lead
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum, IntEnum


class TimeframeLevel(Enum):
    """Timeframe levels for multi-timeframe analysis (KFL intervals)"""
    STRUCTURAL = "D"     # Daily - Structural level (1440 min)
    TACTICAL = "240"     # 4-hour - Tactical level
    ENTRY = "60"         # 1-hour - Entry level
    UTF = "1"            # 1-minute - Micro-entry (UTF = u Timeframe, μ substituut)


class SignalState(IntEnum):
    """
    Integer signal states voor efficiënte berekeningen.
    Mapping: -2 (Strong_Bearish) tot +2 (Strong_Bullish)
    
    Database: PostgreSQL SMALLINT (2 bytes per waarde)
    Performance: 6x snellere concordance berekeningen vs string ENUM
    
    Consistent met KFL unified_indicator_worker discretisatie.
    """
    STRONG_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    STRONG_BULLISH = 2
    
    @classmethod
    def from_string(cls, value: str) -> 'SignalState':
        """Backwards compatibility: convert string to integer"""
        mapping = {
            'Strong_Bearish': cls.STRONG_BEARISH,
            'Bearish': cls.BEARISH,
            'Neutral': cls.NEUTRAL,
            'Bullish': cls.BULLISH,
            'Strong_Bullish': cls.STRONG_BULLISH
        }
        return mapping.get(value, cls.NEUTRAL)
    
    @classmethod
    def to_string(cls, value: int) -> str:
        """Convert integer to string label"""
        mapping = {
            -2: 'Strong_Bearish',
            -1: 'Bearish',
            0: 'Neutral',
            1: 'Bullish',
            2: 'Strong_Bullish'
        }
        return mapping.get(value, 'Neutral')


class NetworkLevel(Enum):
    """Hierarchical levels in Bayesian network"""
    STRUCTURAL = "Structural"  # Long-term trend analysis (Daily)
    TACTICAL = "Tactical"      # Medium-term positioning (4H)
    ENTRY = "Entry"            # Short-term execution (1H)


@dataclass
class BayesianNetworkConfig:
    """Configuration for QuantBayes Nexus Bayesian Network components"""
    
    # Selected indicators - matches KFL unified_indicator_worker output
    selected_indicators: List[str] = field(default_factory=lambda: [
        'rsi',      # RSI momentum
        'macd',     # MACD trend
        'bb',       # Bollinger Bands volatility
        'keltner',  # Keltner Channels
        'atr'       # ATR volatility
    ])
    
    # Timeframe intervals (KFL format: D, 240, 60, 1)
    timeframe_intervals: Dict[str, str] = field(default_factory=lambda: {
        'structural': 'D',    # Daily
        'tactical': '240',    # 4-hour
        'entry': '60',        # 1-hour
        'utf': '1'            # 1-minute (micro)
    })
    
    # Database column naming (matches kfl.mtf_signals_lead)
    signal_column_format: str = "{indicator}_signal_{interval}"
    # Examples: rsi_signal_d, macd_signal_240, bb_signal_60, keltner_signal_1
    
    # Confidence thresholds
    confidence_threshold: float = 0.7
    inference_timeout_ms: int = 25  # Target: < 25ms inference time
    
    # Network structure settings
    max_nodes: int = 15
    laplace_smoothing_alpha: float = 1.0  # For CPT smoothing
    
    # Lookahead bias prevention
    use_completed_candles_only: bool = True
    sync_tolerance_seconds: int = 30


@dataclass
class QBNv2Config:
    """Configuratie voor QBN v2 Bayesian Network."""
    
    # Node configuratie
    outcome_states: int = 7  # -3 tot +3
    regime_states: int = 3   # bearish/ranging/bullish
    composite_states: int = 5  # strong_bearish tot strong_bullish
    entry_timing_states: int = 4  # poor/neutral/good/excellent
    
    # Horizons
    prediction_horizons: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    
    # CPT configuratie
    min_observations_per_cell: int = 30
    laplace_smoothing_alpha: float = 1.0
    
    # Inference configuratie
    inference_timeout_ms: int = 25
    use_gpu: bool = True
    
    # Regime detection
    adx_trending_threshold: float = 25.0
    adx_strong_threshold: float = 30.0
    use_reduced_regimes: bool = os.getenv('USE_REDUCED_REGIMES', 'true').lower() == 'true'  # REASON: Fix "Neutral Trap" by reducing 11 -> 5 states
    
    # Composite aggregation
    composite_strong_threshold: float = 0.6
    composite_weak_threshold: float = 0.2
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'outcome_states': self.outcome_states,
            'regime_states': self.regime_states,
            'composite_states': self.composite_states,
            'entry_timing_states': self.entry_timing_states,
            'prediction_horizons': self.prediction_horizons,
            'min_observations_per_cell': self.min_observations_per_cell,
            'laplace_smoothing_alpha': self.laplace_smoothing_alpha,
            'inference_timeout_ms': self.inference_timeout_ms,
            'use_gpu': self.use_gpu
        }


@dataclass 
class SignalProcessorConfig:
    """Configuration for signal processing (reference only - KFL does actual processing)"""
    
    # Selected indicators - matches KFL
    selected_indicators: List[str] = field(default_factory=lambda: [
        'rsi', 'macd', 'bb', 'keltner', 'atr'
    ])
    
    # Discretization thresholds (reference - KFL uses these)
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    adx_trend_threshold: float = 25.0
    adx_strong_trend: float = 40.0
    
    bb_squeeze_threshold: float = 0.1  # Relative bandwidth
    
    # Multi-timeframe alignment
    htf_lookback_periods: int = 10
    mtf_lookback_periods: int = 20
    ltf_lookback_periods: int = 50
    
    # Error handling
    min_data_points: int = 100
    max_missing_data_pct: float = 0.1  # 10% max missing data


# Default configurations
DEFAULT_BAYESIAN_CONFIG = BayesianNetworkConfig()
DEFAULT_PROCESSOR_CONFIG = SignalProcessorConfig()

