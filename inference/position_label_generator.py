"""
Position Label Generator voor QBN v3.3 Triple Composite Architecture

Genereert training labels voor de 3 position sub-prediction nodes:
- Momentum_Prediction: bearish/neutral/bullish (gebaseerd op return)
- Volatility_Regime: low_vol/normal/high_vol (gebaseerd op ATR ratio)
- Exit_Timing: exit_now/hold/extend (gebaseerd op retrospectieve analyse)

USAGE:
    from inference.position_label_generator import (
        derive_momentum_label,
        derive_volatility_label,
        derive_exit_timing_label
    )
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MOMENTUM_PREDICTION LABELS
# =============================================================================

# Thresholds voor momentum classificatie (0.5% beweging = significant)
MOMENTUM_BEARISH_THRESHOLD = -0.005
MOMENTUM_BULLISH_THRESHOLD = 0.005


def derive_momentum_label(return_value: float) -> str:
    """
    Bepaal momentum label gebaseerd op return.
    
    Args:
        return_value: Return (bijv. T+1h return of return_since_entry)
                      Moet direction-aware zijn (positief = gunstig)
    
    Returns:
        str: 'bearish', 'neutral', of 'bullish'
    """
    if pd.isna(return_value):
        return "neutral"
    
    if return_value < MOMENTUM_BEARISH_THRESHOLD:
        return "bearish"
    elif return_value > MOMENTUM_BULLISH_THRESHOLD:
        return "bullish"
    else:
        return "neutral"


def derive_momentum_labels_vectorized(returns: pd.Series) -> pd.Series:
    """
    Vectorized versie van derive_momentum_label voor hele DataFrame.
    
    Args:
        returns: Series met return values (direction-aware)
    
    Returns:
        Series met momentum labels
    """
    conditions = [
        returns < MOMENTUM_BEARISH_THRESHOLD,
        returns > MOMENTUM_BULLISH_THRESHOLD
    ]
    choices = ['bearish', 'bullish']
    return pd.Series(
        np.select(conditions, choices, default='neutral'),
        index=returns.index
    )


# =============================================================================
# VOLATILITY_REGIME LABELS
# =============================================================================

# Thresholds voor volatility classificatie (20% afwijking = significant)
VOLATILITY_LOW_THRESHOLD = 0.8
VOLATILITY_HIGH_THRESHOLD = 1.2


def derive_volatility_label(atr_ratio: float) -> str:
    """
    Bepaal volatility regime label gebaseerd op ATR ratio.
    
    Args:
        atr_ratio: Ratio van current_atr / entry_atr
                   < 1.0 = lagere volatiliteit dan bij entry
                   > 1.0 = hogere volatiliteit dan bij entry
    
    Returns:
        str: 'low_vol', 'normal', of 'high_vol'
    """
    if pd.isna(atr_ratio) or atr_ratio <= 0:
        return "normal"
    
    if atr_ratio < VOLATILITY_LOW_THRESHOLD:
        return "low_vol"
    elif atr_ratio > VOLATILITY_HIGH_THRESHOLD:
        return "high_vol"
    else:
        return "normal"


def derive_volatility_labels_vectorized(atr_ratios: pd.Series) -> pd.Series:
    """
    Vectorized versie van derive_volatility_label voor hele DataFrame.
    
    Args:
        atr_ratios: Series met ATR ratio values
    
    Returns:
        Series met volatility labels
    """
    # Handle invalid values
    valid_ratios = atr_ratios.fillna(1.0).replace(0, 1.0)
    
    conditions = [
        valid_ratios < VOLATILITY_LOW_THRESHOLD,
        valid_ratios > VOLATILITY_HIGH_THRESHOLD
    ]
    choices = ['low_vol', 'high_vol']
    return pd.Series(
        np.select(conditions, choices, default='normal'),
        index=atr_ratios.index
    )


# =============================================================================
# EXIT_TIMING LABELS
# =============================================================================

# Thresholds voor exit timing classificatie
EXIT_NOW_THRESHOLD = 0.8    # 80%+ van winst gepakt → exit
EXTEND_THRESHOLD = 0.2      # Minder dan 20% gepakt → extend


def derive_exit_timing_label(
    profit_captured_ratio: float,
    remaining_potential_ratio: Optional[float] = None
) -> str:
    """
    Bepaal exit timing label gebaseerd op retrospectieve analyse.
    
    Args:
        profit_captured_ratio: Ratio van huidige winst / maximale mogelijke winst
                              1.0 = alle winst gepakt, 0.0 = geen winst
        remaining_potential_ratio: Ratio van resterende winst potentieel
                                  (optioneel, wordt berekend als 1 - profit_captured)
    
    Returns:
        str: 'exit_now', 'hold', of 'extend'
    """
    if pd.isna(profit_captured_ratio):
        return "hold"
    
    # Clamp to valid range
    profit_captured_ratio = max(0.0, min(1.0, profit_captured_ratio))
    
    if remaining_potential_ratio is None:
        remaining_potential_ratio = 1.0 - profit_captured_ratio
    
    # Als 80%+ van winst al gepakt is, exit nu
    if profit_captured_ratio >= EXIT_NOW_THRESHOLD:
        return "exit_now"
    # Als er nog 80%+ te halen is (minder dan 20% gepakt), extend
    elif remaining_potential_ratio >= EXIT_NOW_THRESHOLD:
        return "extend"
    else:
        return "hold"


def derive_exit_timing_labels_vectorized(
    pnl_current: pd.Series,
    max_profit_event: pd.Series
) -> pd.Series:
    """
    Vectorized versie van derive_exit_timing_label voor hele DataFrame.
    
    Args:
        pnl_current: Series met huidige PnL waarden
        max_profit_event: Series met maximale winst die behaald kon worden in het event
    
    Returns:
        Series met exit timing labels
    """
    # Bereken profit captured ratio
    # Vermijd division by zero
    safe_max = max_profit_event.replace(0, np.nan)
    profit_captured = (pnl_current / safe_max).fillna(0.5)  # 50% als onbekend
    
    # Clamp to valid range
    profit_captured = profit_captured.clip(0.0, 1.0)
    remaining = 1.0 - profit_captured
    
    conditions = [
        profit_captured >= EXIT_NOW_THRESHOLD,
        remaining >= EXIT_NOW_THRESHOLD
    ]
    choices = ['exit_now', 'extend']
    return pd.Series(
        np.select(conditions, choices, default='hold'),
        index=pnl_current.index
    )


# =============================================================================
# COMBINED LABEL GENERATION
# =============================================================================

def generate_position_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Genereer alle position labels voor een DataFrame met event window data.
    
    Args:
        data: DataFrame met columns:
            - return_since_entry (voor momentum)
            - atr_ratio (voor volatility)
            - pnl_current en max_profit_event (voor exit timing) [optioneel]
    
    Returns:
        DataFrame met toegevoegde label columns:
            - momentum_label
            - volatility_label
            - exit_timing_label (indien pnl data beschikbaar)
    """
    result = data.copy()
    
    # Momentum labels
    if 'return_since_entry' in data.columns:
        result['momentum_label'] = derive_momentum_labels_vectorized(
            data['return_since_entry']
        )
        logger.info(f"   Generated momentum labels: {result['momentum_label'].value_counts().to_dict()}")
    else:
        logger.warning("   return_since_entry not found, skipping momentum labels")
    
    # Volatility labels
    if 'atr_ratio' in data.columns:
        result['volatility_label'] = derive_volatility_labels_vectorized(
            data['atr_ratio']
        )
        logger.info(f"   Generated volatility labels: {result['volatility_label'].value_counts().to_dict()}")
    else:
        logger.warning("   atr_ratio not found, skipping volatility labels")
    
    # Exit timing labels (optioneel - vereist retrospectieve data)
    if 'pnl_current' in data.columns and 'max_profit_event' in data.columns:
        result['exit_timing_label'] = derive_exit_timing_labels_vectorized(
            data['pnl_current'],
            data['max_profit_event']
        )
        logger.info(f"   Generated exit_timing labels: {result['exit_timing_label'].value_counts().to_dict()}")
    else:
        # Fallback: gebruik event_outcome om exit timing te schatten
        if 'event_outcome' in data.columns:
            result['exit_timing_label'] = _derive_exit_timing_from_outcome(data)
            logger.info(f"   Generated exit_timing labels (from outcome): {result['exit_timing_label'].value_counts().to_dict()}")
        else:
            logger.warning("   pnl_current/max_profit_event not found, skipping exit_timing labels")
    
    return result


def _derive_exit_timing_from_outcome(data: pd.DataFrame) -> pd.Series:
    """
    Fallback: schat exit timing label op basis van event outcome en time_since_entry.
    
    Logica:
    - Vroeg in event (< 4h) + strong outcome → extend (meer potentieel)
    - Laat in event (> 12h) + weak/timeout → exit_now (verminder verlies)
    - Midden → hold
    """
    labels = pd.Series('hold', index=data.index)
    
    if 'time_since_entry_min' not in data.columns or 'event_outcome' not in data.columns:
        return labels
    
    time_min = data['time_since_entry_min'].fillna(0)
    outcome = data['event_outcome'].fillna('neutral')
    
    # Vroeg + strong → extend
    early_strong = (time_min < 240) & (outcome.str.contains('strong', na=False))
    labels.loc[early_strong] = 'extend'
    
    # Laat + weak/timeout → exit_now
    late_weak = (time_min > 720) & (
        outcome.str.contains('weak', na=False) | 
        (outcome == 'timeout')
    )
    labels.loc[late_weak] = 'exit_now'
    
    return labels
