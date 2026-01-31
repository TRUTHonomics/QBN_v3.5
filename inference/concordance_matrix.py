#!/usr/bin/env python3
"""
ConcordanceMatrix - Multi-timeframe signal agreement analysis

ARCHITECTUUR NOOT:
- Concordance berekening is ook geïmplementeerd in KFL triggers
- Deze module biedt dezelfde logica voor QBN inference
- Kan gebruikt worden voor evidence node generatie in Bayesian Network
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from enum import Enum

# Core imports
from config.bayesian_config import SignalState

logger = logging.getLogger(__name__)


class ConcordanceScenario(Enum):
    """Concordance scenarios voor 4-timeframe analyse (D, 240, 60, 1)"""
    STRONG_BULLISH = "strong_bullish"      # 4/4 of 3/4 Bullish, geen Bearish
    MODERATE_BULLISH = "moderate_bullish"  # 3/4 Bullish met 1 Neutral, of 2/4 Bullish
    WEAK_BULLISH = "weak_bullish"          # Meerderheid Bullish met tegengestelde
    NEUTRAL = "neutral"                    # Mixed of all Neutral
    WEAK_BEARISH = "weak_bearish"          # Meerderheid Bearish met tegengestelde
    MODERATE_BEARISH = "moderate_bearish"  # 3/4 Bearish met 1 Neutral, of 2/4 Bearish
    STRONG_BEARISH = "strong_bearish"      # 4/4 of 3/4 Bearish, geen Bullish
    CONFLICTED = "conflicted"              # Geen duidelijke meerderheid


class ConcordanceMatrix:
    """
    Multi-timeframe concordance analysis voor Bayesian evidence.
    
    Implementeert 4-timeframe classificatie en berekent concordance scores
    voor integratie als evidence node in het Bayesian network.
    
    Timeframe Mapping (KFL):
    - HTF/Structural: Daily (D) - 50%
    - MTF/Tactical: 4H (240) - 25%
    - LTF/Entry: 1H (60) - 15%
    - UTF/Micro: 1m (1) - 10%
    """
    
    def __init__(self, 
                 structural_weight: float = 0.50,
                 tactical_weight: float = 0.25, 
                 entry_weight: float = 0.15,
                 utf_weight: float = 0.10):
        """
        Initialize Concordance Matrix
        
        Args:
            structural_weight: Weging voor structural (Daily) signals (50%)
            tactical_weight: Weging voor tactical (4H) signals (25%)
            entry_weight: Weging voor entry (1H) signals (15%)
            utf_weight: Weging voor micro (1m) signals (10%)
        """
        self.structural_weight = structural_weight
        self.tactical_weight = tactical_weight
        self.entry_weight = entry_weight
        self.utf_weight = utf_weight
        
        total_weight = structural_weight + tactical_weight + entry_weight + utf_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        logger.info(f"ConcordanceMatrix initialized with weights: S={structural_weight}, T={tactical_weight}, E={entry_weight}, U={utf_weight}")
    
    def classify_scenario(self, 
                         htf_signal: SignalState,
                         mtf_signal: SignalState, 
                         ltf_signal: SignalState,
                         utf_signal: Optional[SignalState] = None) -> ConcordanceScenario:
        """
        Classificeer concordance scenario voor 4 timeframes.
        
        Args:
            htf_signal: Higher timeframe signal state (Daily)
            mtf_signal: Medium timeframe signal state (4H)
            ltf_signal: Lower timeframe signal state (1H)
            utf_signal: Micro timeframe signal state (1m), optional voor backwards compatibility
            
        Returns:
            ConcordanceScenario enum
        """
        # REASON: Support zowel 3 als 4 timeframes voor backwards compatibility
        if utf_signal is not None:
            signals = [htf_signal, mtf_signal, ltf_signal, utf_signal]
            total_signals = 4
        else:
            signals = [htf_signal, mtf_signal, ltf_signal]
            total_signals = 3
        
        # Count signal types (handle both SignalState and int)
        def is_bullish(s):
            return (isinstance(s, SignalState) and s in [SignalState.BULLISH, SignalState.STRONG_BULLISH]) or \
                   (isinstance(s, int) and s > 0)
        
        def is_bearish(s):
            return (isinstance(s, SignalState) and s in [SignalState.BEARISH, SignalState.STRONG_BEARISH]) or \
                   (isinstance(s, int) and s < 0)
        
        def is_neutral(s):
            return (isinstance(s, SignalState) and s == SignalState.NEUTRAL) or \
                   (isinstance(s, int) and s == 0)
        
        bullish_count = sum(1 for s in signals if is_bullish(s))
        bearish_count = sum(1 for s in signals if is_bearish(s))
        neutral_count = sum(1 for s in signals if is_neutral(s))
        
        # Strong scenarios (alle of bijna alle gelijk, geen tegengesteld)
        if bullish_count == total_signals:
            return ConcordanceScenario.STRONG_BULLISH
        elif bearish_count == total_signals:
            return ConcordanceScenario.STRONG_BEARISH
        elif bullish_count >= total_signals - 1 and bearish_count == 0:
            return ConcordanceScenario.STRONG_BULLISH
        elif bearish_count >= total_signals - 1 and bullish_count == 0:
            return ConcordanceScenario.STRONG_BEARISH
        
        # Moderate scenarios (meerderheid + neutrals)
        elif bullish_count >= total_signals // 2 + 1 and bearish_count == 0:
            return ConcordanceScenario.MODERATE_BULLISH
        elif bearish_count >= total_signals // 2 + 1 and bullish_count == 0:
            return ConcordanceScenario.MODERATE_BEARISH
        elif bullish_count == total_signals // 2 and bearish_count == 0:
            return ConcordanceScenario.MODERATE_BULLISH
        elif bearish_count == total_signals // 2 and bullish_count == 0:
            return ConcordanceScenario.MODERATE_BEARISH
        
        # Weak scenarios (meerderheid met tegengestelde)
        elif bullish_count > bearish_count and bearish_count > 0:
            return ConcordanceScenario.WEAK_BULLISH
        elif bearish_count > bullish_count and bullish_count > 0:
            return ConcordanceScenario.WEAK_BEARISH
        
        # Neutral scenarios
        elif neutral_count == total_signals:
            return ConcordanceScenario.NEUTRAL
        elif neutral_count >= total_signals // 2 + 1:
            return ConcordanceScenario.NEUTRAL
        
        # Conflicted (gelijke verdeling bullish/bearish)
        elif bullish_count == bearish_count and bullish_count > 0:
            return ConcordanceScenario.CONFLICTED
        
        return ConcordanceScenario.NEUTRAL
    
    def calculate_concordance_score(self, 
                                  htf_signal: SignalState,
                                  mtf_signal: SignalState,
                                  ltf_signal: SignalState,
                                  utf_signal: Optional[SignalState] = None) -> float:
        """
        Bereken numerieke concordance score (0.0 - 1.0).
        
        Gebruikt integer arithmetic met gewogen sum:
        HTF (structural) = 50%, MTF (tactical) = 25%, LTF (entry) = 15%, UTF (micro) = 10%
        
        Args:
            htf_signal: Higher timeframe signal state (Daily)
            mtf_signal: Medium timeframe signal state (4H)
            ltf_signal: Lower timeframe signal state (1H)
            utf_signal: Micro timeframe signal state (1m), optional voor backwards compatibility
        
        Returns:
            Normalized concordance score (0.0 - 1.0)
        """
        # Convert to int if SignalState
        htf_val = int(htf_signal) if isinstance(htf_signal, SignalState) else htf_signal
        mtf_val = int(mtf_signal) if isinstance(mtf_signal, SignalState) else mtf_signal
        ltf_val = int(ltf_signal) if isinstance(ltf_signal, SignalState) else ltf_signal
        
        # REASON: Support zowel 3 als 4 timeframes voor backwards compatibility
        if utf_signal is not None:
            utf_val = int(utf_signal) if isinstance(utf_signal, SignalState) else utf_signal
            # Calculate raw score with integer arithmetic (4 timeframes)
            # HTF=50%, MTF=25%, LTF=15%, UTF=10%
            raw_score = htf_val * 50 + mtf_val * 25 + ltf_val * 15 + utf_val * 10
            # Range: -200 (all -2) tot +200 (all +2)
        else:
            # Backwards compatibility: 3 timeframes met originele gewichten
            raw_score = htf_val * 60 + mtf_val * 30 + ltf_val * 10
            # Range: -200 (all -2) tot +200 (all +2)
        
        # Normalize to 0.0-1.0 range
        normalized = (raw_score + 200.0) / 400.0
        
        return normalized
    
    def classify_signals_dataframe(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classificeer concordance voor DataFrame met multi-timeframe signals.
        
        Args:
            signals_df: DataFrame met htf_signal_state, mtf_signal_state, ltf_signal_state, utf_signal_state columns
                       OR rsi_signal_d, rsi_signal_240, rsi_signal_60, rsi_signal_1 style columns
            
        Returns:
            DataFrame met toegevoegde concordance_scenario en concordance_score columns
        """
        if signals_df.empty:
            return signals_df
        
        # Support both old and new column naming
        htf_col = None
        mtf_col = None
        ltf_col = None
        utf_col = None  # REASON: Toegevoegd voor 4-timeframe support
        
        if 'htf_signal_state' in signals_df.columns:
            htf_col = 'htf_signal_state'
            mtf_col = 'mtf_signal_state'
            ltf_col = 'ltf_signal_state'
            utf_col = 'utf_signal_state' if 'utf_signal_state' in signals_df.columns else None
        elif 'rsi_signal_d' in signals_df.columns:
            # Use RSI as representative signal (can be changed)
            htf_col = 'rsi_signal_d'
            mtf_col = 'rsi_signal_240'
            ltf_col = 'rsi_signal_60'
            utf_col = 'rsi_signal_1' if 'rsi_signal_1' in signals_df.columns else None
        
        if not htf_col:
            raise ValueError("Missing required signal columns")
        
        # Apply classification
        def classify_row(row):
            try:
                htf = SignalState(row[htf_col]) if isinstance(row[htf_col], int) else row[htf_col]
                mtf = SignalState(row[mtf_col]) if isinstance(row[mtf_col], int) else row[mtf_col]
                ltf = SignalState(row[ltf_col]) if isinstance(row[ltf_col], int) else row[ltf_col]
                
                # REASON: UTF optioneel voor backwards compatibility
                utf = None
                if utf_col and utf_col in row.index and pd.notna(row[utf_col]):
                    utf = SignalState(row[utf_col]) if isinstance(row[utf_col], int) else row[utf_col]
                
                scenario = self.classify_scenario(htf, mtf, ltf, utf)
                score = self.calculate_concordance_score(htf, mtf, ltf, utf)
                
                return pd.Series({
                    'concordance_scenario': scenario.value,
                    'concordance_score': score
                })
            except Exception as e:
                logger.warning(f"Failed to classify row: {e}")
                return pd.Series({
                    'concordance_scenario': ConcordanceScenario.NEUTRAL.value,
                    'concordance_score': 0.5
                })
        
        concordance_data = signals_df.apply(classify_row, axis=1)
        result_df = pd.concat([signals_df, concordance_data], axis=1)
        
        logger.info(f"Classified {len(result_df)} signal rows (UTF {'enabled' if utf_col else 'disabled'})")
        return result_df
    
    def get_concordance_distribution(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyseer concordance distributie in dataset."""
        if 'concordance_scenario' not in signals_df.columns:
            signals_df = self.classify_signals_dataframe(signals_df)
        
        scenario_counts = signals_df['concordance_scenario'].value_counts()
        total_count = len(signals_df)
        
        distribution = {}
        for scenario in ConcordanceScenario:
            count = scenario_counts.get(scenario.value, 0)
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            distribution[scenario.value] = {
                'count': count,
                'percentage': percentage
            }
        
        if 'concordance_score' in signals_df.columns:
            score_stats = {
                'mean_score': float(signals_df['concordance_score'].mean()),
                'median_score': float(signals_df['concordance_score'].median()),
                'std_score': float(signals_df['concordance_score'].std()),
                'min_score': float(signals_df['concordance_score'].min()),
                'max_score': float(signals_df['concordance_score'].max())
            }
        else:
            score_stats = {}
        
        return {
            'total_signals': total_count,
            'scenario_distribution': distribution,
            'score_statistics': score_stats,
            'weights': {
                'structural': self.structural_weight,
                'tactical': self.tactical_weight,
                'entry': self.entry_weight,
                'utf': self.utf_weight
            }
        }
    
    def create_evidence_node_data(self, 
                                 htf_signal: SignalState,
                                 mtf_signal: SignalState,
                                 ltf_signal: SignalState,
                                 utf_signal: Optional[SignalState] = None) -> Dict[str, Any]:
        """
        Creëer evidence node data voor Bayesian network integratie.
        
        Args:
            htf_signal: Higher timeframe signal state (Daily)
            mtf_signal: Medium timeframe signal state (4H)
            ltf_signal: Lower timeframe signal state (1H)
            utf_signal: Micro timeframe signal state (1m), optional
        
        Returns:
            Dict met evidence data voor pgmpy inference
        """
        scenario = self.classify_scenario(htf_signal, mtf_signal, ltf_signal, utf_signal)
        score = self.calculate_concordance_score(htf_signal, mtf_signal, ltf_signal, utf_signal)
        
        # Map scenario to discrete evidence state
        if score >= 0.8:
            evidence_state = "High_Concordance"
        elif score >= 0.6:
            evidence_state = "Medium_Concordance"
        elif score >= 0.4:
            evidence_state = "Low_Concordance"
        else:
            evidence_state = "No_Concordance"
        
        # Get int values for signal breakdown
        htf_val = int(htf_signal) if isinstance(htf_signal, SignalState) else htf_signal
        mtf_val = int(mtf_signal) if isinstance(mtf_signal, SignalState) else mtf_signal
        ltf_val = int(ltf_signal) if isinstance(ltf_signal, SignalState) else ltf_signal
        utf_val = None
        if utf_signal is not None:
            utf_val = int(utf_signal) if isinstance(utf_signal, SignalState) else utf_signal
        
        signal_breakdown = {
            'htf': htf_val,
            'mtf': mtf_val,
            'ltf': ltf_val
        }
        if utf_val is not None:
            signal_breakdown['utf'] = utf_val
        
        return {
            'concordance_evidence': evidence_state,
            'concordance_scenario': scenario.value,
            'concordance_score': score,
            'signal_breakdown': signal_breakdown,
            'weights_applied': {
                'structural': self.structural_weight,
                'tactical': self.tactical_weight,
                'entry': self.entry_weight,
                'utf': self.utf_weight
            }
        }
    
    def update_weights(self, structural: float, tactical: float, entry: float, utf: float = 0.0):
        """Update concordance weights (config-driven tuning)."""
        total = structural + tactical + entry + utf
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.structural_weight = structural
        self.tactical_weight = tactical  
        self.entry_weight = entry
        self.utf_weight = utf
        
        logger.info(f"Updated weights: S={structural}, T={tactical}, E={entry}, U={utf}")


def create_concordance_matrix(structural_weight: float = 0.50,
                            tactical_weight: float = 0.25,
                            entry_weight: float = 0.15,
                            utf_weight: float = 0.10) -> ConcordanceMatrix:
    """Factory function voor ConcordanceMatrix met 4 timeframes"""
    return ConcordanceMatrix(structural_weight, tactical_weight, entry_weight, utf_weight)

