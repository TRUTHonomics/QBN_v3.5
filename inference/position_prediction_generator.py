"""
Position_Prediction Generator voor QBN v3.4

v3.4 NIEUW:
- Direct Sub-Predictions: Parents zijn nu Momentum_Prediction, Volatility_Regime, Exit_Timing
- Risk_Adjusted_Confidence ensemble VERWIJDERD
- CPT heeft 27 combinaties (3×3×3) in plaats van 5 (via RAC)

v3.2/v3.3 LEGACY (backward compatible):
- Uniqueness weighting (López de Prado) voor serial correlation correctie
- Gewogen counting in CPT generatie

Parents (v3.4): Momentum_Prediction, Volatility_Regime, Exit_Timing
States: target_hit, stoploss_hit, timeout
"""

import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class PositionPredictionResult:
    """Resultaat van Position_Prediction inference."""
    target_hit: float
    stoploss_hit: float
    timeout: float
    dominant_outcome: str
    confidence: float

@dataclass
class PositionPredictionMetrics:
    """Training metrics voor Position_Prediction."""
    observations: int
    weighted_observations: float  # v3.2: gewogen observaties
    target_hit_rate: float
    stoploss_hit_rate: float
    timeout_rate: float
    coverage: float
    avg_entropy: float


class PositionPredictionGenerator:
    """
    v3.4 CPT generator voor Position_Prediction node.
    
    v3.4 ARCHITECTURE:
    - Direct Sub-Predictions: Parents zijn MP, VR, ET (27 combinaties)
    - Risk_Adjusted_Confidence ensemble VERWIJDERD
    
    NIEUW in v3.2+:
    - Uniqueness weighting (1/N per event)
    - Gewogen counting voor CPT
    """
    
    PREDICTION_STATES = ["target_hit", "stoploss_hit", "timeout"]
    
    # v3.4: Direct Sub-Prediction parents
    MOMENTUM_STATES = ["bearish", "neutral", "bullish"]
    VOLATILITY_STATES = ["low_vol", "normal", "high_vol"]
    EXIT_TIMING_STATES = ["exit_now", "hold", "extend"]
    
    # Legacy states (voor backward compatibility)
    CONFIDENCE_STATES = ["very_low", "low", "medium", "high", "very_high"]  # v3.3 RAC
    CONFIDENCE_STATES_LEGACY = ["low", "medium", "high"]  # v3.2 Position_Confidence
    
    TIME_BUCKETS = ["0-1h", "1-4h", "4-12h", "12-24h"]
    PNL_STATES = ["losing", "breakeven", "winning"]
    
    def __init__(self, laplace_alpha: float = 1.0, use_weighted_mode: bool = True, use_v34_mode: bool = True):
        self.laplace_alpha = laplace_alpha
        self.use_weighted_mode = use_weighted_mode
        self.use_v34_mode = use_v34_mode  # v3.4: Direct sub-predictions
        self._cpt: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self._metrics: Optional[PositionPredictionMetrics] = None
    
    def generate_cpt(
        self, 
        training_data: pd.DataFrame,
        direction_column: str = 'event_direction'
    ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        """
        Genereer CPT van event-gefilterde training data.
        
        v3.4 MODE (use_v34_mode=True):
            Parents: Momentum_Prediction, Volatility_Regime, Exit_Timing
            Training data columns:
                - momentum_state: str (bearish/neutral/bullish)
                - volatility_state: str (low_vol/normal/high_vol)
                - exit_timing_state: str (exit_now/hold/extend)
                - event_outcome: str (up_strong, etc.)
                - event_direction: str (long/short)
                - uniqueness_weight: float (optioneel)
        
        LEGACY MODE (use_v34_mode=False):
            Parents: Position_Confidence, Time, PnL
        """
        mode_str = "v3.4-direct" if self.use_v34_mode else "legacy"
        weight_str = "weighted" if self.use_weighted_mode else "unweighted"
        logger.info(f"🔧 Generating Position_Prediction CPT ({mode_str}, {weight_str})...")
        
        if self.use_v34_mode:
            return self._generate_cpt_v34(training_data, direction_column)
        else:
            return self._generate_cpt_legacy(training_data, direction_column)
    
    def _generate_cpt_v34(
        self,
        training_data: pd.DataFrame,
        direction_column: str = 'event_direction'
    ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        """
        v3.4: Genereer CPT met direct sub-prediction parents.
        
        CPT keys: (momentum_state, volatility_state, exit_timing_state)
        27 combinaties (3×3×3)
        """
        counts = defaultdict(lambda: defaultdict(float))
        total_weighted_obs = 0.0
        total_raw_obs = 0
        
        for _, row in training_data.iterrows():
            # v3.4: Direct sub-prediction states
            momentum = row.get('momentum_state', 'neutral')
            volatility = row.get('volatility_state', 'normal')
            exit_timing = row.get('exit_timing_state', 'hold')
            
            # Derive prediction label
            label = self._derive_prediction_label(
                row.get('event_outcome', 'neutral'),
                row.get(direction_column, 'long')
            )
            
            # Uniqueness weight (López de Prado)
            if self.use_weighted_mode:
                weight = row.get('uniqueness_weight', 1.0) or 1.0
            else:
                weight = 1.0
            
            key = (momentum, volatility, exit_timing)
            counts[key][label] += weight
            total_weighted_obs += weight
            total_raw_obs += 1
        
        # Build CPT met Laplace smoothing (27 combinaties)
        cpt = {}
        for mp in self.MOMENTUM_STATES:
            for vr in self.VOLATILITY_STATES:
                for et in self.EXIT_TIMING_STATES:
                    key = (mp, vr, et)
                    state_counts = counts[key]
                    
                    total = sum(state_counts.values()) + \
                            self.laplace_alpha * len(self.PREDICTION_STATES)
                    
                    cpt[key] = {}
                    for state in self.PREDICTION_STATES:
                        count = state_counts.get(state, 0)
                        cpt[key][state] = (count + self.laplace_alpha) / total
        
        self._cpt = cpt
        self._metrics = self._calculate_metrics_v34(counts, total_raw_obs, total_weighted_obs)
        
        logger.info(f"✅ Position_Prediction CPT v3.4 generated: {len(cpt)} keys (27 expected), "
                   f"{total_raw_obs} raw obs, {total_weighted_obs:.1f} weighted obs")
        return cpt
    
    def _generate_cpt_legacy(
        self,
        training_data: pd.DataFrame,
        direction_column: str = 'event_direction'
    ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        """
        Legacy mode: v3.2/v3.3 CPT met confidence + time + pnl parents.
        """
        counts = defaultdict(lambda: defaultdict(float))
        total_weighted_obs = 0.0
        total_raw_obs = 0
        
        for _, row in training_data.iterrows():
            # v3.4: Gebruik nieuwe parent fields (momentum, volatility, exit)
            momentum = row.get('momentum_prediction')
            volatility = row.get('volatility_regime')
            exit_timing = row.get('exit_timing')
            
            # Fallback naar v3.2 confidence als v3.4 fields ontbreken
            if momentum is None:
                conf = row.get('position_confidence', 'medium')
                momentum = self._confidence_to_momentum(conf)
            
            # Time discretization
            time_bucket = row.get('time_bucket')
            if time_bucket is None:
                time_min = row.get('time_since_entry_min', 0) or 0
                time_bucket = self._discretize_time(time_min)
                
            # PNL discretization
            pnl_atr = row.get('current_pnl_atr', 0) or 0
            pnl_state = self._discretize_pnl(pnl_atr)
            
            # Derive prediction label
            label = self._derive_prediction_label(
                row.get('event_outcome', 'neutral'),
                row.get(direction_column, 'long')
            )
            
            # Uniqueness weight
            if self.use_weighted_mode:
                weight = row.get('uniqueness_weight', 1.0) or 1.0
            else:
                weight = 1.0
            
            key = (conf, time_bucket, pnl_state)
            counts[key][label] += weight
            total_weighted_obs += weight
            total_raw_obs += 1
        
        # Build CPT met Laplace smoothing
        cpt = {}
        for conf in self.CONFIDENCE_STATES:
            for time_bucket in self.TIME_BUCKETS:
                for pnl_state in self.PNL_STATES:
                    key = (conf, time_bucket, pnl_state)
                    state_counts = counts[key]
                    
                    total = sum(state_counts.values()) + \
                            self.laplace_alpha * len(self.PREDICTION_STATES)
                    
                    cpt[key] = {}
                    for state in self.PREDICTION_STATES:
                        count = state_counts.get(state, 0)
                        cpt[key][state] = (count + self.laplace_alpha) / total
        
        self._cpt = cpt
        self._metrics = self._calculate_metrics(counts, total_raw_obs, total_weighted_obs)
        
        logger.info(f"✅ Position_Prediction CPT (legacy) generated: {len(cpt)} keys, "
                   f"{total_raw_obs} raw obs, {total_weighted_obs:.1f} weighted obs")
        return cpt
    
    def predict(
        self, 
        momentum_state: str = None,
        volatility_state: str = None,
        exit_timing_state: str = None,
        # Legacy parameters (voor backward compatibility)
        confidence: str = None,
        time_since_entry_min: int = None,
        current_pnl_atr: float = None
    ) -> PositionPredictionResult:
        """
        Voorspel uitkomst voor actieve positie.
        
        v3.4 MODE:
            Args:
                momentum_state: bearish/neutral/bullish
                volatility_state: low_vol/normal/high_vol
                exit_timing_state: exit_now/hold/extend
        
        LEGACY MODE:
            Args:
                confidence: v3.2 position_confidence
                time_since_entry_min: minuten sinds entry
                current_pnl_atr: huidige PnL in ATR units
        """
        # v3.4: Direct sub-prediction parents
        if momentum_state is not None and volatility_state is not None and exit_timing_state is not None:
            key = (momentum_state, volatility_state, exit_timing_state)
        # Legacy: confidence + time + pnl
        elif confidence is not None:
            time_bucket = self._discretize_time(time_since_entry_min or 0)
            pnl_state = self._discretize_pnl(current_pnl_atr or 0)
            key = (confidence, time_bucket, pnl_state)
        else:
            logger.warning("No valid parameters for Position_Prediction, using uniform distribution")
            key = None
        
        distribution = self._cpt.get(key, self._uniform_distribution()) if key else self._uniform_distribution()
        
        # Confidence: 1 - normalized entropy
        probs = list(distribution.values())
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(self.PREDICTION_STATES))
        conf_score = 1 - (entropy / max_entropy)
        
        dominant = max(distribution, key=distribution.get)
        
        return PositionPredictionResult(
            target_hit=distribution['target_hit'],
            stoploss_hit=distribution['stoploss_hit'],
            timeout=distribution['timeout'],
            dominant_outcome=dominant,
            confidence=conf_score
        )
    
    def _derive_prediction_label(self, event_outcome: str, direction: str) -> str:
        """Derive prediction label van event outcome."""
        if event_outcome in ['neutral', 'timeout', 'none']:
            return 'timeout'
        
        outcome_is_up = str(event_outcome).startswith('up_')
        direction_is_long = direction == 'long'
        
        aligned = (direction_is_long and outcome_is_up) or \
                  (not direction_is_long and not outcome_is_up)
        
        return 'target_hit' if aligned else 'stoploss_hit'
    
    def _discretize_pnl(self, pnl_atr: float) -> str:
        """Discretiseer PnL naar state."""
        if pnl_atr < -0.25: return 'losing'
        elif pnl_atr > 0.25: return 'winning'
        else: return 'breakeven'

    def _discretize_time(self, minutes: float) -> str:
        """Convert minuten naar time bucket."""
        if minutes <= 60: return '0-1h'
        elif minutes <= 240: return '1-4h'
        elif minutes <= 720: return '4-12h'
        else: return '12-24h'

    def _uniform_distribution(self) -> Dict[str, float]:
        p = 1.0 / len(self.PREDICTION_STATES)
        return {s: p for s in self.PREDICTION_STATES}

    def _calculate_metrics(
        self, 
        counts: Dict, 
        total_raw_obs: int,
        total_weighted_obs: float
    ) -> PositionPredictionMetrics:
        """Calculate metrics for legacy CPT (confidence + time + pnl)."""
        actual_keys = sum(1 for k in counts if sum(counts[k].values()) > 0)
        expected_keys = len(self.CONFIDENCE_STATES) * len(self.TIME_BUCKETS) * len(self.PNL_STATES)
        
        # Gewogen totalen
        total_target = sum(c.get('target_hit', 0) for c in counts.values())
        total_sl = sum(c.get('stoploss_hit', 0) for c in counts.values())
        total_timeout = sum(c.get('timeout', 0) for c in counts.values())
        
        entropies = []
        for dist in self._cpt.values():
            probs = list(dist.values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            entropies.append(entropy)
        
        # Gebruik weighted obs voor rates
        denom = max(1.0, total_weighted_obs)
            
        return PositionPredictionMetrics(
            observations=total_raw_obs,
            weighted_observations=total_weighted_obs,
            target_hit_rate=total_target / denom,
            stoploss_hit_rate=total_sl / denom,
            timeout_rate=total_timeout / denom,
            coverage=actual_keys / expected_keys,
            avg_entropy=np.mean(entropies) if entropies else 0
        )
    
    def _calculate_metrics_v34(
        self, 
        counts: Dict, 
        total_raw_obs: int,
        total_weighted_obs: float
    ) -> PositionPredictionMetrics:
        """Calculate metrics for v3.4 CPT (MP × VR × ET = 27 combinaties)."""
        actual_keys = sum(1 for k in counts if sum(counts[k].values()) > 0)
        expected_keys = len(self.MOMENTUM_STATES) * len(self.VOLATILITY_STATES) * len(self.EXIT_TIMING_STATES)  # 27
        
        # Gewogen totalen
        total_target = sum(c.get('target_hit', 0) for c in counts.values())
        total_sl = sum(c.get('stoploss_hit', 0) for c in counts.values())
        total_timeout = sum(c.get('timeout', 0) for c in counts.values())
        
        entropies = []
        for dist in self._cpt.values():
            probs = list(dist.values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            entropies.append(entropy)
        
        # Gebruik weighted obs voor rates
        denom = max(1.0, total_weighted_obs)
            
        return PositionPredictionMetrics(
            observations=total_raw_obs,
            weighted_observations=total_weighted_obs,
            target_hit_rate=total_target / denom,
            stoploss_hit_rate=total_sl / denom,
            timeout_rate=total_timeout / denom,
            coverage=actual_keys / expected_keys,
            avg_entropy=np.mean(entropies) if entropies else 0
        )

    def save_cpt(self, asset_id: int, run_id: str):
        """Sla CPT op in database."""
        from inference.cpt_cache_manager import CPTCacheManager
        cache_manager = CPTCacheManager()
        
        # v3.4: Direct sub-predictions
        if self.use_v34_mode:
            version = '3.4'
            parents = ['Momentum_Prediction', 'Volatility_Regime', 'Exit_Timing']
            training_method = 'weighted_direct_subpredictions' if self.use_weighted_mode else 'direct_subpredictions'
        else:
            version = '3.2' if self.use_weighted_mode else '3.1'
            parents = ['Position_Confidence', 'Time_Since_Entry', 'Current_PnL_ATR']
            training_method = 'weighted_event_driven' if self.use_weighted_mode else 'event_driven'
        
        cpt_data = {
            'node_name': 'Position_Prediction',
            'parents': parents,
            'states': self.PREDICTION_STATES,
            'conditional_probabilities': {
                f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self._cpt.items()
            },
            'metadata': {
                'version': version,
                'training_method': training_method,
                'laplace_alpha': self.laplace_alpha,
                'weighted_mode': self.use_weighted_mode,
                'v34_mode': self.use_v34_mode,
                'metrics': asdict(self._metrics) if self._metrics else {}
            },
            'observations': self._metrics.observations if self._metrics else 0
        }
        
        cache_manager.save_cpt(
            asset_id=asset_id,
            node_name='Position_Prediction',
            cpt_data=cpt_data,
            scope_type='single',
            scope_key=f"asset_{asset_id}",
            source_assets=[asset_id],
            run_id=run_id
        )
        logger.info(f"💾 Saved Position_Prediction CPT v{version} for asset {asset_id}")
