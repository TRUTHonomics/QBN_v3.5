"""
Trade-Aligned Inference Engine voor QBN v3.4 (Direct Sub-Predictions Architecture).

Implementeert forward inference voor de v3.4 architectuur:
- HTF_Regime (root) → 5/11 states
- Trade_Hypothesis (Entry) → Gebaseerd op Leading_Composite
- Multi-horizon predictions (1h, 4h, 1d) → 5 barrier states

v3.4 WIJZIGINGEN:
- Position_Prediction direct gekoppeld aan MP/VR/ET (27 combinaties)
- Alle sub-predictions gaan direct naar TSEM

v3.2 LEGACY:
- Position_Confidence + Time + PnL flow
"""

import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

from .node_types import CompositeState, OutcomeState, RegimeState, SemanticClass
from .regime_detector import HTFRegimeDetector
from .signal_aggregator import SignalAggregator
from .network_structure import QBNv3NetworkStructure
from .cpt_generator import OUTCOME_ATR_MIDPOINTS
from .barrier_config import BarrierPrediction, BarrierInferenceResult
from .position_prediction_generator import PositionPredictionResult, PositionPredictionGenerator

# v3 Generators voor deterministische mapping
from .trade_hypothesis_generator import TradeHypothesisGenerator
from .position_confidence_generator import PositionConfidenceGenerator

# v3.4 Generators voor Direct Sub-Predictions Architecture
from .momentum_prediction_generator import MomentumPredictionGenerator
from .volatility_regime_generator import VolatilityRegimeGenerator
from .exit_timing_generator import ExitTimingGenerator

if TYPE_CHECKING:
    from config.threshold_loader import ThresholdLoader

logger = logging.getLogger(__name__)

@dataclass
class SignalEvidence:
    """Evidence van actieve signalen voor inference."""
    asset_id: int
    timestamp: Any
    symbol: str = ""  # Voor TSEM signaling

    # Signal activations per semantic class
    leading_signals: Dict[str, int] = field(default_factory=dict)
    coincident_signals: Dict[str, int] = field(default_factory=dict)
    confirming_signals: Dict[str, int] = field(default_factory=dict)

    # HTF indicators voor regime detectie
    adx_d: Optional[float] = None
    adx_240: Optional[float] = None
    di_plus_d: Optional[float] = None
    di_minus_d: Optional[float] = None
    macd_histogram_d: Optional[float] = None

    # Data completeness
    rolling_60m_completeness: float = 1.0

@dataclass
class PositionContext:
    """Context voor actieve positie."""
    position_id: str
    direction: str  # 'LONG' of 'SHORT'
    entry_time: datetime
    entry_price: float
    current_price: float
    atr_at_entry: float
    
    # v3.3: Entry-time composite scores voor delta berekening
    entry_leading_score: float = 0.0
    entry_coincident_score: float = 0.0
    entry_confirming_score: float = 0.0
    
    @property
    def time_since_entry_min(self) -> int:
        """Minuten sinds entry."""
        from datetime import timezone
        delta = datetime.now(timezone.utc) - self.entry_time
        return int(delta.total_seconds() / 60)
    
    @property
    def current_pnl_atr(self) -> float:
        """Huidige PnL in ATR units."""
        price_change = self.current_price - self.entry_price
        if self.direction.upper() == 'SHORT':
            price_change = -price_change
        return price_change / (self.atr_at_entry or 1.0)

@dataclass
class DualInferenceResult:
    """Resultaat van v3.1 Dual-Prediction inference."""
    asset_id: int
    timestamp: datetime
    
    # Entry Predictions (voor nieuwe trades)
    entry_predictions: Dict[str, str] = field(default_factory=dict) # horizon -> state
    entry_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    expected_atr_moves: Dict[str, float] = field(default_factory=dict)
    entry_confidences: Dict[str, float] = field(default_factory=dict)
    
    # Position Prediction (voor actieve posities, optioneel)
    position_prediction: Optional[PositionPredictionResult] = None
    
    # Node States
    regime: str = ""
    leading_composite: str = ""
    coincident_composite: str = ""
    confirming_composite: str = ""
    trade_hypothesis: str = ""
    position_confidence: str = ""
    position_confidence_score: float = 0.0
    position_confidence_distribution: Dict[str, float] = field(default_factory=dict)
    
    # v3.4 Position Sub-Predictions (direct naar TSEM)
    momentum_prediction: str = ""
    momentum_distribution: Dict[str, float] = field(default_factory=dict)
    volatility_regime: str = ""
    volatility_distribution: Dict[str, float] = field(default_factory=dict)
    exit_timing: str = ""
    exit_timing_distribution: Dict[str, float] = field(default_factory=dict)
    
    # v3.3 Delta scores (voor debugging)
    delta_leading: float = 0.0
    delta_coincident: float = 0.0
    delta_confirming: float = 0.0

    # Metadata
    inference_time_ms: float = 0.0
    model_version: str = "3.1"

class TradeAlignedInference:
    """
    Trade-Aligned Inference Engine (QBN v3).
    
    Voert forward inference uit op de v3 DAG structuur.
    """
    
    def __init__(
        self, 
        cpts: Dict[str, Dict[str, Any]], 
        signal_classification: Dict[str, Dict],
        threshold_loader: Optional['ThresholdLoader'] = None,
        position_prediction_cpt: Optional[Dict] = None,
        v33_cpts: Optional[Dict[str, Dict[str, Any]]] = None,
        use_v34_mode: bool = True  # v3.4: Direct sub-predictions (zonder RAC)
    ):
        """
        Initialiseer Trade-Aligned Inference Engine.
        
        Args:
            cpts: Dictionary met CPTs per node
            signal_classification: Signal classificatie met weights
            threshold_loader: Optional ThresholdLoader voor DB-driven thresholds.
            position_prediction_cpt: Optionele CPT voor Position_Prediction node.
            v33_cpts: v3.4 Sub-prediction CPTs (Momentum, Volatility, Exit).
            use_v34_mode: Altijd True in v3.4 (behouden voor compatibility)
        """
        self.cpts = cpts
        self.signal_classification = signal_classification
        self._threshold_loader = threshold_loader
        self.v33_cpts = v33_cpts or {}
        self.use_v34_mode = use_v34_mode
        
        self.regime_detector = HTFRegimeDetector()
        self.signal_aggregator = SignalAggregator(signal_classification, threshold_loader=threshold_loader)
        self.network = QBNv3NetworkStructure()
        
        # v3 Sub-generators
        self.hypothesis_gen = TradeHypothesisGenerator()
        self.position_gen = PositionConfidenceGenerator()
        
        # v3.4 Sub-generators (MP, VR, ET)
        self.momentum_gen = None
        self.volatility_gen = None
        self.exit_timing_gen = None
        
        if self.v33_cpts:
            # Initialize sub-prediction generators met hun CPTs
            if 'Momentum_Prediction' in self.v33_cpts:
                self.momentum_gen = MomentumPredictionGenerator()
                self.momentum_gen._cpt = self._parse_v33_cpt(
                    self.v33_cpts['Momentum_Prediction'], tuple_size=2
                )
            
            if 'Volatility_Regime' in self.v33_cpts:
                self.volatility_gen = VolatilityRegimeGenerator()
                self.volatility_gen._cpt = self._parse_v33_cpt(
                    self.v33_cpts['Volatility_Regime'], tuple_size=2
                )
            
            if 'Exit_Timing' in self.v33_cpts:
                self.exit_timing_gen = ExitTimingGenerator()
                self.exit_timing_gen._cpt = self._parse_v33_cpt(
                    self.v33_cpts['Exit_Timing'], tuple_size=3
                )
            
            logger.info(f"Position generators initialized (v3.4-direct): "
                       f"momentum={self.momentum_gen is not None}, "
                       f"volatility={self.volatility_gen is not None}, "
                       f"exit={self.exit_timing_gen is not None}")
        
        # Position Prediction Generator
        self.position_pred_gen = None
        if position_prediction_cpt:
            # v3.4: Detect mode based on parents in CPT
            cpt_parents = position_prediction_cpt.get('parents', [])
            is_v34_cpt = 'Momentum_Prediction' in cpt_parents
            
            self.position_pred_gen = PositionPredictionGenerator(use_v34_mode=is_v34_cpt)
            raw_probs = position_prediction_cpt.get('conditional_probabilities', {})
            parsed_cpt = {}
            for k, v in raw_probs.items():
                # Split key "mp|vr|et" of "conf|time|pnl" -> tuple
                parts = tuple(k.split('|'))
                if len(parts) == 3:
                    parsed_cpt[parts] = v
            self.position_pred_gen._cpt = parsed_cpt
            logger.info(f"Position_Prediction CPT loaded: {'v3.4' if is_v34_cpt else 'legacy'} mode, "
                       f"{len(parsed_cpt)} keys")
        
        self.outcome_states = OutcomeState.state_names()
        
        threshold_source = threshold_loader.source if threshold_loader else "fallback"
        logger.info(f"Trade-Aligned Inference initialized (v3.4) with {len(cpts)} CPTs "
                   f"(thresholds: {threshold_source})")

    def run_inference(
        self, 
        evidence: SignalEvidence,
        position_context: Optional[PositionContext] = None
    ) -> DualInferenceResult:
        """
        Run v3.4 Dual-Prediction inference.
        
        Args:
            evidence: Signal evidence voor huidige moment
            position_context: Optionele positie context (indien actieve positie)
        
        v3.4 ARCHITECTURE:
        Als sub-prediction CPTs beschikbaar zijn EN position_context bestaat:
        - Direct Sub-Predictions flow:
          1. Momentum_Prediction (Delta_Leading, Time)
          2. Volatility_Regime (Delta_Coincident, Time)
          3. Exit_Timing (Delta_Confirming, Time, PnL)
          4. Position_Prediction (MP, VR, ET)
        
        v3.2 FALLBACK:
        Als sub-prediction CPTs niet beschikbaar zijn:
        - Position_Confidence (Coincident, Confirming, Time)
        - Position_Prediction (Position_Confidence, Time, PnL)
        
        Returns:
            DualInferenceResult met entry + position predictions
        """
        start_time = time.perf_counter()
        
        # 1. Shared Nodes
        regime = self._compute_regime(evidence)
        composites = self._compute_composites(evidence, horizon='1h')
        leading_val = composites[SemanticClass.LEADING].value
        coinc_val = composites[SemanticClass.COINCIDENT].value
        conf_val = composites[SemanticClass.CONFIRMING].value
        
        hypothesis = self.hypothesis_gen.derive_hypothesis(leading_val)

        # 2. Entry Branch (Predictions)
        entry_results = {}
        for horizon in ['1h', '4h', '1d']:
            entry_results[horizon] = self._compute_prediction(horizon, regime, hypothesis)
            
        # 3. Position Branch
        position_prediction = None
        position_confidence = ""
        position_confidence_score = 0.0
        position_confidence_dist = {}
        
        # v3.4 position outputs (direct sub-predictions)
        momentum_pred = ""
        momentum_dist = {}
        volatility_regime = ""
        volatility_dist = {}
        exit_timing = ""
        exit_timing_dist = {}
        delta_leading = 0.0
        delta_coincident = 0.0
        delta_confirming = 0.0
        
        if position_context:
            # v3.4: Direct Sub-Predictions flow
            if self._has_subprediction_generators():
                pos_result = self._compute_position_v34(
                    composites,
                    position_context
                )
                
                momentum_pred = pos_result['momentum_prediction']
                momentum_dist = pos_result['momentum_distribution']
                volatility_regime = pos_result['volatility_regime']
                volatility_dist = pos_result['volatility_distribution']
                exit_timing = pos_result['exit_timing']
                exit_timing_dist = pos_result['exit_timing_distribution']
                delta_leading = pos_result['delta_leading']
                delta_coincident = pos_result['delta_coincident']
                delta_confirming = pos_result['delta_confirming']
                
                # Position Prediction
                if hasattr(self, 'position_pred_gen') and self.position_pred_gen:
                    position_prediction = pos_result.get('position_prediction')
                
                # Map naar legacy position_confidence (MP -> confidence)
                position_confidence = self._map_momentum_to_confidence(momentum_pred)
            else:
                # v3.2 DEPRECATED Fallback: Data-driven Position Confidence
                # REASON: Blijft bestaan voor backward compatibility met runs zonder v3.4 CPTs
                logger.warning("⚠️ Using DEPRECATED v3.2 Position_Confidence fallback (no v3.4 CPTs found)")
                position_confidence, position_confidence_score, position_confidence_dist = self.position_gen.get_confidence(
                    coinc_val,
                    conf_val,
                    position_context.time_since_entry_min
                )
                
                # Position Prediction (legacy)
                if hasattr(self, 'position_pred_gen') and self.position_pred_gen:
                    position_prediction = self.position_pred_gen.predict(
                        confidence=position_confidence,
                        time_since_entry_min=position_context.time_since_entry_min,
                        current_pnl_atr=position_context.current_pnl_atr
                    )
        else:
            # T=0 case: initiële confidence (voor entry check)
            # DEPRECATED: v3.2 fallback voor backward compatibility
            position_confidence, position_confidence_score, position_confidence_dist = self.position_gen.get_confidence(
                coinc_val,
                conf_val,
                0
            )
            
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return DualInferenceResult(
            asset_id=evidence.asset_id,
            timestamp=evidence.timestamp,
            entry_predictions={h: p['state'] for h, p in entry_results.items()},
            entry_distributions={h: p['distribution'] for h, p in entry_results.items()},
            expected_atr_moves={h: p['expected_atr'] for h, p in entry_results.items()},
            entry_confidences={h: p['confidence'] for h, p in entry_results.items()},
            position_prediction=position_prediction,
            regime=regime,
            leading_composite=leading_val,
            coincident_composite=coinc_val,
            confirming_composite=conf_val,
            trade_hypothesis=hypothesis,
            position_confidence=position_confidence,
            position_confidence_score=position_confidence_score,
            position_confidence_distribution=position_confidence_dist,
            momentum_prediction=momentum_pred,
            momentum_distribution=momentum_dist,
            volatility_regime=volatility_regime,
            volatility_distribution=volatility_dist,
            exit_timing=exit_timing,
            exit_timing_distribution=exit_timing_dist,
            delta_leading=delta_leading,
            delta_coincident=delta_coincident,
            delta_confirming=delta_confirming,
            inference_time_ms=inference_time,
            model_version="3.4" if self.use_v34_mode else "3.3"
        )

    def infer(self, evidence: SignalEvidence) -> BarrierInferenceResult:
        """Voer v3 forward inference uit (Legacy wrapper voor v3.1)."""
        # Run the new inference
        dual_result = self.run_inference(evidence)
        
        # Map to BarrierInferenceResult for backwards compatibility where expected
        predictions = {}
        for window in ['1h', '4h', '1d']:
            dist = dual_result.entry_distributions.get(window, {})
            predictions[window] = BarrierPrediction(
                window_minutes={'1h': 60, '4h': 240, '1d': 1440}[window],
                p_up_strong=dist.get('up_strong', 0.2),
                p_up_weak=dist.get('up_weak', 0.2),
                p_neutral=dist.get('neutral', 0.2),
                p_down_weak=dist.get('down_weak', 0.2),
                p_down_strong=dist.get('down_strong', 0.2)
            )

        return BarrierInferenceResult(
            asset_id=evidence.asset_id,
            timestamp=evidence.timestamp,
            regime=dual_result.regime,
            leading_composite=dual_result.leading_composite,
            coincident_composite=dual_result.coincident_composite,
            confirming_composite=dual_result.confirming_composite,
            trade_hypothesis=dual_result.trade_hypothesis,
            position_confidence=dual_result.position_confidence,
            predictions=predictions,
            inference_time_ms=dual_result.inference_time_ms,
            model_version="3.1-dual"
        )

    def _compute_regime(self, evidence: SignalEvidence) -> str:
        regime = self.regime_detector.detect_regime(
            adx_signal_d=evidence.adx_d,
            adx_signal_240=evidence.adx_240
        )
        regime_val = regime.value
        
        from config.bayesian_config import QBNv2Config
        from .state_reduction import reduce_regime_state
        if QBNv2Config().use_reduced_regimes:
            regime_val = reduce_regime_state(regime_val)
        return regime_val

    def _compute_composites(self, evidence: SignalEvidence, horizon: str) -> Dict[SemanticClass, CompositeState]:
        all_signals = {**evidence.leading_signals, **evidence.coincident_signals, **evidence.confirming_signals}
        return self.signal_aggregator.aggregate_all_classes(all_signals, horizon=horizon)

    def _compute_prediction(self, horizon: str, regime: str, hypothesis: str, confidence: str = None) -> Dict[str, Any]:
        """
        Inference voor Target nodes: P(Target | Regime, Hypothesis).

        v3.1: Entry_Confidence is verwijderd als parent. De confidence parameter
        wordt genegeerd en is alleen aanwezig voor backwards compatibility.
        """
        node_name = f"Prediction_{horizon}"
        cpt = self.cpts.get(node_name)

        # v3.1: CPT keys zijn nu 2-tuple (regime|hypothesis)
        # Entry_Confidence is verwijderd uit de prediction parents
        parent_key = f"{regime}|{hypothesis}"
        
        cond_probs = cpt.get('conditional_probabilities', {}) if cpt else {}
        distribution = cond_probs.get(parent_key)
        
        if not distribution:
            # Fallback naar prior of uniform
            if cpt and 'probabilities' in cpt:
                distribution = cpt['probabilities']
            else:
                u = 1.0 / len(self.outcome_states)
                distribution = {s: u for s in self.outcome_states}
        
        # Bepaal meest waarschijnlijke state
        best_state = max(distribution, key=distribution.get)
        confidence_val = distribution[best_state]
        expected_atr = self._calculate_expected_atr(distribution)
        
        return {
            'state': best_state,
            'distribution': distribution,
            'expected_atr': expected_atr,
            'confidence': confidence_val
        }

    def _compute_barrier_prediction(
        self,
        window: str,  # '1h', '4h', '1d'
        regime: str,
        hypothesis: str,
        confidence: str = None  # v3.1: deprecated, niet meer gebruikt
    ) -> BarrierPrediction:
        """
        Inference voor barrier-based prediction.

        v3.1: Entry_Confidence is verwijderd als parent. De confidence parameter
        wordt genegeerd voor backwards compatibility.

        Returns:
            BarrierPrediction met probability distribution
        """
        window_map = {'1h': 60, '4h': 240, '1d': 1440}
        window_min = window_map[window]

        node_name = f"Prediction_{window}"
        cpt = self.cpts.get(node_name)

        if not cpt:
            # Fallback naar uniform (of prior als beschikbaar)
            distribution = {
                'up_strong': 0.20,
                'up_weak': 0.20,
                'neutral': 0.20,
                'down_weak': 0.20,
                'down_strong': 0.20
            }

            return BarrierPrediction(
                window_minutes=window_min,
                p_up_strong=distribution.get('up_strong', 0.20),
                p_up_weak=distribution.get('up_weak', 0.20),
                p_neutral=distribution.get('neutral', 0.20),
                p_down_weak=distribution.get('down_weak', 0.20),
                p_down_strong=distribution.get('down_strong', 0.20)
            )

        # v3.1: CPT lookup met 2-tuple key (Entry_Confidence verwijderd)
        parent_key = f"{regime}|{hypothesis}"
        distribution = cpt.get('conditional_probabilities', {}).get(parent_key)
        
        if not distribution:
            # Fallback naar prior
            distribution = cpt.get('probabilities', {
                'up_strong': 0.15,
                'up_weak': 0.20,
                'neutral': 0.30,
                'down_weak': 0.20,
                'down_strong': 0.15
            })
        
        return BarrierPrediction(
            window_minutes=window_min,
            p_up_strong=distribution.get('up_strong', 0.15),
            p_up_weak=distribution.get('up_weak', 0.20),
            p_neutral=distribution.get('neutral', 0.30),
            p_down_weak=distribution.get('down_weak', 0.20),
            p_down_strong=distribution.get('down_strong', 0.15)
        )

    def _calculate_expected_atr(self, distribution: Dict[str, float]) -> float:
        expected = 0.0
        state_to_id = {
            'Strong_Bearish': -3, 'Bearish': -2, 'Slight_Bearish': -1,
            'Neutral': 0,
            'Slight_Bullish': 1, 'Bullish': 2, 'Strong_Bullish': 3
        }
        for state_name, prob in distribution.items():
            state_id = state_to_id.get(state_name, 0)
            midpoint = OUTCOME_ATR_MIDPOINTS.get(state_id, 0.0)
            expected += prob * midpoint
        return expected
    
    # =========================================================================
    # v3.4 DIRECT SUB-PREDICTIONS ARCHITECTURE METHODS
    # =========================================================================
    
    def _has_subprediction_generators(self) -> bool:
        """
        Check of sub-prediction generators beschikbaar zijn.
        
        v3.4: MP, VR, ET zijn vereist (RAC niet)
        v3.3: MP, VR, ET, RAC zijn vereist
        """
        base_generators = all([
            self.momentum_gen is not None,
            self.volatility_gen is not None,
            self.exit_timing_gen is not None
        ])
        return base_generators
    
    def _has_v33_generators(self) -> bool:
        """DEPRECATED: Use _has_subprediction_generators() instead."""
        return self._has_subprediction_generators()
    
    def _parse_v33_cpt(self, cpt_data: Dict[str, Any], tuple_size: int) -> Dict[tuple, Dict[str, float]]:
        """
        Parse v3.3 CPT data van database formaat naar generator formaat.
        
        Args:
            cpt_data: CPT data uit database (met 'conditional_probabilities' key)
            tuple_size: Aantal parents (2 of 3)
        
        Returns:
            Dict met tuple keys en probability distributions
        """
        raw_probs = cpt_data.get('conditional_probabilities', {})
        parsed = {}
        
        for key_str, probs in raw_probs.items():
            # Split "state1|state2|state3" -> ("state1", "state2", "state3")
            parts = tuple(key_str.split('|'))
            if len(parts) == tuple_size:
                parsed[parts] = probs
        
        return parsed
    
    def _compute_position_v34(
        self,
        composites: Dict[SemanticClass, CompositeState],
        position_context: PositionContext
    ) -> Dict[str, Any]:
        """
        v3.4 Direct Sub-Predictions Architecture inference voor actieve posities.
        
        Flow:
        1. Bereken delta scores (current - entry)
        2. Discretiseer deltas en time_since_entry
        3. Momentum_Prediction (Delta_Leading, Time)
        4. Volatility_Regime (Delta_Coincident, Time)
        5. Exit_Timing (Delta_Confirming, Time, PnL)
        6. Position_Prediction (MP, VR, ET) - DIRECT zonder RAC
        
        Returns:
            Dict met alle v3.4 outputs (zonder RAC)
        """
        # 1. Bereken delta scores
        current_leading = composites[SemanticClass.LEADING].score
        current_coincident = composites[SemanticClass.COINCIDENT].score
        current_confirming = composites[SemanticClass.CONFIRMING].score
        
        delta_leading = current_leading - position_context.entry_leading_score
        delta_coincident = current_coincident - position_context.entry_coincident_score
        delta_confirming = current_confirming - position_context.entry_confirming_score
        
        # 2. Discretiseer
        delta_lead_state = self.momentum_gen._discretize_delta(
            delta_leading, 
            self.momentum_gen.delta_threshold
        )
        delta_coinc_state = self.volatility_gen._discretize_delta(
            delta_coincident,
            self.volatility_gen.delta_threshold
        )
        delta_conf_state = self.exit_timing_gen._discretize_delta(
            delta_confirming,
            self.exit_timing_gen.delta_threshold
        )
        
        time_bucket = self.momentum_gen._discretize_time(position_context.time_since_entry_min)
        pnl_state = self.exit_timing_gen._discretize_pnl(position_context.current_pnl_atr)
        
        # 3. Sub-predictions
        momentum_dist = self.momentum_gen.predict(delta_lead_state, time_bucket)
        momentum_pred = max(momentum_dist, key=momentum_dist.get)
        
        volatility_dist = self.volatility_gen.predict(delta_coinc_state, time_bucket)
        volatility_regime = max(volatility_dist, key=volatility_dist.get)
        
        exit_dist = self.exit_timing_gen.predict(delta_conf_state, time_bucket, pnl_state)
        exit_timing = max(exit_dist, key=exit_dist.get)
        
        # 4. v3.4: Position_Prediction DIRECT met MP/VR/ET als parents (geen RAC!)
        position_prediction = None
        if hasattr(self, 'position_pred_gen') and self.position_pred_gen:
            position_prediction = self.position_pred_gen.predict(
                momentum_state=momentum_pred,
                volatility_state=volatility_regime,
                exit_timing_state=exit_timing
            )
        
        return {
            'momentum_prediction': momentum_pred,
            'momentum_distribution': momentum_dist,
            'volatility_regime': volatility_regime,
            'volatility_distribution': volatility_dist,
            'exit_timing': exit_timing,
            'exit_timing_distribution': exit_dist,
            'position_prediction': position_prediction,
            'delta_leading': delta_leading,
            'delta_coincident': delta_coincident,
            'delta_confirming': delta_confirming
        }
    
    def _map_momentum_to_confidence(self, momentum: str) -> str:
        """
        v3.4: Map Momentum_Prediction naar legacy position_confidence.
        
        Mapping:
        - bullish -> high
        - neutral -> medium
        - bearish -> low
        """
        mapping = {
            'bullish': 'high',
            'neutral': 'medium',
            'bearish': 'low'
        }
        return mapping.get(momentum, 'medium')

