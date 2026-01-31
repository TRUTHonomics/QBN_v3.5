"""Writer voor QBN inference output naar database + pg_notify.

Schrijft complete QBN inference output naar qbn.output_entry.
De database trigger zorgt automatisch voor pg_notify naar TSEM.

NOTE: QBN levert inference output (states, scores, distributions).
TSEM bepaalt sizing, leverage, stop-loss etc.

Changelog:
- v3.3: position_confidence verwijderd - gaat naar aparte qbn.output_position tabel (Entry/Position splitsing)
- v3.2: Geconsolideerd naar output_entry, bayesian_predictions deprecated
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from database.db import get_cursor


@dataclass
class QBNInferenceOutput:
    """Complete QBN inference output voor opslag in qbn.output_entry."""
    asset_id: int
    symbol: str

    # Trade decision
    trade_hypothesis: str  # no_setup/weak_long/strong_long/weak_short/strong_short

    # Entry confidence (TSEM bepaalt sizing/leverage hierop)
    entry_confidence: str         # low/medium/high
    entry_confidence_score: float # -1.0 to +1.0

    # NOTE: position_confidence is verplaatst naar toekomstige qbn.output_position tabel
    # Zie: .docs/v3_migration_plans/3.1_position_confidence_node.md

    # Context (semantic nodes)
    regime: Optional[str] = None
    leading_composite: Optional[str] = None
    coincident_composite: Optional[str] = None
    confirming_composite: Optional[str] = None

    # Predictions (state names)
    prediction_1h: Optional[str] = None
    prediction_4h: Optional[str] = None
    prediction_1d: Optional[str] = None

    # Full distributions (JSONB) - 7-state probability distributions
    distribution_1h: Optional[Dict[str, float]] = None
    distribution_4h: Optional[Dict[str, float]] = None
    distribution_1d: Optional[Dict[str, float]] = None

    # Per-horizon confidence (max probability from distribution)
    confidence_1h: Optional[float] = None
    confidence_4h: Optional[float] = None
    confidence_1d: Optional[float] = None

    # Expected ATR moves
    expected_atr_1h: Optional[float] = None
    expected_atr_4h: Optional[float] = None
    expected_atr_1d: Optional[float] = None

    # Entry timing distribution
    entry_timing_distribution: Optional[Dict[str, float]] = None

    # Barrier Predictions (v3.3+)
    barrier_prediction_1h: Optional[Dict[str, Any]] = None
    barrier_prediction_4h: Optional[Dict[str, Any]] = None
    barrier_prediction_1d: Optional[Dict[str, Any]] = None
    outcome_mode: str = "point_in_time"

    # Metadata
    inference_time_ms: Optional[float] = None
    model_version: str = "3.3"  # v3.3: Entry/Position output splitsing


# Backward compatibility alias
QBNEntrySignal = QBNInferenceOutput


class QBNOutputWriter:
    """Schrijft complete QBN inference output naar qbn.output_entry tabel."""

    # NOTE: position_confidence verwijderd - gaat naar aparte qbn.output_position tabel (later)
    INSERT_SQL = """
        INSERT INTO qbn.output_entry (
            asset_id, symbol, trade_hypothesis,
            entry_confidence, entry_confidence_score,
            regime, leading_composite, coincident_composite, confirming_composite,
            prediction_1h, prediction_4h, prediction_1d,
            distribution_1h, distribution_4h, distribution_1d,
            confidence_1h, confidence_4h, confidence_1d,
            expected_atr_1h, expected_atr_4h, expected_atr_1d,
            entry_timing_distribution,
            barrier_prediction_1h, barrier_prediction_4h, barrier_prediction_1d,
            outcome_mode,
            inference_time_ms, model_version
        ) VALUES (
            %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s,
            %s, %s, %s,
            %s,
            %s, %s
        )
    """

    def write(self, output: QBNInferenceOutput) -> None:
        """
        Write inference output naar database.

        De database trigger zorgt automatisch voor pg_notify('qbn_signal').
        ALLE inference output wordt geschreven (geen filtering op trade_hypothesis).
        """
        with get_cursor(commit=True) as cur:
            cur.execute(self.INSERT_SQL, (
                output.asset_id,
                output.symbol,
                output.trade_hypothesis,
                output.entry_confidence,
                output.entry_confidence_score,
                # NOTE: position_confidence verwijderd - gaat naar qbn.output_position
                output.regime,
                output.leading_composite,
                output.coincident_composite,
                output.confirming_composite,
                output.prediction_1h,
                output.prediction_4h,
                output.prediction_1d,
                json.dumps(output.distribution_1h) if output.distribution_1h else None,
                json.dumps(output.distribution_4h) if output.distribution_4h else None,
                json.dumps(output.distribution_1d) if output.distribution_1d else None,
                output.confidence_1h,
                output.confidence_4h,
                output.confidence_1d,
                output.expected_atr_1h,
                output.expected_atr_4h,
                output.expected_atr_1d,
                json.dumps(output.entry_timing_distribution) if output.entry_timing_distribution else None,
                json.dumps(output.barrier_prediction_1h) if output.barrier_prediction_1h else None,
                json.dumps(output.barrier_prediction_4h) if output.barrier_prediction_4h else None,
                json.dumps(output.barrier_prediction_1d) if output.barrier_prediction_1d else None,
                output.outcome_mode,
                output.inference_time_ms,
                output.model_version
            ))

    @classmethod
    def from_inference_result(cls, result, symbol: str) -> QBNInferenceOutput:
        """
        Maak QBNInferenceOutput van InferenceResult of BarrierInferenceResult.

        Args:
            result: InferenceResult van TradeAlignedInference
            symbol: Trading symbol (bijv. PI_XBTUSD)
        """
        # REASON: Detecteer of we met een BarrierInferenceResult te maken hebben
        from inference.barrier_config import BarrierInferenceResult, BarrierPrediction
        from inference.trade_aligned_inference import DualInferenceResult
        
        is_barrier = isinstance(result, BarrierInferenceResult)
        is_dual = isinstance(result, DualInferenceResult)
        
        # Basis velden
        output = QBNInferenceOutput(
            asset_id=result.asset_id,
            symbol=symbol,
            trade_hypothesis=result.trade_hypothesis,
            entry_confidence=result.entry_confidence,
            entry_confidence_score=getattr(result, 'entry_confidence_score', 0.0),
            regime=result.regime,
            leading_composite=result.leading_composite,
            coincident_composite=result.coincident_composite,
            confirming_composite=result.confirming_composite,
            inference_time_ms=result.inference_time_ms,
            model_version=getattr(result, 'model_version', '3.3')
        )

        if is_barrier:
            # Vul barrier specifieke velden
            output.outcome_mode = "barrier"
            
            if '1h' in result.predictions:
                p = result.predictions['1h']
                output.barrier_prediction_1h = p.to_dict()
                output.prediction_1h = p.expected_direction
                output.confidence_1h = max(p.p_up_strong + p.p_up_weak, p.p_down_strong + p.p_down_weak, p.p_neutral)
            
            if '4h' in result.predictions:
                p = result.predictions['4h']
                output.barrier_prediction_4h = p.to_dict()
                output.prediction_4h = p.expected_direction
                output.confidence_4h = max(p.p_up_strong + p.p_up_weak, p.p_down_strong + p.p_down_weak, p.p_neutral)
                
            if '1d' in result.predictions:
                p = result.predictions['1d']
                output.barrier_prediction_1d = p.to_dict()
                output.prediction_1d = p.expected_direction
                output.confidence_1d = max(p.p_up_strong + p.p_up_weak, p.p_down_strong + p.p_down_weak, p.p_neutral)
        elif is_dual:
            # Dual-Prediction mapping (v3.1+)
            output.outcome_mode = "dual"
            output.prediction_1h = result.entry_predictions.get('1h')
            output.prediction_4h = result.entry_predictions.get('4h')
            output.prediction_1d = result.entry_predictions.get('1d')
            output.distribution_1h = result.entry_distributions.get('1h')
            output.distribution_4h = result.entry_distributions.get('4h')
            output.distribution_1d = result.entry_distributions.get('1d')
            output.confidence_1h = result.entry_confidences.get('1h')
            output.confidence_4h = result.entry_confidences.get('4h')
            output.confidence_1d = result.entry_confidences.get('1d')
            output.expected_atr_1h = result.expected_atr_moves.get('1h')
            output.expected_atr_4h = result.expected_atr_moves.get('4h')
            output.expected_atr_1d = result.expected_atr_moves.get('1d')
        else:
            # Legacy mapping
            output.outcome_mode = "point_in_time"
            output.prediction_1h = getattr(result, 'predictions', {}).get('1h')
            output.prediction_4h = getattr(result, 'predictions', {}).get('4h')
            output.prediction_1d = getattr(result, 'predictions', {}).get('1d')
            output.distribution_1h = getattr(result, 'distributions', {}).get('1h')
            output.distribution_4h = getattr(result, 'distributions', {}).get('4h')
            output.distribution_1d = getattr(result, 'distributions', {}).get('1d')
            output.confidence_1h = getattr(result, 'confidences', {}).get('1h')
            output.confidence_4h = getattr(result, 'confidences', {}).get('4h')
            output.confidence_1d = getattr(result, 'confidences', {}).get('1d')
            output.expected_atr_1h = getattr(result, 'expected_atr_moves', {}).get('1h')
            output.expected_atr_4h = getattr(result, 'expected_atr_moves', {}).get('1h')
            output.expected_atr_1d = getattr(result, 'expected_atr_moves', {}).get('1h')
            output.entry_timing_distribution = getattr(result, 'entry_timing_distribution', None)

        return output


# Backward compatibility aliases
QBNEntrySignalWriter = QBNOutputWriter
