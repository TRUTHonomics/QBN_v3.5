import pytest
import pandas as pd
from inference.trade_hypothesis_generator import TradeHypothesisGenerator
from inference.entry_confidence_generator import EntryConfidenceGenerator
from inference.position_confidence_generator import PositionConfidenceGenerator
from inference.trade_aligned_inference import TradeAlignedInference, SignalEvidence

def test_trade_hypothesis_derivation():
    gen = TradeHypothesisGenerator()
    assert gen.derive_hypothesis('strong_bullish') == 'strong_long'
    assert gen.derive_hypothesis('bullish') == 'weak_long'
    assert gen.derive_hypothesis('neutral') == 'no_setup'
    assert gen.derive_hypothesis('bearish') == 'weak_short'
    assert gen.derive_hypothesis('strong_bearish') == 'strong_short'

def test_entry_confidence_derivation():
    gen = EntryConfidenceGenerator()
    # High confidence (aligned)
    assert gen.derive_confidence('bullish', 'strong_bullish') == 'high'
    assert gen.derive_confidence('bearish', 'bearish') == 'high'
    
    # Medium confidence (one neutral)
    assert gen.derive_confidence('neutral', 'bullish') == 'medium'
    assert gen.derive_confidence('bearish', 'neutral') == 'medium'
    
    # Low confidence (opposing)
    assert gen.derive_confidence('bullish', 'bearish') == 'low'
    assert gen.derive_confidence('bearish', 'strong_bullish') == 'low'

def test_position_confidence_derivation():
    gen = PositionConfidenceGenerator()
    assert gen.derive_confidence('strong_bullish') == 'high'
    assert gen.derive_confidence('neutral') == 'medium'
    assert gen.derive_confidence('bearish') == 'low'
    
    params = gen.get_risk_parameters('high')
    assert params['stop_loss_atr'] == 2.0
    assert params['position_size_pct'] == 100

def test_trade_aligned_inference_flow():
    # Mock CPTs
    mock_cpts = {
        'Prediction_1h': {
            'conditional_probabilities': {
                'sync_bullish|strong_long|high': {'Bullish': 0.8, 'Neutral': 0.2}
            }
        }
    }
    mock_classification = {} # Not used for this minimal test
    
    engine = TradeAlignedInference(mock_cpts, mock_classification)
    
    # Evidence mocking (we override detection for testing)
    evidence = SignalEvidence(
        asset_id=1,
        timestamp='2026-01-05T12:00:00'
    )
    
    # Test internal helper directly to avoid full aggregator/detector logic
    regime = 'sync_bullish'
    hypothesis = 'strong_long'
    confidence = 'high'
    
    pred = engine._compute_prediction('1h', regime, hypothesis, confidence)
    assert pred['state'] == 'Bullish'
    assert pred['distribution']['Bullish'] == 0.8

