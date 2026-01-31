import pytest
import time
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock

from inference.trade_aligned_inference import TradeAlignedInference, SignalEvidence, DualInferenceResult
from inference.node_types import OutcomeState, RegimeState, SemanticClass, CompositeState

@pytest.fixture
def mock_cpts():
    """Mock CPTs voor testing."""
    return {
        'Prediction_1h': {
            'conditional_probabilities': {
                'bullish_trend|strong_long': {
                    'up_strong': 0.3, 'up_weak': 0.4, 'neutral': 0.1, 'down_weak': 0.1, 'down_strong': 0.1
                }
            }
        }
    }

@pytest.fixture
def mock_classification():
    """Mock signal classification."""
    return {
        'rsi_signal_1': {'semantic_class': 'LEADING', 'polarity': 'LONG', 'weights': {'1h': 1.0}},
        'macd_signal_1': {'semantic_class': 'COINCIDENT', 'polarity': 'LONG', 'weights': {'1h': 1.0}},
        'adx_signal_1': {'semantic_class': 'CONFIRMING', 'polarity': 'NEUTRAL', 'weights': {'1h': 1.0}}
    }

@pytest.fixture
def engine(mock_cpts, mock_classification):
    return TradeAlignedInference(mock_cpts, mock_classification)

def test_inference_basic(engine):
    """Test of inference een resultaat teruggeeft."""
    evidence = SignalEvidence(
        asset_id=1,
        timestamp=datetime.now(timezone.utc),
        leading_signals={'rsi_signal_1': 1},     # Bullish -> strong_long hypothesis
        coincident_signals={'macd_signal_1': 1}, # Bullish
        confirming_signals={'adx_signal_1': 0},  # Neutral
        adx_d=30.0, # Trending
        di_plus_d=25.0, di_minus_d=15.0 # Bullish trend
    )
    
    result = engine.run_inference(evidence)
    
    assert isinstance(result, DualInferenceResult)
    assert result.asset_id == 1
    assert result.regime == 'bullish_trend'
    assert result.trade_hypothesis == 'strong_long'
    
    # Check predictions
    assert '1h' in result.entry_predictions
    assert result.entry_predictions['1h'] in ['up_strong', 'up_weak', 'neutral', 'down_weak', 'down_strong']

def test_expected_atr_calculation(engine):
    """Test de gewogen ATR berekening."""
    # Dummy distributie voor 5-state (v3.1)
    dist = {
        'up_strong': 1.0, 'up_weak': 0.0, 'neutral': 0.0, 'down_weak': 0.0, 'down_strong': 0.0
    }
    
    # We mocken de ATR midpoints niet, maar testen of de functie draait
    expected = engine._calculate_expected_atr(dist)
    assert isinstance(expected, float)

def test_inference_latency(engine):
    """Test of de latency binnen het target van 25ms blijft."""
    evidence = SignalEvidence(
        asset_id=1,
        timestamp=datetime.now(timezone.utc),
        leading_signals={'rsi_signal_1': 1},
        coincident_signals={'macd_signal_1': 1}
    )
    
    # Warmup
    for _ in range(5):
        engine.run_inference(evidence)
        
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        engine.run_inference(evidence)
    end = time.perf_counter()
    
    avg_latency = (end - start) / iterations * 1000
    print(f"Average latency: {avg_latency:.2f}ms")
    assert avg_latency < 25.0

