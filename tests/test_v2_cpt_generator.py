import pytest
import pandas as pd
import numpy as np
from inference.qbn_v2_cpt_generator import QBNv2CPTGenerator
from inference.node_types import SemanticClass

@pytest.fixture
def generator():
    return QBNv2CPTGenerator()

def test_regime_classification(generator):
    # Bullish
    row = pd.Series({'macd_signal_d': 1, 'adx_signal_d': 1})
    assert generator._classify_regime(row) == 'bullish_trend'
    
    # Bearish
    row = pd.Series({'macd_signal_d': -1, 'adx_signal_d': 1})
    assert generator._classify_regime(row) == 'bearish_trend'
    
    # Ranging (ADX 0)
    row = pd.Series({'macd_signal_d': 1, 'adx_signal_d': 0})
    assert generator._classify_regime(row) == 'ranging'

def test_entry_timing_derivation(generator):
    data = pd.DataFrame({
        'outcome_1h': [3, 2, 0, -2, -3]
    })
    timings = generator._derive_entry_timing(data)
    assert timings.tolist() == ['excellent', 'good', 'poor', 'good', 'excellent']

def test_validation_logic(generator):
    cpt_data = {
        'conditional_probabilities': {
            'combo1': {'state1': 0.5, 'state2': 0.5},
            'combo2': {'state1': 0.0, 'state2': 0.0},
            'combo3': {'state1': 0.0, 'state2': 0.0}
        },
        'total_observations': 100
    }
    result = generator._validate_cpt_coverage(cpt_data)
    # 2/6 cells filled = 0.333
    assert round(result.coverage, 2) == 0.33
    assert result.recommendation == 'reduce_states'

