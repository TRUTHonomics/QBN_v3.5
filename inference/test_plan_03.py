import pandas as pd
import numpy as np
from .position_confidence_generator import PositionConfidenceGenerator

def test_position_confidence_generation():
    print("Testing PositionConfidenceGenerator...")
    gen = PositionConfidenceGenerator(laplace_alpha=1.0)
    
    # Create mock training data
    # Winning events (long + up_strong) -> high confidence
    # Losing events (long + down_strong) -> low confidence
    # Timeout events -> low confidence
    
    data = [
        # High confidence scenario: Coinc=strong_bullish, Conf=strong_bullish, Time=0-1h -> high outcome
        {'coincident_composite': 'strong_bullish', 'confirming_composite': 'strong_bullish', 'time_since_entry_min': 10, 'event_outcome': 'up_strong', 'event_direction': 'long'},
        {'coincident_composite': 'strong_bullish', 'confirming_composite': 'strong_bullish', 'time_since_entry_min': 20, 'event_outcome': 'up_strong', 'event_direction': 'long'},
        
        # Low confidence scenario: Coinc=bearish, Conf=bearish, Time=4-12h -> low outcome
        {'coincident_composite': 'bearish', 'confirming_composite': 'bearish', 'time_since_entry_min': 300, 'event_outcome': 'down_strong', 'event_direction': 'long'},
        {'coincident_composite': 'bearish', 'confirming_composite': 'bearish', 'time_since_entry_min': 400, 'event_outcome': 'timeout', 'event_direction': 'long'},
    ]
    
    df = pd.DataFrame(data)
    
    print("\nGenerating CPT...")
    cpt = gen.generate_cpt(df)
    
    print("\nVerifying inference for High Confidence (strong_bullish, strong_bullish, 10m)...")
    state, dist = gen.get_confidence('strong_bullish', 'strong_bullish', 10)
    print(f"Result: {state}, Distribution: {dist}")
    assert state == 'high'
    
    print("\nVerifying inference for Low Confidence (bearish, bearish, 300m)...")
    state, dist = gen.get_confidence('bearish', 'bearish', 300)
    print(f"Result: {state}, Distribution: {dist}")
    assert state == 'low'
    
    print("\nVerifying Laplace smoothing for unseen combination (neutral, neutral, 5m)...")
    state, dist = gen.get_confidence('neutral', 'neutral', 5)
    print(f"Result: {state}, Distribution: {dist}")
    # Should be uniform-ish
    assert abs(dist['low'] - 1/3) < 0.1
    
    print("\n✅ Plan 03 Verification Successful!")

if __name__ == "__main__":
    test_position_confidence_generation()
