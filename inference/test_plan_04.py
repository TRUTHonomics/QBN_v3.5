import pandas as pd
import numpy as np
from .position_prediction_generator import PositionPredictionGenerator

def test_position_prediction_generation():
    print("Testing PositionPredictionGenerator...")
    gen = PositionPredictionGenerator(laplace_alpha=1.0)
    
    # Create mock training data
    # Winning events (long + up_strong) -> target_hit
    # Losing events (long + down_strong) -> stoploss_hit
    # Timeout events -> timeout
    
    data = [
        # Target hit scenario
        {'position_confidence': 'high', 'time_since_entry_min': 30, 'current_pnl_atr': 0.5, 'event_outcome': 'up_strong', 'event_direction': 'long'},
        {'position_confidence': 'high', 'time_since_entry_min': 40, 'current_pnl_atr': 0.6, 'event_outcome': 'up_strong', 'event_direction': 'long'},
        
        # Stoploss hit scenario
        {'position_confidence': 'low', 'time_since_entry_min': 60, 'current_pnl_atr': -0.5, 'event_outcome': 'down_strong', 'event_direction': 'long'},
        
        # Timeout scenario
        {'position_confidence': 'medium', 'time_since_entry_min': 730, 'current_pnl_atr': 0.0, 'event_outcome': 'timeout', 'event_direction': 'long'},
    ]
    
    df = pd.DataFrame(data)
    
    print("\nGenerating CPT...")
    cpt = gen.generate_cpt(df)
    
    print("\nVerifying inference for Target Hit (high, 30m, 0.5 ATR)...")
    res = gen.predict('high', 30, 0.5)
    print(f"Result: {res}")
    assert res.dominant_outcome == 'target_hit'
    assert res.target_hit > 0.5
    
    print("\nVerifying inference for Stoploss Hit (low, 60m, -0.5 ATR)...")
    res = gen.predict('low', 60, -0.5)
    print(f"Result: {res}")
    assert res.dominant_outcome == 'stoploss_hit'
    
    print("\nVerifying inference for Timeout (medium, 730m, 0.0 ATR)...")
    res = gen.predict('medium', 730, 0.0)
    print(f"Result: {res}")
    assert res.dominant_outcome == 'timeout'
    
    print("\n✅ Plan 04 Verification Successful!")

if __name__ == "__main__":
    test_position_prediction_generation()
