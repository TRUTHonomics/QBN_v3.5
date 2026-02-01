import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import logging

from core.logging_utils import setup_logging

logger = setup_logging("debug_stability")

from inference.qbn_v2_cpt_generator import QBNv2CPTGenerator
from inference.validation.cpt_validator import CPTValidator
from inference.node_types import RegimeState

def debug_stability():
    asset_id = 1
    lookback_days = 30
    
    generator = QBNv2CPTGenerator(laplace_alpha=0.1)
    validator = CPTValidator()
    
    print(f"--- Debugging Stability for Asset {asset_id} ---")
    
    # 1. Generate current CPTs
    print(f"Generating current CPTs (lookback={lookback_days})...")
    current_cpts = generator.generate_all_cpts(asset_id, lookback_days=lookback_days, save_to_db=False, validate_quality=False)
    current_regime = current_cpts.get('HTF_Regime')
    
    # 2. Generate old CPTs (subset)
    half_lookback = lookback_days // 2
    print(f"Generating old CPTs (lookback={half_lookback})...")
    old_cpts = generator.generate_all_cpts(asset_id, lookback_days=half_lookback, save_to_db=False, validate_quality=False)
    old_regime = old_cpts.get('HTF_Regime')
    
    if not current_regime or not old_regime:
        print("Error: Could not generate regime CPTs")
        return

    print("\n--- Current Regime Probabilities ---")
    for s, p in sorted(current_regime['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {s:25}: {p:.4f}")
        
    print("\n--- Old Regime Probabilities ---")
    for s, p in sorted(old_regime['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {s:25}: {p:.4f}")
        
    stability = validator.calculate_stability(current_regime, old_regime)
    print(f"\nCalculated Stability: {stability:.4f}")
    
    # Test JS Distance manually
    def js_dist(p_dist, q_dist):
        states = sorted(list(set(p_dist.keys()).union(set(q_dist.keys()))))
        p = np.array([p_dist.get(s, 1e-10) for s in states])
        q = np.array([q_dist.get(s, 1e-10) for s in states])
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        def kl(a, b):
            mask = (a > 0) & (b > 0)
            return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
        js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return np.sqrt(max(0, js))

    dist = js_dist(current_regime['probabilities'], old_regime['probabilities'])
    print(f"Manual JS Distance: {dist:.4f}")
    print(f"Manual Stability (1-dist): {1-dist:.4f}")

if __name__ == "__main__":
    debug_stability()
