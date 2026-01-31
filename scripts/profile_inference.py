#!/usr/bin/env python3
"""
Profile Entry Model Inference performance voor QBN v2.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.trade_aligned_inference import SignalEvidence
from inference.inference_loader import InferenceLoader

from core.logging_utils import setup_logging

logger = setup_logging("profile_inference")

def profile_inference(asset_id: int, iterations: int = 1000):
    """Meet de latency van de inference engine over vele iteraties."""
    print(f"🚀 Start profiling voor Asset {asset_id} ({iterations} iteraties)...")
    
    loader = InferenceLoader()
    try:
        engine = loader.load_inference_engine(asset_id)
    except Exception as e:
        print(f"❌ Kon engine niet laden: {e}")
        return
    
    # 1. Maak realistische test-evidence
    evidence = SignalEvidence(
        asset_id=asset_id,
        timestamp=time.time(),
        leading_signals={'rsi_oversold_1': 1, 'stoch_oversold_1': 1},
        coincident_signals={'macd_bullish_cross_1': 1},
        confirming_signals={'adx_signal_d': 1, 'adx_signal_240': 1}
    )
    
    # 2. Warm-up (JIT/Cache)
    for _ in range(50):
        engine.infer(evidence)
    
    # 3. Profiling
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        engine.infer(evidence)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    times = np.array(times)
    
    # 4. Rapportage
    print("\n" + "="*40)
    print("📊 INFERENCE LATENCY PROFIEL")
    print("="*40)
    print(f"  Gemiddelde: {np.mean(times):.2f} ms")
    print(f"  Mediaan:    {np.median(times):.2f} ms")
    print(f"  P95:        {np.percentile(times, 95):.2f} ms")
    print(f"  P99:        {np.percentile(times, 99):.2f} ms")
    print(f"  Maximum:    {np.max(times):.2f} ms")
    print("-"*40)
    print(f"  Target:     25.00 ms")
    
    p99 = np.percentile(times, 99)
    if p99 <= 25.0:
        print(f"  ✅ STATUS: Binnen target (P99: {p99:.2f}ms)")
    else:
        print(f"  ⚠️  STATUS: Target overschreden (P99: {p99:.2f}ms)")
    print("="*40 + "\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Profile QBN v2 Inference performance.")
    parser.add_argument("--asset-id", type=int, default=1, help="Asset ID to profile (default: 1)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations (default: 1000)")
    
    args = parser.parse_args()
    profile_inference(asset_id=args.asset_id, iterations=args.iterations)

