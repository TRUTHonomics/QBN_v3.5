#!/usr/bin/env python3
"""benchmark_barriers.py - Performance benchmarks"""

import logging
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

# Voeg project root toe aan path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.barrier_outcome_generator import BarrierOutcomeGenerator, BarrierConfig
from inference.gpu_barrier_calculator import GPUBarrierCalculator
from database.db import get_cursor
from core.logging_utils import setup_logging

logger = setup_logging("benchmark_barriers")

def benchmark_cpu_single(n_iterations: int = 50):
    """Benchmark single timestamp CPU."""
    logger.info(f"🚀 Running CPU single benchmark ({n_iterations} iterations)...")
    config = BarrierConfig()
    gen = BarrierOutcomeGenerator(config)
    
    # Mock data
    window_min = 2880
    mock_klines = pd.DataFrame({
        'high': np.random.rand(window_min) + 100,
        'low': np.random.rand(window_min) + 99,
        'close': np.random.rand(window_min) + 99.5
    })
    
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = gen._calculate_barriers(
            klines=mock_klines,
            ref_price=100.0,
            atr=2.0,
            barriers=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50],
            direction='up'
        )
        times.append((time.perf_counter() - start) * 1000)
    
    avg = np.mean(times)
    logger.info(f"✅ CPU Single: {avg:.2f} ms")
    return avg

def benchmark_gpu_batch(n_rows: int = 10000):
    """Benchmark batch processing GPU."""
    logger.info(f"🚀 Running GPU batch benchmark ({n_rows} rows)...")
    
    barriers = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
    window_size = 1440
    calc = GPUBarrierCalculator(barriers=barriers, max_obs_min=window_size)
    
    # Mock data: (N, T, 2) array - [high, low]
    # N = n_rows, T = window_size
    prices_batch = np.random.rand(n_rows, window_size, 2).astype(np.float32) + 100
    ref_prices = np.random.rand(n_rows).astype(np.float32) + 100
    atrs = np.random.rand(n_rows).astype(np.float32) * 2.0 + 0.1
    
    # Warmup
    _ = calc.calculate_batch(prices_batch[:10], ref_prices[:10], atrs[:10])
    
    start = time.perf_counter()
    _ = calc.calculate_batch(
        prices_batch=prices_batch,
        ref_prices=ref_prices,
        atrs=atrs
    )
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"✅ GPU Batch: {elapsed:.2f} ms ({elapsed/n_rows:.4f} ms per row)")
    return elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, help='Custom output directory')
    args = parser.parse_args()
    
    results = []
    results.append(f"# Barrier Performance Benchmark Report")
    results.append(f"**Gegenereerd:** {datetime.now().isoformat()}\n")
    
    cpu_avg = benchmark_cpu_single()
    gpu_total = benchmark_gpu_batch()
    
    results.append("## Results")
    results.append(f"- **CPU Single (avg):** {cpu_avg:.2f} ms")
    results.append(f"- **GPU Batch (10k rows):** {gpu_total:.2f} ms")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write("\n".join(results))
        logger.info(f"💾 Report saved to {filename}")

if __name__ == "__main__":
    main()
