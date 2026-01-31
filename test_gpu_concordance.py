#!/usr/bin/env python3
"""
GPU Concordance Matrix Test Script

Tests GPU-accelerated concordance classification and compares performance with CPU version.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from config import GPUConfig, SignalState
from inference.concordance_matrix import ConcordanceMatrix
from inference.gpu import GPUConcordanceMatrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def generate_test_signals(n_samples: int = 1000) -> pd.DataFrame:
    """Generate random test signals.

    Args:
        n_samples: Number of signal samples to generate

    Returns:
        DataFrame with HTF, MTF, LTF signal columns
    """
    np.random.seed(42)  # For reproducibility

    # Generate random signals (-2 to 2)
    signals = np.random.randint(-2, 3, size=(n_samples, 3))

    df = pd.DataFrame({
        'rsi_signal_d': signals[:, 0],     # HTF
        'rsi_signal_240': signals[:, 1],   # MTF
        'rsi_signal_60': signals[:, 2],    # LTF
    })

    logger.info(f"Generated {n_samples} test signals")
    return df


def test_gpu_availability():
    """Test if GPU is available."""
    try:
        import cupy as cp
        logger.info("✓ CuPy is installed")

        try:
            device = cp.cuda.Device(0)
            device.use()
            logger.info(f"✓ GPU available: {device.name.decode('utf-8')}")
            mem_info = device.mem_info
            logger.info(f"  GPU Memory: {mem_info[1] / (1024**3):.2f} GB total, "
                       f"{mem_info[0] / (1024**3):.2f} GB free")
            return True
        except Exception as e:
            logger.warning(f"✗ GPU not available: {e}")
            return False
    except ImportError:
        logger.warning("✗ CuPy not installed")
        return False


def test_concordance_cpu(signals_df: pd.DataFrame):
    """Test CPU-based concordance classification."""
    logger.info("\n" + "="*80)
    logger.info("Testing CPU Concordance Classification")
    logger.info("="*80)

    matrix = ConcordanceMatrix(
        structural_weight=0.6,
        tactical_weight=0.3,
        entry_weight=0.1
    )

    try:
        start_time = time.perf_counter()
        result_df = matrix.classify_signals_dataframe(signals_df)
        elapsed = time.perf_counter() - start_time

        logger.info(f"✓ CPU concordance classification completed")
        logger.info(f"  Time: {elapsed*1000:.2f} ms ({elapsed:.4f} seconds)")
        logger.info(f"  Processed: {len(result_df)} signals")

        # Get distribution
        distribution = matrix.get_concordance_distribution(result_df)
        logger.info(f"\n  Scenario Distribution:")
        for scenario, data in distribution['scenario_distribution'].items():
            if data['count'] > 0:
                logger.info(f"    {scenario}: {data['count']} ({data['percentage']:.1f}%)")

        if distribution['score_statistics']:
            stats = distribution['score_statistics']
            logger.info(f"\n  Score Statistics:")
            logger.info(f"    Mean: {stats['mean_score']:.4f}")
            logger.info(f"    Median: {stats['median_score']:.4f}")
            logger.info(f"    Std: {stats['std_score']:.4f}")
            logger.info(f"    Range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]")

        return result_df, elapsed, distribution

    except Exception as e:
        logger.error(f"✗ CPU test failed: {e}", exc_info=True)
        return None, None, None


def test_concordance_gpu(signals_df: pd.DataFrame):
    """Test GPU-based concordance classification."""
    logger.info("\n" + "="*80)
    logger.info("Testing GPU Concordance Classification")
    logger.info("="*80)

    # Configure GPU
    gpu_config = GPUConfig(
        use_gpu=True,
        device_id=0,
        min_size_for_gpu=100,
        auto_fallback_on_error=True,
        log_gpu_usage=True
    )

    logger.info(f"GPU Config: {gpu_config}")

    matrix = GPUConcordanceMatrix(
        structural_weight=0.6,
        tactical_weight=0.3,
        entry_weight=0.1,
        config=gpu_config
    )

    try:
        start_time = time.perf_counter()
        result_df = matrix.classify_signals_dataframe(signals_df, use_gpu=True)
        elapsed = time.perf_counter() - start_time

        logger.info(f"✓ GPU concordance classification completed")
        logger.info(f"  Time: {elapsed*1000:.2f} ms ({elapsed:.4f} seconds)")
        logger.info(f"  Processed: {len(result_df)} signals")

        # Get distribution
        distribution = matrix.get_concordance_distribution(result_df, use_gpu=True)
        logger.info(f"\n  Scenario Distribution:")
        for scenario, data in distribution['scenario_distribution'].items():
            if data['count'] > 0:
                logger.info(f"    {scenario}: {data['count']} ({data['percentage']:.1f}%)")

        if distribution['score_statistics']:
            stats = distribution['score_statistics']
            logger.info(f"\n  Score Statistics:")
            logger.info(f"    Mean: {stats['mean_score']:.4f}")
            logger.info(f"    Median: {stats['median_score']:.4f}")
            logger.info(f"    Std: {stats['std_score']:.4f}")
            logger.info(f"    Range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]")

        # Get performance stats
        perf_stats = matrix.get_performance_stats()
        logger.info(f"\n  Performance Stats:")
        logger.info(f"    Total operations: {perf_stats['total_operations']}")
        logger.info(f"    GPU preferred: {perf_stats['gpu_preferred']}")
        logger.info(f"    GPU failures: {perf_stats['gpu_failures']}")

        if 'memory' in perf_stats and perf_stats['memory'].get('available'):
            mem = perf_stats['memory']
            logger.info(f"    GPU memory: {mem['used_mb']:.1f} MB / {mem['total_mb']:.1f} MB "
                       f"({mem['usage_percent']:.1f}% used)")

        return result_df, elapsed, distribution

    except Exception as e:
        logger.error(f"✗ GPU test failed: {e}", exc_info=True)
        return None, None, None


def compare_results(cpu_df, gpu_df, cpu_dist, gpu_dist):
    """Compare CPU and GPU results."""
    logger.info("\n" + "="*80)
    logger.info("Comparing CPU vs GPU Results")
    logger.info("="*80)

    if cpu_df is None or gpu_df is None:
        logger.error("Cannot compare - one or both results are None")
        return

    # Compare samples
    logger.info(f"Samples: CPU={len(cpu_df)}, GPU={len(gpu_df)}, Match={len(cpu_df) == len(gpu_df)}")

    # Compare scenarios
    scenario_matches = (
        cpu_df['concordance_scenario'].values ==
        gpu_df['concordance_scenario'].values
    ).sum()
    scenario_match_rate = scenario_matches / len(cpu_df) * 100

    logger.info(f"Scenario matches: {scenario_matches}/{len(cpu_df)} ({scenario_match_rate:.1f}%)")

    # Compare scores
    if 'concordance_score' in cpu_df.columns and 'concordance_score' in gpu_df.columns:
        score_diff = np.abs(
            cpu_df['concordance_score'].values -
            gpu_df['concordance_score'].values
        )
        max_diff = np.max(score_diff)
        mean_diff = np.mean(score_diff)

        logger.info(f"Score differences: max={max_diff:.6e}, mean={mean_diff:.6e}")

        if max_diff < 1e-6:
            logger.info(f"✓ Scores match perfectly (max diff: {max_diff:.6e})")
        elif max_diff < 1e-3:
            logger.info(f"✓ Scores match within tolerance (max diff: {max_diff:.6e})")
        else:
            logger.warning(f"✗ Scores differ significantly (max diff: {max_diff:.6e})")

    # Compare distributions
    if cpu_dist and gpu_dist:
        cpu_stats = cpu_dist.get('score_statistics', {})
        gpu_stats = gpu_dist.get('score_statistics', {})

        if cpu_stats and gpu_stats:
            for key in ['mean_score', 'median_score', 'std_score']:
                if key in cpu_stats and key in gpu_stats:
                    diff = abs(cpu_stats[key] - gpu_stats[key])
                    logger.info(f"{key}: CPU={cpu_stats[key]:.6f}, "
                              f"GPU={gpu_stats[key]:.6f}, diff={diff:.6e}")


def benchmark_scaling(sizes=[100, 500, 1000, 5000, 10000]):
    """Benchmark CPU vs GPU at different dataset sizes."""
    logger.info("\n" + "="*80)
    logger.info("Benchmarking Scaling Performance")
    logger.info("="*80)

    results = []

    for size in sizes:
        logger.info(f"\nTesting with {size} samples...")

        # Generate test data
        signals_df = generate_test_signals(size)

        # CPU benchmark
        cpu_matrix = ConcordanceMatrix()
        cpu_times = []
        for _ in range(3):  # Average of 3 runs
            start = time.perf_counter()
            cpu_matrix.classify_signals_dataframe(signals_df)
            cpu_times.append(time.perf_counter() - start)
        cpu_time = np.mean(cpu_times)

        # GPU benchmark
        gpu_matrix = GPUConcordanceMatrix()
        gpu_times = []
        for _ in range(3):  # Average of 3 runs
            start = time.perf_counter()
            gpu_matrix.classify_signals_dataframe(signals_df, use_gpu=True)
            gpu_times.append(time.perf_counter() - start)
        gpu_time = np.mean(gpu_times)

        speedup = cpu_time / gpu_time

        results.append({
            'size': size,
            'cpu_time_ms': cpu_time * 1000,
            'gpu_time_ms': gpu_time * 1000,
            'speedup': speedup
        })

        logger.info(f"  CPU: {cpu_time*1000:.2f} ms")
        logger.info(f"  GPU: {gpu_time*1000:.2f} ms")
        logger.info(f"  Speedup: {speedup:.2f}x {'🚀' if speedup > 2 else ''}")

    # Summary table
    logger.info("\n" + "="*80)
    logger.info("Scaling Summary")
    logger.info("="*80)
    logger.info(f"{'Size':>8} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>10}")
    logger.info("-" * 80)
    for r in results:
        logger.info(f"{r['size']:>8} | {r['cpu_time_ms']:>10.2f} | "
                   f"{r['gpu_time_ms']:>10.2f} | {r['speedup']:>10.2f}x")


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("QBN GPU Concordance Matrix Test")
    logger.info("="*80)

    # Check GPU availability
    gpu_available = test_gpu_availability()

    # Generate test signals
    signals_df = generate_test_signals(n_samples=10000)

    # Test CPU version
    cpu_df, cpu_time, cpu_dist = test_concordance_cpu(signals_df)

    # Test GPU version
    if gpu_available:
        gpu_df, gpu_time, gpu_dist = test_concordance_gpu(signals_df)

        # Compare results
        if cpu_df is not None and gpu_df is not None:
            compare_results(cpu_df, gpu_df, cpu_dist, gpu_dist)

            # Calculate speedup
            if cpu_time and gpu_time:
                speedup = cpu_time / gpu_time
                logger.info("\n" + "="*80)
                logger.info(f"Performance Summary")
                logger.info("="*80)
                logger.info(f"Dataset size: {len(signals_df)} signals")
                logger.info(f"CPU Time: {cpu_time*1000:.2f} ms")
                logger.info(f"GPU Time: {gpu_time*1000:.2f} ms")
                logger.info(f"Speedup: {speedup:.2f}x {'🚀' if speedup > 2 else ''}")

        # Benchmark scaling
        try:
            benchmark_scaling(sizes=[100, 500, 1000, 5000, 10000])
        except Exception as e:
            logger.warning(f"Scaling benchmark failed: {e}")

    else:
        logger.warning("\nGPU not available - skipping GPU tests")
        logger.info("\nTo enable GPU:")
        logger.info("1. Install CuPy: pip install cupy-cuda12x")
        logger.info("2. Ensure NVIDIA GPU drivers are installed")
        logger.info("3. Ensure CUDA toolkit is installed")

    logger.info("\n" + "="*80)
    logger.info("Test completed")
    logger.info("="*80)


if __name__ == '__main__':
    main()
