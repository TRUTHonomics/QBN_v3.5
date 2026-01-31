#!/usr/bin/env python3
"""
GPU CPT Generator Test Script

Tests GPU-accelerated CPT generation and compares performance with CPU version.
"""

import time
import logging
from typing import Dict, Any

from config import GPUConfig, SignalState
from inference.cpt_generator import ConditionalProbabilityTableGenerator
from inference.gpu import GPUCPTGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


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


def test_cpt_generation_cpu():
    """Test CPU-based CPT generation."""
    logger.info("\n" + "="*80)
    logger.info("Testing CPU CPT Generation")
    logger.info("="*80)

    generator = ConditionalProbabilityTableGenerator(laplace_alpha=1.0)

    # Test parameters
    asset_id = 1  # BTC/USDT
    node_name = "structural_trend"
    parent_nodes = []
    lookback_days = 30
    db_columns = ['macd_signal_d', 'bb_signal_d', 'rsi_signal_d']

    try:
        start_time = time.perf_counter()
        cpt = generator.generate_cpt_for_asset(
            asset_id=asset_id,
            node_name=node_name,
            parent_nodes=parent_nodes,
            lookback_days=lookback_days,
            db_columns=db_columns,
            aggregation_method='majority'
        )
        elapsed = time.perf_counter() - start_time

        logger.info(f"✓ CPU CPT generation completed")
        logger.info(f"  Time: {elapsed*1000:.2f} ms ({elapsed:.4f} seconds)")
        logger.info(f"  Observations: {cpt.get('total_observations', 0)}")
        logger.info(f"  Type: {cpt.get('type')}")

        if 'probabilities' in cpt:
            logger.info("  Probabilities:")
            for state, prob in cpt['probabilities'].items():
                logger.info(f"    {state}: {prob:.4f}")

        return cpt, elapsed

    except Exception as e:
        logger.error(f"✗ CPU test failed: {e}", exc_info=True)
        return None, None


def test_cpt_generation_gpu():
    """Test GPU-based CPT generation."""
    logger.info("\n" + "="*80)
    logger.info("Testing GPU CPT Generation")
    logger.info("="*80)

    # Configure GPU
    gpu_config = GPUConfig(
        use_gpu=True,
        device_id=0,
        batch_size_cpt=100000,
        min_size_for_gpu=1000,
        auto_fallback_on_error=True,
        log_gpu_usage=True
    )

    logger.info(f"GPU Config: {gpu_config}")

    generator = GPUCPTGenerator(laplace_alpha=1.0, config=gpu_config)

    # Test parameters (same as CPU test)
    asset_id = 1  # BTC/USDT
    node_name = "structural_trend"
    parent_nodes = []
    lookback_days = 30
    db_columns = ['macd_signal_d', 'bb_signal_d', 'rsi_signal_d']

    try:
        start_time = time.perf_counter()
        cpt = generator.generate_cpt_for_asset(
            asset_id=asset_id,
            node_name=node_name,
            parent_nodes=parent_nodes,
            lookback_days=lookback_days,
            db_columns=db_columns,
            aggregation_method='majority'
        )
        elapsed = time.perf_counter() - start_time

        logger.info(f"✓ GPU CPT generation completed")
        logger.info(f"  Time: {elapsed*1000:.2f} ms ({elapsed:.4f} seconds)")
        logger.info(f"  Observations: {cpt.get('total_observations', 0)}")
        logger.info(f"  Type: {cpt.get('type')}")

        if 'probabilities' in cpt:
            logger.info("  Probabilities:")
            for state, prob in cpt['probabilities'].items():
                logger.info(f"    {state}: {prob:.4f}")

        # Get performance stats
        stats = generator.get_performance_stats()
        logger.info(f"\n  Performance Stats:")
        logger.info(f"    Total operations: {stats['total_operations']}")
        logger.info(f"    GPU preferred: {stats['gpu_preferred']}")
        logger.info(f"    GPU failures: {stats['gpu_failures']}")

        if 'cache' in stats:
            cache = stats['cache']
            logger.info(f"    Cache hits: {cache['hits']}, misses: {cache['misses']}, "
                       f"hit rate: {cache['hit_rate']:.2%}")

        if 'memory' in stats and stats['memory'].get('available'):
            mem = stats['memory']
            logger.info(f"    GPU memory: {mem['used_mb']:.1f} MB / {mem['total_mb']:.1f} MB "
                       f"({mem['usage_percent']:.1f}% used)")

        return cpt, elapsed

    except Exception as e:
        logger.error(f"✗ GPU test failed: {e}", exc_info=True)
        return None, None


def compare_results(cpu_cpt, gpu_cpt):
    """Compare CPU and GPU results."""
    logger.info("\n" + "="*80)
    logger.info("Comparing CPU vs GPU Results")
    logger.info("="*80)

    if cpu_cpt is None or gpu_cpt is None:
        logger.error("Cannot compare - one or both results are None")
        return

    # Compare observations
    cpu_obs = cpu_cpt.get('total_observations', 0)
    gpu_obs = gpu_cpt.get('total_observations', 0)
    logger.info(f"Observations: CPU={cpu_obs}, GPU={gpu_obs}, Match={cpu_obs == gpu_obs}")

    # Compare probabilities
    if 'probabilities' in cpu_cpt and 'probabilities' in gpu_cpt:
        cpu_probs = cpu_cpt['probabilities']
        gpu_probs = gpu_cpt['probabilities']

        max_diff = 0.0
        for state in cpu_probs:
            if state in gpu_probs:
                diff = abs(cpu_probs[state] - gpu_probs[state])
                max_diff = max(max_diff, diff)
                if diff > 1e-6:
                    logger.warning(f"  {state}: CPU={cpu_probs[state]:.6f}, "
                                 f"GPU={gpu_probs[state]:.6f}, diff={diff:.6e}")

        if max_diff < 1e-6:
            logger.info(f"✓ Probabilities match (max diff: {max_diff:.6e})")
        else:
            logger.warning(f"✗ Probabilities differ (max diff: {max_diff:.6e})")


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("QBN GPU CPT Generator Test")
    logger.info("="*80)

    # Check GPU availability
    gpu_available = test_gpu_availability()

    # Test CPU version
    cpu_cpt, cpu_time = test_cpt_generation_cpu()

    # Test GPU version
    if gpu_available:
        gpu_cpt, gpu_time = test_cpt_generation_gpu()

        # Compare results
        if cpu_cpt and gpu_cpt:
            compare_results(cpu_cpt, gpu_cpt)

            # Calculate speedup
            if cpu_time and gpu_time:
                speedup = cpu_time / gpu_time
                logger.info("\n" + "="*80)
                logger.info(f"Performance Summary")
                logger.info("="*80)
                logger.info(f"CPU Time: {cpu_time*1000:.2f} ms")
                logger.info(f"GPU Time: {gpu_time*1000:.2f} ms")
                logger.info(f"Speedup: {speedup:.2f}x {'🚀' if speedup > 2 else ''}")
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
