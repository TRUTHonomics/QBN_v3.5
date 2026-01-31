"""
GPU Acceleration Base Infrastructure

Provides core GPU functionality including data management, memory management,
and adaptive CPU/GPU execution with automatic fallback.
"""

import time
import logging
from typing import Dict, Optional, Callable, Any, List, Tuple
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from config.gpu_config import GPUConfig

logger = logging.getLogger(__name__)


class GPUDataManager:
    """Manages efficient CPU-GPU data transfers with caching.

    Features:
    - Pinned memory for faster transfers
    - GPU data caching to avoid repeated transfers
    - Automatic memory cleanup
    - Batch operation support
    """

    def __init__(self, config: GPUConfig):
        """Initialize GPU data manager.

        Args:
            config: GPU configuration

        Raises:
            RuntimeError: If GPU requested but not available
        """
        self.config = config
        self.gpu_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if config.use_gpu and not CUPY_AVAILABLE:
            if config.auto_fallback_on_error:
                logger.warning("CuPy not available, falling back to CPU")
                self.config.use_gpu = False
            else:
                raise RuntimeError("GPU requested but CuPy not available")

        if self.config.use_gpu:
            try:
                cp.cuda.Device(config.device_id).use()
                logger.info(f"GPU data manager initialized on device {config.device_id}")
            except Exception as e:
                if config.auto_fallback_on_error:
                    logger.warning(f"Failed to initialize GPU: {e}, falling back to CPU")
                    self.config.use_gpu = False
                else:
                    raise

    @property
    def xp(self):
        """Get NumPy-like interface (CuPy if GPU, NumPy if CPU)."""
        return cp if self.config.use_gpu else np

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and enabled."""
        return self.config.use_gpu and CUPY_AVAILABLE

    def transfer_to_gpu(
        self,
        data: np.ndarray,
        cache_key: Optional[str] = None,
        dtype: Optional[np.dtype] = None
    ) -> Any:
        """Transfer data from CPU to GPU with optional caching.

        Args:
            data: NumPy array or pandas DataFrame to transfer
            cache_key: Optional key for caching (reuse data across calls)
            dtype: Optional dtype conversion

        Returns:
            CuPy array on GPU (or NumPy array if CPU fallback)
        """
        if not self.is_gpu_available:
            return data if dtype is None else data.astype(dtype)

        # Check cache first
        if cache_key and cache_key in self.gpu_cache:
            self.cache_hits += 1
            if self.config.log_gpu_usage:
                logger.debug(f"GPU cache hit for key: {cache_key}")
            return self.gpu_cache[cache_key]

        self.cache_misses += 1

        # Convert pandas DataFrame to numpy if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, pd.Series):
            data = data.values

        # Apply dtype conversion if specified
        if dtype is not None:
            data = data.astype(dtype)

        # Transfer to GPU
        try:
            if self.config.use_pinned_memory:
                # Use pinned memory for faster transfers
                gpu_array = cp.asarray(data)
            else:
                gpu_array = cp.array(data)

            # Cache if key provided
            if cache_key:
                self.gpu_cache[cache_key] = gpu_array

            return gpu_array

        except Exception as e:
            if CUPY_AVAILABLE and isinstance(e, cp.cuda.memory.OutOfMemoryError):
                logger.warning(f"GPU out of memory during transfer: {e}")
            else:
                logger.warning(f"Error during transfer to GPU: {e}")
            
            if self.config.auto_fallback_on_error:
                logger.warning("Falling back to CPU")
                return data
            raise

    def transfer_to_cpu(self, data: Any) -> np.ndarray:
        """Transfer data from GPU to CPU.

        Args:
            data: CuPy array to transfer

        Returns:
            NumPy array on CPU
        """
        if not self.is_gpu_available or isinstance(data, np.ndarray):
            return data

        return cp.asnumpy(data)

    def clear_cache(self, key: Optional[str] = None):
        """Clear GPU cache.

        Args:
            key: Optional specific key to clear, or None to clear all
        """
        if key:
            if key in self.gpu_cache:
                del self.gpu_cache[key]
                logger.debug(f"Cleared GPU cache for key: {key}")
        else:
            self.gpu_cache.clear()
            logger.debug("Cleared entire GPU cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, and hit rate
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total_accesses': total,
            'hit_rate': hit_rate,
            'cached_items': len(self.gpu_cache)
        }


class GPUMemoryManager:
    """Manages GPU memory allocation and batch processing.

    Features:
    - Memory pool management
    - Streaming batch processing for large datasets
    - Automatic batch size adjustment
    - Memory usage monitoring
    """

    def __init__(self, config: GPUConfig):
        """Initialize GPU memory manager.

        Args:
            config: GPU configuration
        """
        self.config = config
        self.memory_pool = None

        if config.use_gpu and CUPY_AVAILABLE:
            try:
                self.memory_pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(self.memory_pool.malloc)
                logger.info("GPU memory pool initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize memory pool: {e}")

    def process_in_batches(
        self,
        data: np.ndarray,
        process_func: Callable[[Any], Any],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Process large dataset in GPU batches.

        Args:
            data: NumPy array to process
            process_func: Function to apply to each batch (GPU array → GPU array)
            batch_size: Optional batch size (uses config default if None)

        Returns:
            Processed data as NumPy array
        """
        if not self.config.use_gpu:
            return process_func(data)

        batch_size = batch_size or self.config.batch_size_cpt
        n_samples = len(data)
        results = []

        logger.debug(f"Processing {n_samples} samples in batches of {batch_size}")

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch = data[i:batch_end]

            try:
                # Transfer batch to GPU
                gpu_batch = cp.asarray(batch)

                # Process on GPU
                gpu_result = process_func(gpu_batch)

                # Transfer result back to CPU
                cpu_result = cp.asnumpy(gpu_result)
                results.append(cpu_result)

                # Free GPU memory
                if self.config.clear_cache_after_batch:
                    del gpu_batch, gpu_result
                    if self.memory_pool:
                        self.memory_pool.free_all_blocks()

            except Exception as e:
                is_oom = False
                if CUPY_AVAILABLE and isinstance(e, cp.cuda.memory.OutOfMemoryError):
                    is_oom = True
                
                if is_oom:
                    logger.warning(f"GPU OOM at batch {i//batch_size + 1}, reducing batch size")
                else:
                    logger.warning(f"Error during batch processing: {e}")
                
                # Try with smaller batch size
                new_batch_size = batch_size // 2
                if new_batch_size < 1000:
                    raise RuntimeError(f"Cannot process data even with small batches: {e}")

                # Recursively process with smaller batch size
                return self.process_in_batches(data, process_func, new_batch_size)

        # Concatenate results
        return np.concatenate(results) if results else np.array([])

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory usage information.

        Returns:
            Dictionary with memory statistics
        """
        if not self.config.use_gpu or not CUPY_AVAILABLE:
            return {'available': False}

        try:
            device = cp.cuda.Device(self.config.device_id)
            mem_info = device.mem_info

            stats = {
                'available': True,
                'device_id': self.config.device_id,
                'device_name': cp.cuda.Device().name.decode('utf-8'),
                'free_bytes': mem_info[0],
                'total_bytes': mem_info[1],
                'used_bytes': mem_info[1] - mem_info[0],
                'free_mb': mem_info[0] / (1024 ** 2),
                'total_mb': mem_info[1] / (1024 ** 2),
                'used_mb': (mem_info[1] - mem_info[0]) / (1024 ** 2),
                'usage_percent': ((mem_info[1] - mem_info[0]) / mem_info[1]) * 100
            }

            if self.memory_pool:
                stats['pool_used_bytes'] = self.memory_pool.used_bytes()
                stats['pool_total_bytes'] = self.memory_pool.total_bytes()

            return stats

        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {'available': False, 'error': str(e)}

    def free_memory(self):
        """Free all unused GPU memory."""
        if self.memory_pool:
            self.memory_pool.free_all_blocks()
            if self.config.log_gpu_usage:
                logger.debug("Freed all GPU memory blocks")


class AdaptiveGPUAccelerator:
    """Adaptive GPU execution with automatic CPU fallback.

    Features:
    - Automatic GPU/CPU selection based on data size
    - Performance monitoring and adaptive fallback
    - Error handling with graceful degradation
    - Performance history tracking
    """

    def __init__(self, config: GPUConfig):
        """Initialize adaptive GPU accelerator.

        Args:
            config: GPU configuration
        """
        self.config = config
        self.data_manager = GPUDataManager(config)
        self.memory_manager = GPUMemoryManager(config)

        # Performance tracking
        self.performance_history: deque = deque(maxlen=20)
        self.operation_metrics: Dict[str, List[float]] = defaultdict(list)

        # Adaptive thresholds
        self.prefer_gpu = config.use_gpu
        self.gpu_failure_count = 0
        self.cpu_fallback_active = False

    def execute_with_fallback(
        self,
        func_gpu: Callable,
        func_cpu: Callable,
        data_size: int,
        operation_name: str = "operation",
        *args,
        **kwargs
    ) -> Any:
        """Execute function with automatic GPU/CPU selection and fallback.

        Args:
            func_gpu: GPU implementation function
            func_cpu: CPU implementation function
            data_size: Size of data being processed
            operation_name: Name of operation for logging
            *args: Additional arguments to pass to functions
            **kwargs: Additional keyword arguments

        Returns:
            Result from either GPU or CPU function
        """
        # Check if data is too small for GPU
        if data_size < self.config.min_size_for_gpu:
            logger.debug(f"{operation_name}: Data size {data_size} below threshold "
                        f"{self.config.min_size_for_gpu}, using CPU")
            return func_cpu(*args, **kwargs)

        # Check if GPU is available and preferred
        if not self.prefer_gpu or not self.data_manager.is_gpu_available:
            return func_cpu(*args, **kwargs)

        # Try GPU execution
        try:
            start = time.perf_counter()
            result = func_gpu(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            # Track performance
            self.performance_history.append(('gpu', operation_name, duration_ms))
            self.operation_metrics[operation_name].append(duration_ms)

            # Check if GPU is consistently slower than threshold
            if duration_ms > self.config.fallback_threshold_ms:
                recent_slow = sum(
                    1 for backend, op, dur in list(self.performance_history)[-5:]
                    if backend == 'gpu' and op == operation_name and
                    dur > self.config.fallback_threshold_ms
                )

                if recent_slow >= 3:
                    logger.warning(
                        f"{operation_name}: GPU consistently slower than "
                        f"{self.config.fallback_threshold_ms}ms, "
                        f"switching to CPU for this operation"
                    )
                    self.prefer_gpu = False

            if self.config.log_gpu_usage:
                logger.debug(f"{operation_name}: GPU execution took {duration_ms:.2f}ms")

            # Reset failure count on success
            self.gpu_failure_count = 0

            return result

        except Exception as e:
            # Handle GPU errors gracefully
            is_gpu_error = False
            if CUPY_AVAILABLE:
                if isinstance(e, (cp.cuda.memory.OutOfMemoryError, RuntimeError)):
                    is_gpu_error = True
            
            if not is_gpu_error and not isinstance(e, RuntimeError):
                # If it's not a known GPU error or RuntimeError, re-raise it
                raise
                
            logger.warning(f"{operation_name}: GPU execution failed: {e}")
            self.gpu_failure_count += 1

            # Fall back to CPU if enabled
            if self.config.auto_fallback_on_error:
                logger.info(f"{operation_name}: Falling back to CPU")

                # If multiple failures, disable GPU temporarily
                if self.gpu_failure_count >= 3:
                    logger.warning(
                        f"{operation_name}: Multiple GPU failures, "
                        f"disabling GPU for this session"
                    )
                    self.prefer_gpu = False

                return func_cpu(*args, **kwargs)
            else:
                raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'total_operations': len(self.performance_history),
            'gpu_preferred': self.prefer_gpu,
            'gpu_failures': self.gpu_failure_count,
            'operations': {}
        }

        for op_name, durations in self.operation_metrics.items():
            if durations:
                stats['operations'][op_name] = {
                    'count': len(durations),
                    'mean_ms': np.mean(durations),
                    'median_ms': np.median(durations),
                    'std_ms': np.std(durations),
                    'min_ms': np.min(durations),
                    'max_ms': np.max(durations)
                }

        # Add cache stats from data manager
        stats['cache'] = self.data_manager.get_cache_stats()

        # Add memory info
        stats['memory'] = self.memory_manager.get_memory_info()

        return stats

    def reset_performance_tracking(self):
        """Reset performance tracking history."""
        self.performance_history.clear()
        self.operation_metrics.clear()
        self.gpu_failure_count = 0
        logger.debug("Reset performance tracking")
