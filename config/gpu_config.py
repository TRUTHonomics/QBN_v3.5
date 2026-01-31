"""
GPU Configuration Module for QBN

Provides GPU-specific configuration options for accelerated inference operations.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUConfig:
    """GPU acceleration configuration.

    Attributes:
        use_gpu: Enable GPU acceleration if available (auto-disable if no GPU found)
        device_id: GPU device index to use (default: 0)
        batch_size_cpt: Number of rows to process per GPU batch for CPT generation
        batch_size_inference: Number of assets to process per inference batch
        min_size_for_gpu: Minimum dataset size to use GPU (smaller datasets use CPU)
        max_gpu_memory_mb: Maximum GPU memory per operation in megabytes
        use_pinned_memory: Use pinned memory for faster CPU-GPU transfers
        clear_cache_after_batch: Free GPU memory aggressively after each batch
        auto_fallback_on_error: Automatically switch to CPU if GPU execution fails
        fallback_threshold_ms: Switch to CPU if GPU consistently slower than this
        use_float64: Use double precision for probability calculations
        enable_profiling: Enable CUDA profiling for performance analysis
        log_gpu_usage: Log GPU memory usage information
    """

    # GPU selection
    use_gpu: bool = True
    device_id: int = 0

    # Performance tuning
    batch_size_cpt: int = 100000  # Rows per GPU batch for CPT generation
    batch_size_inference: int = 1000  # Assets per inference batch
    min_size_for_gpu: int = 10000  # Minimum dataset size to use GPU

    # Memory management
    max_gpu_memory_mb: int = 512  # Maximum GPU memory per operation
    use_pinned_memory: bool = True  # Faster CPU-GPU transfers
    clear_cache_after_batch: bool = True  # Free GPU memory aggressively

    # Fallback behavior
    auto_fallback_on_error: bool = True  # Auto switch to CPU on GPU error
    fallback_threshold_ms: float = 100.0  # Switch to CPU if GPU slower

    # Precision
    use_float64: bool = True  # Use double precision for probabilities

    # Monitoring
    enable_profiling: bool = False  # Enable CUDA profiling
    log_gpu_usage: bool = True  # Log GPU memory usage

    @classmethod
    def from_env(cls) -> 'GPUConfig':
        """Load GPU configuration from environment variables.

        Environment variables:
            QBN_USE_GPU: Enable/disable GPU (true/false)
            QBN_GPU_DEVICE: GPU device index (integer)
            QBN_GPU_BATCH_SIZE: CPT generation batch size (integer)
            QBN_GPU_MAX_MEMORY_MB: Maximum GPU memory in MB (integer)
            QBN_GPU_AUTO_FALLBACK: Enable CPU fallback (true/false)
            QBN_GPU_MIN_SIZE: Minimum size for GPU execution (integer)

        Returns:
            GPUConfig instance with values from environment
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, str(default)).lower()
            return val in ('true', '1', 'yes', 'on')

        def get_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except ValueError:
                return default

        return cls(
            use_gpu=get_bool('QBN_USE_GPU', True),
            device_id=get_int('QBN_GPU_DEVICE', 0),
            batch_size_cpt=get_int('QBN_GPU_BATCH_SIZE', 100000),
            batch_size_inference=get_int('QBN_GPU_BATCH_SIZE_INFERENCE', 1000),
            min_size_for_gpu=get_int('QBN_GPU_MIN_SIZE', 10000),
            max_gpu_memory_mb=get_int('QBN_GPU_MAX_MEMORY_MB', 512),
            use_pinned_memory=get_bool('QBN_GPU_USE_PINNED_MEMORY', True),
            clear_cache_after_batch=get_bool('QBN_GPU_CLEAR_CACHE', True),
            auto_fallback_on_error=get_bool('QBN_GPU_AUTO_FALLBACK', True),
            fallback_threshold_ms=get_float('QBN_GPU_FALLBACK_THRESHOLD_MS', 100.0),
            use_float64=get_bool('QBN_GPU_USE_FLOAT64', True),
            enable_profiling=get_bool('QBN_GPU_ENABLE_PROFILING', False),
            log_gpu_usage=get_bool('QBN_GPU_LOG_USAGE', True)
        )

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.device_id < 0:
            raise ValueError(f"device_id must be >= 0, got {self.device_id}")

        if self.batch_size_cpt < 1:
            raise ValueError(f"batch_size_cpt must be >= 1, got {self.batch_size_cpt}")

        if self.batch_size_inference < 1:
            raise ValueError(f"batch_size_inference must be >= 1, got {self.batch_size_inference}")

        if self.min_size_for_gpu < 0:
            raise ValueError(f"min_size_for_gpu must be >= 0, got {self.min_size_for_gpu}")

        if self.max_gpu_memory_mb < 1:
            raise ValueError(f"max_gpu_memory_mb must be >= 1, got {self.max_gpu_memory_mb}")

        if self.fallback_threshold_ms < 0:
            raise ValueError(f"fallback_threshold_ms must be >= 0, got {self.fallback_threshold_ms}")

    def __str__(self) -> str:
        """String representation of GPU configuration."""
        return (
            f"GPUConfig(use_gpu={self.use_gpu}, device_id={self.device_id}, "
            f"batch_size_cpt={self.batch_size_cpt}, min_size={self.min_size_for_gpu}, "
            f"max_memory_mb={self.max_gpu_memory_mb}, auto_fallback={self.auto_fallback_on_error})"
        )


# Default GPU configuration instance
DEFAULT_GPU_CONFIG = GPUConfig()
