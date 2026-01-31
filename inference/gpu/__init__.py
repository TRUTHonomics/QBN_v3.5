"""
GPU Acceleration Module for QBN Inference Engine

This module provides GPU-accelerated implementations of compute-intensive
operations using CuPy for drop-in NumPy replacement.

Key Components:
- accelerator: Base GPU infrastructure (data management, memory management, fallback)
- gpu_cpt_generator: GPU-accelerated CPT generation from historical data
- gpu_concordance: GPU-accelerated concordance matrix calculations

Usage:
    from inference.gpu import GPUCPTGenerator, GPUConfig

    config = GPUConfig(use_gpu=True, batch_size_cpt=100000)
    generator = GPUCPTGenerator(config)
    cpt = generator.generate_cpt_for_asset(asset_id=1, ...)
"""

from .accelerator import (
    GPUDataManager,
    GPUMemoryManager,
    AdaptiveGPUAccelerator
)

from .gpu_cpt_generator import GPUCPTGenerator
from .gpu_concordance import GPUConcordanceMatrix, create_gpu_concordance_matrix
from .gpu_bayesian_network import GPUBayesianNetworkHelper, get_gpu_bayesian_helper

__all__ = [
    'GPUDataManager',
    'GPUMemoryManager',
    'AdaptiveGPUAccelerator',
    'GPUCPTGenerator',
    'GPUConcordanceMatrix',
    'create_gpu_concordance_matrix',
    'GPUBayesianNetworkHelper',
    'get_gpu_bayesian_helper',
]

__version__ = '0.1.0'
