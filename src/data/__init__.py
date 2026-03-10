"""
Data processing utilities for circuit GNN training.

This module provides utilities for:
- Graph building from circuit netlists
- Parameter sampling (LHS)
- Pre-batching for efficient training
- Visualization and plotting
- Feature normalization
"""

from src.data.encoding import Trie, build_circuit_trie
from src.data.graph_builder import CircuitGraphBuilder, TerminalNode, NetNode
from src.data.sampling import (
    generate_lhs_samples,
    generate_random_params,
    generate_netlist,
    worker_generate_sample,
)
from src.data.batching import create_prebatched_dataset, load_prebatched_variant
from src.data.plotting import (
    plot_parameter_distributions,
    plot_node_voltage_distributions,
    plot_device_current_distributions,
    plot_target_normalization_distributions,
)
from src.data.normalization import compute_normalization_stats, normalize_features

__all__ = [
    # Encoding
    'Trie',
    'build_circuit_trie',
    # Graph builder
    'CircuitGraphBuilder',
    'TerminalNode',
    'NetNode',
    # Sampling
    'generate_lhs_samples',
    'generate_random_params',
    'generate_netlist',
    'worker_generate_sample',
    # Batching
    'create_prebatched_dataset',
    'load_prebatched_variant',
    # Plotting
    'plot_parameter_distributions',
    'plot_node_voltage_distributions',
    'plot_device_current_distributions',
    'plot_target_normalization_distributions',
    # Normalization
    'compute_normalization_stats',
    'normalize_features',
]
