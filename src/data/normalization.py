"""
Feature normalization utilities for circuit graphs.

This module provides functions for computing normalization statistics
and applying z-score normalization to node features.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np


def compute_normalization_stats(samples: List[Dict], num_global_features: int = 5) -> Dict:
    """
    Compute per-type mean/std for node features only (NOT vdc or global features!).

    Args:
        samples: List of samples with graphs
        num_global_features: Number of global feature columns at the end of x (default: 5)
                             These are already domain-normalized and should not be z-score normalized

    Returns:
        Dict mapping node types to feature statistics
    """
    type_values = defaultdict(lambda: defaultdict(list))

    for sample in samples:
        graph = sample['graph']
        for i, ntype in enumerate(graph.node_types):
            x = graph.x[i]
            if x.numel() > 0:
                num_base_features = x.numel() - num_global_features
                for j in range(num_base_features):
                    type_values[ntype][j].append(x[j].item())

    stats = {}
    for ntype, prop_dict in type_values.items():
        stats[ntype] = {}
        for prop_idx, values in prop_dict.items():
            values = np.array(values)
            mean = float(values.mean())
            std = float(values.std())
            if std == 0:
                std = 1.0
            stats[ntype][prop_idx] = {'mean': mean, 'std': std}
            print(f"  {ntype} prop {prop_idx}: mean={mean:.6f}, std={std:.6f}")

    print(f"  (Skipped last {num_global_features} global features - already domain-normalized)")

    return stats


def normalize_features(samples: List[Dict], stats: Dict, num_global_features: int = 5) -> List[Dict]:
    """
    Normalize base features only (NOT vdc targets or global features to avoid data leakage).

    IMPORTANT:
    - vdc and node_voltage_targets are kept as RAW voltages!
    - Global features (last num_global_features columns) are kept as-is (already domain-normalized)
    - Only base features (W, L for terminals) are z-score normalized
    - The training script will compute normalization stats for vdc from train split only

    Args:
        samples: List of samples with graphs
        stats: Dict of normalization stats from compute_normalization_stats()
        num_global_features: Number of global feature columns to skip (default: 5)

    Returns:
        Samples with normalized features
    """
    for sample in samples:
        graph = sample['graph']
        new_x = graph.x.clone()

        for i, ntype in enumerate(graph.node_types):
            if ntype in stats:
                num_features = graph.x[i].numel()
                num_base_features = num_features - num_global_features

                for j in range(num_base_features):
                    if j in stats[ntype]:
                        mean = stats[ntype][j]['mean']
                        std = stats[ntype][j]['std']
                        new_x[i, j] = (graph.x[i, j] - mean) / std

        graph.x = new_x

    return samples
