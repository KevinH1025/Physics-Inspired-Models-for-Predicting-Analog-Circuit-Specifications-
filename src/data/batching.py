"""
Pre-batching utilities for efficient training.

This module provides functions to create pre-batched datasets with multiple
shuffling variants for efficient GPU training without runtime batching overhead.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np


def _add_device_ptr_tensors(batched_graph, batch_graphs):
    """Add ptr tensors for variable-size device attributes (MOSFET, resistor, etc.).

    These enable mixed-topology batches where graphs have different numbers of
    MOSFETs, resistors, isources, or capacitors. Each ptr tensor is shape
    [num_graphs + 1] with cumulative sizes, matching the PyG convention.
    """
    for attr, ptr_name in [('mosfet_info', 'mosfet_ptr'),
                           ('resistor_info', 'resistor_ptr'),
                           ('isource_info', 'isource_ptr'),
                           ('capacitor_info', 'capacitor_ptr')]:
        sizes = []
        for g in batch_graphs:
            t = getattr(g, attr, None)
            sizes.append(t.shape[0] if t is not None else 0)
        ptr = torch.zeros(len(sizes) + 1, dtype=torch.long)
        torch.cumsum(torch.tensor(sizes, dtype=torch.long), dim=0, out=ptr[1:])
        setattr(batched_graph, ptr_name, ptr)


def create_prebatched_dataset(
    samples: List[Dict],
    output_dir: Path,
    batch_size: int = 1024,
    num_variants: int = 10,
    seed: int = 42,
    is_validation: bool = False
) -> None:
    """
    Create pre-batched dataset with multiple batching variants for better shuffling.

    Strategy:
    - For training: Create N variants, each with a different shuffle of samples into batches
    - For validation/test: Create only 1 variant (no need for shuffling)

    Args:
        samples: List of sample dicts with 'graph' key
        output_dir: Output directory path
        batch_size: Samples per batch (default 1024)
        num_variants: Number of different batching variants (default 10)
        seed: Base random seed (each variant uses seed + variant_id)
        is_validation: If True, only create 1 variant
    """
    from torch_geometric.data import Batch

    actual_num_variants = 1 if is_validation else num_variants

    print(f"    Batch size: {batch_size}")
    print(f"    Num variants: {actual_num_variants}")
    print(f"    Base seed: {seed}")

    all_graphs = [s['graph'] for s in samples]
    num_samples = len(all_graphs)
    batches_per_variant = (num_samples + batch_size - 1) // batch_size

    print(f"    Samples: {num_samples}, Batches per variant: {batches_per_variant}")

    for variant_id in range(actual_num_variants):
        variant_seed = seed + variant_id

        random.seed(variant_seed)
        np.random.seed(variant_seed)

        indices = list(range(num_samples))
        random.shuffle(indices)

        batches = []
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_graphs = [all_graphs[idx] for idx in batch_indices]
            batched_graph = Batch.from_data_list(batch_graphs)
            _add_device_ptr_tensors(batched_graph, batch_graphs)
            batches.append(batched_graph)

        variant_path = output_dir / f'variant_{variant_id}.pkl'
        with open(variant_path, 'wb') as f:
            pickle.dump(batches, f)

        if actual_num_variants <= 5 or variant_id == 0 or variant_id == actual_num_variants - 1:
            print(f"    Variant {variant_id}: {len(batches)} batches (seed={variant_seed})")
        elif variant_id == 1:
            print(f"    ...")

    metadata = {
        'batch_size': batch_size,
        'num_variants': actual_num_variants,
        'total_samples': num_samples,
        'batches_per_variant': batches_per_variant,
        'base_seed': seed,
        'is_validation': is_validation,
    }
    metadata_path = output_dir / 'metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"    Metadata saved to {metadata_path.name}")
    if not is_validation:
        print(f"    Training will rotate through {actual_num_variants} variants each epoch")


def load_prebatched_variant(split_dir: str, variant_id: int = 0) -> Tuple[List, Dict]:
    """
    Load a specific batching variant from a pre-batched dataset.

    Args:
        split_dir: Path to train/ or val/ directory
        variant_id: Which variant to load (0 to num_variants-1)

    Returns:
        batches: List of pre-batched PyG Batch objects
        metadata: Dict with batch_size, num_variants, etc.
    """
    split_dir = Path(split_dir)

    metadata_path = split_dir / 'metadata.pkl'
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    variant_path = split_dir / f'variant_{variant_id}.pkl'
    with open(variant_path, 'rb') as f:
        batches = pickle.load(f)

    return batches, metadata
