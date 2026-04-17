#!/usr/bin/env python3
"""
Re-batch an existing dataset, adding terminal_train_mask and terminal_vdc attributes.

This avoids re-running SPICE simulations. It loads the raw sample pkl files,
adds the new attributes to each graph, and re-creates the pre-batched variants.

Usage:
    python scripts/rebatch_dataset.py --dataset datasets/opamp_5k_onehead_kcl_v1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pickle
import torch

from src.data.batching import create_prebatched_dataset


def add_terminal_train_mask(graph):
    """Add terminal_train_mask and terminal_vdc to an existing graph."""
    num_terms = graph.num_terminals
    src, dst = graph.edge_index

    # Find terminal->net edges (src is terminal, dst is net)
    is_term_to_net = (src < num_terms) & (dst >= num_terms)
    term_indices = src[is_term_to_net]
    net_indices = dst[is_term_to_net]

    # Mark terminals connected to nets with train_mask=True
    terminal_train_mask = torch.zeros(graph.train_mask.shape[0], dtype=torch.bool)
    for ti, ni in zip(term_indices, net_indices):
        if graph.train_mask[ni]:
            terminal_train_mask[ti] = True

    # Extract voltage targets for these terminals
    terminal_train_indices = terminal_train_mask.nonzero(as_tuple=True)[0]
    terminal_vdc = graph.node_voltage_targets[terminal_train_indices].unsqueeze(-1)

    graph.terminal_train_mask = terminal_train_mask
    graph.terminal_vdc = terminal_vdc

    return graph


def main():
    parser = argparse.ArgumentParser(description='Re-batch dataset with terminal supervision attributes')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--num-variants', type=int, default=10, help='Number of batching variants')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    for split_name in ['train', 'val', 'test']:
        pkl_path = dataset_path / f'dataset_{split_name}.pkl'
        if not pkl_path.exists():
            print(f"Skipping {split_name} (no {pkl_path.name})")
            continue

        print(f"\n=== Processing {split_name} ===")
        with open(pkl_path, 'rb') as f:
            samples = pickle.load(f)
        print(f"Loaded {len(samples)} samples from {pkl_path.name}")

        if len(samples) == 0:
            print(f"  Empty split, skipping")
            continue

        # Add new attributes to each graph
        for sample in samples:
            add_terminal_train_mask(sample['graph'])

        # Verify
        g = samples[0]['graph']
        print(f"  terminal_train_mask count: {g.terminal_train_mask.sum().item()}")
        print(f"  terminal_vdc shape: {g.terminal_vdc.shape}")
        print(f"  train_mask count: {g.train_mask.sum().item()}")

        # Save updated samples back
        with open(pkl_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"  Saved updated samples to {pkl_path.name}")

        # Re-batch
        split_output_dir = dataset_path / split_name
        split_output_dir.mkdir(exist_ok=True)
        is_val = (split_name != 'train')
        print(f"  Re-batching...")
        create_prebatched_dataset(
            samples, split_output_dir,
            batch_size=args.batch_size,
            num_variants=args.num_variants,
            seed=args.seed,
            is_validation=is_val,
        )

    print("\nDone! You can now train with use_terminal_voltage_loss: true")


if __name__ == '__main__':
    main()
