#!/usr/bin/env python3
"""Patch existing dataset to use log-scale normalization for W and L.

The original normalization used linear min-max for W and L even though
they are sampled log-uniformly. This causes 50% of samples to be crammed
into 8% of the feature range.

This script re-normalizes x[:, 0] (W) and x[:, 1] (L) using the param_specs
from the dataset config (respecting scale: log).

Usage:
    python scripts/patch_dataset_log_norm.py \
        --dataset datasets/opamp_3stage_fan_smc_v6 \
        --config configs/opamp_dataset/opamp_3stage_fan_smc_wide_20k.yaml
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def log_normalize(value: float, min_val: float, max_val: float) -> float:
    log_val = math.log10(max(value, 1e-30))
    log_min = math.log10(max(min_val, 1e-30))
    log_max = math.log10(max(max_val, 1e-30))
    return (log_val - log_min) / (log_max - log_min) if log_max > log_min else 0.0


def log_normalize_wl_ratio(wl: float, w_spec: dict, l_spec: dict) -> float:
    wl_min = w_spec['min'] / l_spec['max']
    wl_max = w_spec['max'] / l_spec['min']
    log_wl = math.log10(max(wl, 1e-30))
    log_wl_min = math.log10(wl_min)
    log_wl_max = math.log10(wl_max)
    return (log_wl - log_wl_min) / (log_wl_max - log_wl_min) if log_wl_max > log_wl_min else 0.5


def linear_normalize(value: float, min_val: float, max_val: float) -> float:
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    param_specs = config.get('parameters', {})
    device_to_group = config.get('device_to_group', {})

    print(f"Patching {dataset_path}/dataset.pkl")
    with open(dataset_path / 'dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples, x shape: {data[0]['graph'].x.shape}")

    # Verify feature layout: x[:, 0]=W, x[:, 1]=L, x[:, 2]=W/L, x[:, 3]=M, x[:, 4:]=global
    sample = data[0]
    node_names = sample['graph'].node_names
    params = sample['params']

    # Build per-node (W_old, L_old, W_new, L_new) for verification on first sample
    print("\nVerification on sample 0:")
    for i, name in enumerate(node_names):
        if name.upper().endswith('_DRAIN'):
            device = name[:name.rfind('_')]
            device_id = device.upper().replace('X', '')
            group = device_to_group.get(device_id)
            if group:
                w_key = f'W_{group}'
                l_key = f'L_{group}'
                w_spec = param_specs.get(w_key, {})
                l_spec = param_specs.get(l_key, {})
                w_raw = params.get(w_key, None)
                l_raw = params.get(l_key, None)
                if w_raw and l_raw and w_spec and l_spec:
                    w_old = sample['graph'].x[i, 0].item()
                    l_old = sample['graph'].x[i, 1].item()
                    wl_old = sample['graph'].x[i, 2].item()
                    w_new = log_normalize(w_raw, w_spec['min'], w_spec['max']) if w_spec.get('scale') == 'log' else w_old
                    l_new = log_normalize(l_raw, l_spec['min'], l_spec['max']) if l_spec.get('scale') == 'log' else l_old
                    wl_new = log_normalize_wl_ratio(w_raw / l_raw, w_spec, l_spec) if (w_spec.get('scale') == 'log' and l_spec.get('scale') == 'log') else wl_old
                    print(f"  {device_id} ({group}): W {w_old:.4f}->{w_new:.4f}, L {l_old:.4f}->{l_new:.4f}, W/L {wl_old:.4f}->{wl_new:.4f}")
                    break

    # Patch all samples
    for sample in data:
        graph = sample['graph']
        params = sample['params']
        node_names = graph.node_names
        x = graph.x

        for i, name in enumerate(node_names):
            name_upper = name.upper()
            for suffix in ['_DRAIN', '_GATE', '_SOURCE', '_BULK']:
                if name_upper.endswith(suffix):
                    device = name[:len(name) - len(suffix)]
                    device_id = device.upper().replace('X', '')
                    group = device_to_group.get(device_id)
                    if group:
                        # W
                        w_key = f'W_{group}'
                        w_spec = param_specs.get(w_key, {})
                        w_raw = params.get(w_key)
                        if w_spec.get('scale') == 'log' and w_raw:
                            x[i, 0] = log_normalize(w_raw, w_spec['min'], w_spec['max'])
                        # L
                        l_key = f'L_{group}'
                        l_spec = param_specs.get(l_key, {})
                        l_raw = params.get(l_key)
                        if l_spec.get('scale') == 'log' and l_raw:
                            x[i, 1] = log_normalize(l_raw, l_spec['min'], l_spec['max'])
                        # W/L ratio (use full log range, not clipped at 1.0)
                        if (w_raw and l_raw and
                                w_spec.get('scale') == 'log' and l_spec.get('scale') == 'log'):
                            x[i, 2] = log_normalize_wl_ratio(w_raw / l_raw, w_spec, l_spec)
                    break

    print(f"\nSaving patched dataset.pkl...")
    with open(dataset_path / 'dataset.pkl', 'wb') as f:
        pickle.dump(data, f)

    # Re-split and re-save train/val pkls
    import random
    import numpy as np
    from src.data.batching import create_prebatched_dataset

    n = len(data)
    indices = list(range(n))
    random.seed(88)
    np.random.seed(88)
    random.shuffle(indices)

    n_train = int(n * 0.8)
    train_samples = [data[i] for i in indices[:n_train]]
    val_samples = [data[i] for i in indices[n_train:]]

    with open(dataset_path / 'dataset_train.pkl', 'wb') as f:
        pickle.dump(train_samples, f)
    with open(dataset_path / 'dataset_val.pkl', 'wb') as f:
        pickle.dump(val_samples, f)
    with open(dataset_path / 'dataset_test.pkl', 'wb') as f:
        pickle.dump([], f)
    print(f"Saved train ({len(train_samples)}) / val ({len(val_samples)}) splits")
    print("Done!")


if __name__ == '__main__':
    main()
