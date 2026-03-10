#!/usr/bin/env python3
"""
Extract per-device MOSFET samples from circuit simulation data.

Extracts (Vgs, Vds, W, L, is_nmos, I_ds) for each MOSFET in each circuit sample,
creating a dataset for training a standalone Device MLP.

Usage:
    python scripts/extract_device_data.py \
        --dataset datasets/opamp_5k_onehead_constraints_v3 \
        --output datasets/device_mlp/mosfet_samples.pkl
"""

import argparse
import pickle
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch


def extract_device_name_from_graph(graph, mosfet_idx: int) -> str:
    """Extract device name for a MOSFET from graph node names.

    Args:
        graph: PyG Data object with node_names
        mosfet_idx: Index into mosfet_info tensor

    Returns:
        Device name like 'XM1', 'Xm1', etc.
    """
    # Get gate terminal index from mosfet_info
    gate_term_idx = graph.mosfet_info[mosfet_idx, 0].item()

    # Node name is like 'XM1_gate' or 'Xm1_G'
    node_name = graph.node_names[gate_term_idx]

    # Extract device name (before underscore)
    device_name = node_name.split('_')[0]
    return device_name


def normalize_device_name(name: str) -> str:
    """Normalize device name for params lookup.

    Converts 'XM1', 'Xm1', 'xm1' -> 'M1' for params lookup.
    """
    return name.upper().lstrip('X')


def extract_device_samples(
    dataset_path: Path,
    split: str = 'train',
) -> List[Dict[str, Any]]:
    """Extract per-device MOSFET samples from dataset.

    Args:
        dataset_path: Path to dataset directory
        split: 'train' or 'val'

    Returns:
        List of device sample dicts with keys:
            - Vgs: Gate-source voltage (raw, not normalized)
            - Vds: Drain-source voltage (raw, not normalized)
            - W: Width in meters
            - L: Length in meters
            - is_nmos: 1 for NMOS, 0 for PMOS
            - I_ds: Drain-source current in Amps (positive)
            - region: Operating region (0=cutoff, 1=triode, 2=saturation, -1=unknown)
            - device_name: Device name like 'M1'
            - sample_id: Original circuit sample ID
    """
    # Load dataset
    dataset_file = dataset_path / f'dataset_{split}.pkl'
    print(f"Loading {dataset_file}...")

    with open(dataset_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} circuit samples")

    device_samples = []
    stats = defaultdict(int)

    for sample_idx, sample in enumerate(samples):
        graph = sample['graph']
        params = sample['params']
        specs = sample['specs']
        vdd = sample.get('vdd', 1.8)

        # Get ground truth voltages and currents from SPICE
        node_voltage_targets = graph.node_voltage_targets  # Raw voltages or normalized?
        device_currents = specs.get('_all_device_currents', {})
        mosfet_regions = specs.get('_mosfet_regions', {})

        # Check voltage normalization: if max voltage is close to 1.0, it's normalized [-1, 1]
        # If max voltage is around VDD (1.8V), it's raw
        max_v = node_voltage_targets.max().item()
        if max_v > 1.5:
            # Raw voltages
            voltage_scale = 1.0
            voltage_offset = 0.0
        else:
            # Normalized to [-1, 1]: V_raw = (V_norm + 1) * VDD / 2
            voltage_scale = vdd / 2.0
            voltage_offset = vdd / 2.0

        # Extract samples for each MOSFET
        num_mosfets = len(graph.mosfet_info)
        for mosfet_idx in range(num_mosfets):
            mosfet_row = graph.mosfet_info[mosfet_idx]

            # Get indices
            gate_term_idx = mosfet_row[0].item()
            drain_term_idx = mosfet_row[1].item()
            source_term_idx = mosfet_row[2].item()
            gate_net_idx = mosfet_row[3].item()
            drain_net_idx = mosfet_row[4].item()
            source_net_idx = mosfet_row[5].item()
            is_nmos = mosfet_row[6].item()

            # Get device name
            device_name = extract_device_name_from_graph(graph, mosfet_idx)
            norm_name = normalize_device_name(device_name)

            # Get voltages at net nodes (where voltage is actually defined)
            V_g_raw = node_voltage_targets[gate_net_idx].item()
            V_d_raw = node_voltage_targets[drain_net_idx].item()
            V_s_raw = node_voltage_targets[source_net_idx].item()

            # Denormalize if needed
            V_g = V_g_raw * voltage_scale + voltage_offset
            V_d = V_d_raw * voltage_scale + voltage_offset
            V_s = V_s_raw * voltage_scale + voltage_offset

            # Compute Vgs and Vds
            Vgs = V_g - V_s
            Vds = V_d - V_s

            # Get W and L from params
            w_key = f'W_{norm_name}'
            l_key = f'L_{norm_name}'

            W = params.get(w_key)
            L = params.get(l_key)

            if W is None or L is None:
                # Try alternative key format
                w_key_alt = f'W{norm_name}'
                l_key_alt = f'L{norm_name}'
                W = params.get(w_key_alt, params.get(w_key.lower(), None))
                L = params.get(l_key_alt, params.get(l_key.lower(), None))

            if W is None or L is None:
                stats['missing_params'] += 1
                continue

            # Get current from SPICE results
            # Try different naming conventions
            I_ds = None
            for name_variant in [device_name.lower(), norm_name.lower(), f'x{norm_name.lower()}']:
                if name_variant in device_currents:
                    I_ds = abs(device_currents[name_variant])
                    break

            if I_ds is None:
                stats['missing_current'] += 1
                continue

            # Filter out near-zero currents (numerical noise)
            if I_ds < 1e-15:
                stats['zero_current'] += 1
                continue

            # Get operating region
            region = -1  # unknown
            if mosfet_regions:
                for name_variant in [device_name.lower(), norm_name.lower(), f'x{norm_name.lower()}']:
                    if name_variant in mosfet_regions:
                        region_info = mosfet_regions[name_variant]
                        region_str = region_info.get('region', 'unknown')
                        region = {'cutoff': 0, 'triode': 1, 'saturation': 2}.get(region_str, -1)
                        break

            device_samples.append({
                'Vgs': Vgs,
                'Vds': Vds,
                'W': W,
                'L': L,
                'is_nmos': is_nmos,
                'I_ds': I_ds,
                'region': region,
                'device_name': norm_name,
                'sample_id': sample_idx,
            })
            stats['extracted'] += 1

        if (sample_idx + 1) % 500 == 0:
            print(f"  Processed {sample_idx + 1}/{len(samples)} samples, extracted {stats['extracted']} devices")

    print(f"\nExtraction complete:")
    print(f"  Total device samples: {stats['extracted']}")
    print(f"  Missing params: {stats['missing_params']}")
    print(f"  Missing current: {stats['missing_current']}")
    print(f"  Zero current (filtered): {stats['zero_current']}")

    return device_samples


def compute_statistics(samples: List[Dict]) -> Dict:
    """Compute statistics for the extracted samples."""
    Vgs_vals = [s['Vgs'] for s in samples]
    Vds_vals = [s['Vds'] for s in samples]
    W_vals = [s['W'] for s in samples]
    L_vals = [s['L'] for s in samples]
    I_vals = [s['I_ds'] for s in samples]
    regions = [s['region'] for s in samples]
    is_nmos = [s['is_nmos'] for s in samples]
    devices = [s['device_name'] for s in samples]

    log_I = [math.log10(max(i, 1e-15)) for i in I_vals]
    log_W = [math.log10(max(w, 1e-15)) for w in W_vals]
    log_L = [math.log10(max(l, 1e-15)) for l in L_vals]

    stats = {
        'num_samples': len(samples),
        'Vgs': {'min': min(Vgs_vals), 'max': max(Vgs_vals), 'mean': np.mean(Vgs_vals), 'std': np.std(Vgs_vals)},
        'Vds': {'min': min(Vds_vals), 'max': max(Vds_vals), 'mean': np.mean(Vds_vals), 'std': np.std(Vds_vals)},
        'log10_W': {'min': min(log_W), 'max': max(log_W), 'mean': np.mean(log_W), 'std': np.std(log_W)},
        'log10_L': {'min': min(log_L), 'max': max(log_L), 'mean': np.mean(log_L), 'std': np.std(log_L)},
        'log10_I': {'min': min(log_I), 'max': max(log_I), 'mean': np.mean(log_I), 'std': np.std(log_I)},
        'nmos_count': sum(is_nmos),
        'pmos_count': len(is_nmos) - sum(is_nmos),
        'region_counts': {
            'cutoff': sum(1 for r in regions if r == 0),
            'triode': sum(1 for r in regions if r == 1),
            'saturation': sum(1 for r in regions if r == 2),
            'unknown': sum(1 for r in regions if r == -1),
        },
        'device_counts': {},
    }

    for dev in set(devices):
        stats['device_counts'][dev] = sum(1 for d in devices if d == dev)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract per-device MOSFET samples from circuit data')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--splits', type=str, default='train,val', help='Comma-separated splits to extract')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = args.splits.split(',')

    all_samples = []
    for split in splits:
        dataset_file = dataset_path / f'dataset_{split}.pkl'
        if dataset_file.exists():
            print(f"\n=== Extracting from {split} split ===")
            samples = extract_device_samples(dataset_path, split)
            all_samples.extend(samples)
        else:
            print(f"Skipping {split} split (file not found)")

    print(f"\n=== Total samples: {len(all_samples)} ===")

    # Compute and print statistics
    stats = compute_statistics(all_samples)
    print("\n=== Statistics ===")
    print(f"Total samples: {stats['num_samples']}")
    print(f"NMOS: {stats['nmos_count']}, PMOS: {stats['pmos_count']}")
    print(f"\nVgs: [{stats['Vgs']['min']:.3f}, {stats['Vgs']['max']:.3f}], mean={stats['Vgs']['mean']:.3f}, std={stats['Vgs']['std']:.3f}")
    print(f"Vds: [{stats['Vds']['min']:.3f}, {stats['Vds']['max']:.3f}], mean={stats['Vds']['mean']:.3f}, std={stats['Vds']['std']:.3f}")
    print(f"log10(W): [{stats['log10_W']['min']:.2f}, {stats['log10_W']['max']:.2f}], mean={stats['log10_W']['mean']:.2f}")
    print(f"log10(L): [{stats['log10_L']['min']:.2f}, {stats['log10_L']['max']:.2f}], mean={stats['log10_L']['mean']:.2f}")
    print(f"log10(I): [{stats['log10_I']['min']:.2f}, {stats['log10_I']['max']:.2f}], mean={stats['log10_I']['mean']:.2f}")
    print(f"\nRegion distribution: {stats['region_counts']}")
    print(f"Device distribution: {stats['device_counts']}")

    # Save samples and stats
    output_data = {
        'samples': all_samples,
        'stats': stats,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n=== Saved to {output_path} ===")


if __name__ == '__main__':
    main()
