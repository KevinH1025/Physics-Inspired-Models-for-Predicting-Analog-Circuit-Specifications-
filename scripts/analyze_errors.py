#!/usr/bin/env python3
"""
Analyze prediction errors by node to identify where the model struggles.

Usage:
    python scripts/analyze_errors.py --checkpoint datasets/opamp_5k_onehead_constraints_v1/best_model.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from src.training.data_loading import (
    load_prebatched_variant,
    load_prebatched_metadata,
    normalize_batches_vdc,
    normalize_batches_current,
)
from src.training.checkpoint import load_checkpoint
from src.training.metrics import denormalize_voltage, denormalize_current


def analyze_errors(checkpoint_path: str, device: str = 'cuda', max_batches: int = None):
    """Analyze prediction errors by node type."""

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    model, config, stats = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    # Get normalization stats
    vdc_mean = stats['vdc']['mean']
    vdc_std = stats['vdc']['std']
    current_mean = stats['current']['mean']
    current_std = stats['current']['std']

    print(f"Voltage norm: mean={vdc_mean:.4f}, std={vdc_std:.4f}")
    print(f"Current norm: mean={current_mean:.2f}, std={current_std:.2f}")

    # Load validation data
    dataset_path = Path(checkpoint_path).parent
    val_dir = dataset_path / 'val'

    if not val_dir.exists():
        print(f"Validation directory not found: {val_dir}")
        return

    print(f"\nLoading validation data from {val_dir}")
    val_batches = load_prebatched_variant(val_dir, variant_id=0, device=device)
    print(f"Loaded {len(val_batches)} validation batches")

    # Normalize batches using checkpoint stats (batches are stored unnormalized)
    print("Normalizing batches...")
    normalize_batches_vdc(val_batches, vdc_mean, vdc_std)
    normalize_batches_current(val_batches, current_mean, current_std)

    # Collect errors by node name
    voltage_errors = defaultdict(list)  # node_name -> list of absolute errors (mV)
    current_errors = defaultdict(list)  # node_name -> list of absolute errors (uA)
    voltage_predictions = defaultdict(list)  # node_name -> list of (pred, target) pairs
    current_predictions = defaultdict(list)

    num_batches = len(val_batches) if max_batches is None else min(max_batches, len(val_batches))

    print(f"\nAnalyzing {num_batches} batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_batches[:num_batches], desc="Processing")):
            batch = batch.to(device)

            # Forward pass
            out_dict = model(batch)
            pred_v = out_dict['node_voltages']
            pred_i = out_dict.get('node_currents')

            # Get masks
            train_mask = batch.train_mask  # Nodes to predict
            has_current = batch.has_current_mask if hasattr(batch, 'has_current_mask') else None

            # Get targets
            target_v = batch.vdc.flatten()
            target_i = batch.node_current_targets if hasattr(batch, 'node_current_targets') else None

            # Get node names per graph
            node_names = batch.node_names  # List of lists
            ptr = batch.ptr  # Graph boundaries

            # Process each graph in batch
            num_graphs = len(ptr) - 1
            for g in range(num_graphs):
                start_idx = ptr[g].item()
                end_idx = ptr[g + 1].item()

                graph_names = node_names[g] if isinstance(node_names[g], list) else node_names[g]
                graph_mask = train_mask[start_idx:end_idx]

                # Voltage errors
                graph_pred_v = pred_v[start_idx:end_idx][graph_mask]

                # Need to get corresponding targets
                # The target indices align with masked predictions
                mask_indices = torch.where(graph_mask)[0]

                # Denormalize predictions
                pred_v_mv = (graph_pred_v * vdc_std + vdc_mean) * 1000

                # Get target slice - targets are only for train_mask nodes
                # Need to figure out target indexing
                # For prebatched, vdc is flattened for all masked nodes
                # Let's compute cumulative mask count
                cum_mask = train_mask[:start_idx].sum().item()
                num_masked = graph_mask.sum().item()
                graph_target_v = target_v[cum_mask:cum_mask + num_masked]
                target_v_mv = (graph_target_v * vdc_std + vdc_mean) * 1000

                # Compute errors
                v_errors = (pred_v_mv - target_v_mv).abs().cpu().numpy()

                # Map to node names
                masked_names = [graph_names[i] for i in mask_indices.cpu().numpy()]
                for name, err in zip(masked_names, v_errors):
                    voltage_errors[name].append(err)

                # Current errors
                if pred_i is not None and target_i is not None and has_current is not None:
                    graph_has_current = has_current[start_idx:end_idx]
                    if graph_has_current.any():
                        curr_indices = torch.where(graph_has_current)[0]

                        graph_pred_i = pred_i[start_idx:end_idx][graph_has_current]
                        graph_target_i = target_i[start_idx:end_idx][graph_has_current]

                        # Denormalize currents: I = 10^(norm * std + mean)
                        pred_i_ua = torch.pow(10, graph_pred_i * current_std + current_mean) * 1e6
                        target_i_ua = torch.pow(10, graph_target_i * current_std + current_mean) * 1e6

                        i_errors = (pred_i_ua - target_i_ua).abs().cpu().numpy()

                        curr_names = [graph_names[i] for i in curr_indices.cpu().numpy()]
                        for name, err in zip(curr_names, i_errors):
                            current_errors[name].append(err)

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("VOLTAGE ERROR ANALYSIS (mV)")
    print("=" * 80)

    voltage_stats = []
    for name, errors in voltage_errors.items():
        errors = np.array(errors)
        voltage_stats.append({
            'name': name,
            'mean': errors.mean(),
            'std': errors.std(),
            'max': errors.max(),
            'p90': np.percentile(errors, 90),
            'p99': np.percentile(errors, 99),
            'count': len(errors),
        })

    # Sort by mean error descending
    voltage_stats.sort(key=lambda x: x['mean'], reverse=True)

    print(f"\n{'Node Name':<20} {'Mean':>10} {'Std':>10} {'P90':>10} {'P99':>10} {'Max':>10} {'Count':>8}")
    print("-" * 80)
    for s in voltage_stats[:30]:  # Top 30 worst nodes
        print(f"{s['name']:<20} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p90']:>10.2f} {s['p99']:>10.2f} {s['max']:>10.2f} {s['count']:>8}")

    # Group by node type
    print("\n" + "=" * 80)
    print("VOLTAGE ERROR BY NODE TYPE")
    print("=" * 80)

    type_errors = defaultdict(list)
    for name, errors in voltage_errors.items():
        # Extract type from name (e.g., Xm1_drain -> MOSFET_drain, vout -> net)
        if '_' in name:
            parts = name.split('_')
            if parts[0].lower().startswith('xm'):
                node_type = f"MOSFET_{parts[-1]}"
            elif parts[0].lower().startswith('x'):
                node_type = f"Device_{parts[-1]}"
            else:
                node_type = name
        else:
            node_type = "Net" if name.islower() or name == '0' else name

        type_errors[node_type].extend(errors)

    type_stats = []
    for ntype, errors in type_errors.items():
        errors = np.array(errors)
        type_stats.append({
            'type': ntype,
            'mean': errors.mean(),
            'std': errors.std(),
            'p90': np.percentile(errors, 90),
            'count': len(errors),
        })

    type_stats.sort(key=lambda x: x['mean'], reverse=True)

    print(f"\n{'Node Type':<25} {'Mean':>10} {'Std':>10} {'P90':>10} {'Count':>10}")
    print("-" * 70)
    for s in type_stats:
        print(f"{s['type']:<25} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p90']:>10.2f} {s['count']:>10}")

    # Current errors
    if current_errors:
        print("\n" + "=" * 80)
        print("CURRENT ERROR ANALYSIS (µA)")
        print("=" * 80)

        current_stats = []
        for name, errors in current_errors.items():
            errors = np.array(errors)
            current_stats.append({
                'name': name,
                'mean': errors.mean(),
                'std': errors.std(),
                'max': errors.max(),
                'p90': np.percentile(errors, 90),
                'count': len(errors),
            })

        current_stats.sort(key=lambda x: x['mean'], reverse=True)

        print(f"\n{'Node Name':<20} {'Mean':>10} {'Std':>10} {'P90':>10} {'Max':>10} {'Count':>8}")
        print("-" * 70)
        for s in current_stats[:20]:
            print(f"{s['name']:<20} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p90']:>10.2f} {s['max']:>10.2f} {s['count']:>8}")

        # Group by device
        print("\n" + "=" * 80)
        print("CURRENT ERROR BY DEVICE")
        print("=" * 80)

        device_errors = defaultdict(list)
        for name, errors in current_errors.items():
            # Extract device name (e.g., Xm1_drain -> Xm1)
            if '_' in name:
                device = name.split('_')[0]
            else:
                device = name
            device_errors[device].extend(errors)

        device_stats = []
        for dev, errors in device_errors.items():
            errors = np.array(errors)
            device_stats.append({
                'device': dev,
                'mean': errors.mean(),
                'std': errors.std(),
                'p90': np.percentile(errors, 90),
                'count': len(errors),
            })

        device_stats.sort(key=lambda x: x['mean'], reverse=True)

        print(f"\n{'Device':<15} {'Mean':>10} {'Std':>10} {'P90':>10} {'Count':>10}")
        print("-" * 60)
        for s in device_stats:
            print(f"{s['device']:<15} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p90']:>10.2f} {s['count']:>10}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_v_errors = np.concatenate([np.array(e) for e in voltage_errors.values()])
    print(f"\nOverall Voltage MAE: {all_v_errors.mean():.2f} mV")
    print(f"Overall Voltage P90: {np.percentile(all_v_errors, 90):.2f} mV")
    print(f"Overall Voltage P99: {np.percentile(all_v_errors, 99):.2f} mV")

    if current_errors:
        all_i_errors = np.concatenate([np.array(e) for e in current_errors.values()])
        print(f"\nOverall Current MAE: {all_i_errors.mean():.2f} µA")
        print(f"Overall Current P90: {np.percentile(all_i_errors, 90):.2f} µA")
        print(f"Overall Current P99: {np.percentile(all_i_errors, 99):.2f} µA")

    # Identify worst nodes
    print("\n" + "=" * 80)
    print("TOP 10 WORST VOLTAGE NODES (by mean error)")
    print("=" * 80)
    for i, s in enumerate(voltage_stats[:10], 1):
        print(f"{i:2d}. {s['name']:<20} Mean={s['mean']:.1f}mV, Max={s['max']:.1f}mV")

    if current_errors:
        print("\n" + "=" * 80)
        print("TOP 10 WORST CURRENT NODES (by mean error)")
        print("=" * 80)
        for i, s in enumerate(current_stats[:10], 1):
            print(f"{i:2d}. {s['name']:<20} Mean={s['mean']:.1f}µA, Max={s['max']:.1f}µA")

    return voltage_stats, current_stats if current_errors else None


def main():
    parser = argparse.ArgumentParser(description='Analyze prediction errors by node')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-batches', type=int, default=None, help='Limit number of batches to analyze')
    args = parser.parse_args()

    analyze_errors(args.checkpoint, args.device, args.max_batches)


if __name__ == '__main__':
    main()
