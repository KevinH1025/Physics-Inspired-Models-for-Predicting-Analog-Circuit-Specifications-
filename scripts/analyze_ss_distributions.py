#!/usr/bin/env python3
"""Analyze gm/gds distributions for 2-stage vs 3-stage datasets."""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_val_batch(dataset_path):
    """Load validation batch from prebatched dataset."""
    val_path = Path(dataset_path) / 'val' / 'variant_0.pkl'
    with open(val_path, 'rb') as f:
        batches = pickle.load(f)
    return batches[0] if isinstance(batches, list) else batches

def extract_per_device_ss(batch):
    """Extract per-device gm/gds values with device names."""
    num_graphs = batch.ptr.shape[0] - 1

    # Get mosfet_gm/gds (linear space, per MOSFET)
    mosfet_gm = batch.mosfet_gm.cpu().numpy()
    mosfet_gds = batch.mosfet_gds.cpu().numpy()

    # Compute log10 from linear values (filter out near-zero)
    valid_gm = mosfet_gm > 1e-15
    valid_gds = mosfet_gds > 1e-15
    log_gm = np.log10(mosfet_gm[valid_gm])
    log_gds = np.log10(mosfet_gds[valid_gds])

    # Per-device breakdown using mosfet_ptr if available
    mosfet_ptr = batch.mosfet_ptr.cpu().numpy() if hasattr(batch, 'mosfet_ptr') else None

    if mosfet_ptr is not None:
        devs_per_graph = mosfet_ptr[1] - mosfet_ptr[0]
    else:
        devs_per_graph = len(mosfet_gm) // num_graphs

    return {
        'mosfet_gm': mosfet_gm,
        'mosfet_gds': mosfet_gds,
        'log_gm': log_gm,
        'log_gds': log_gds,
        'num_graphs': num_graphs,
        'devs_per_graph': devs_per_graph,
    }

def plot_distributions(data_2stage, data_3stage, output_path):
    """Plot gm/gds distribution comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: gm distributions ---
    # Overall gm histogram
    ax = axes[0, 0]
    ax.hist(data_2stage['log_gm'], bins=80, alpha=0.6, label=f"2-stage (n={len(data_2stage['log_gm'])})", density=True, color='steelblue')
    ax.hist(data_3stage['log_gm'], bins=80, alpha=0.6, label=f"3-stage (n={len(data_3stage['log_gm'])})", density=True, color='coral')
    ax.set_xlabel('log10(gm) [S]')
    ax.set_ylabel('Density')
    ax.set_title('gm Distribution (all devices)')
    ax.legend()

    # Stats text
    for data, name, color in [(data_2stage, '2-stage', 'steelblue'), (data_3stage, '3-stage', 'coral')]:
        mean = np.mean(data['log_gm'])
        std = np.std(data['log_gm'])
        rng = np.ptp(data['log_gm'])
        ax.axvline(mean, color=color, linestyle='--', alpha=0.7)

    # Per-device gm boxplot - 2-stage
    ax = axes[0, 1]
    dpg_2 = data_2stage['devs_per_graph']
    n_graphs_2 = data_2stage['num_graphs']
    gm_2 = data_2stage['mosfet_gm']
    per_dev_gm_2 = {}
    for i in range(dpg_2):
        dev_vals = np.log10(np.maximum(gm_2[i::dpg_2], 1e-20))
        valid = dev_vals > -15
        dev_name = f'M{i}'
        per_dev_gm_2[dev_name] = dev_vals[valid]

    positions = np.arange(len(per_dev_gm_2))
    bp = ax.boxplot([per_dev_gm_2[f'M{i}'] for i in range(dpg_2)],
                    positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.set_xticklabels([f'M{i}' for i in range(dpg_2)], rotation=45, fontsize=7)
    ax.set_ylabel('log10(gm) [S]')
    ax.set_title(f'2-stage: Per-device gm ({dpg_2} MOSFETs)')

    # Per-device gm boxplot - 3-stage
    ax = axes[0, 2]
    dpg_3 = data_3stage['devs_per_graph']
    gm_3 = data_3stage['mosfet_gm']
    per_dev_gm_3 = {}
    for i in range(dpg_3):
        dev_vals = np.log10(np.maximum(gm_3[i::dpg_3], 1e-20))
        valid = dev_vals > -15
        per_dev_gm_3[f'M{i}'] = dev_vals[valid]

    positions = np.arange(len(per_dev_gm_3))
    bp = ax.boxplot([per_dev_gm_3[f'M{i}'] for i in range(dpg_3)],
                    positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.6)
    ax.set_xticklabels([f'M{i}' for i in range(dpg_3)], rotation=45, fontsize=7)
    ax.set_ylabel('log10(gm) [S]')
    ax.set_title(f'3-stage: Per-device gm ({dpg_3} MOSFETs)')

    # --- Row 2: gds distributions ---
    # Overall gds histogram
    ax = axes[1, 0]
    ax.hist(data_2stage['log_gds'], bins=80, alpha=0.6, label=f"2-stage (n={len(data_2stage['log_gds'])})", density=True, color='steelblue')
    ax.hist(data_3stage['log_gds'], bins=80, alpha=0.6, label=f"3-stage (n={len(data_3stage['log_gds'])})", density=True, color='coral')
    ax.set_xlabel('log10(gds) [S]')
    ax.set_ylabel('Density')
    ax.set_title('gds Distribution (all devices)')
    ax.legend()

    for data, name, color in [(data_2stage, '2-stage', 'steelblue'), (data_3stage, '3-stage', 'coral')]:
        mean = np.mean(data['log_gds'])
        ax.axvline(mean, color=color, linestyle='--', alpha=0.7)

    # Per-device gds boxplot - 2-stage
    ax = axes[1, 1]
    gds_2 = data_2stage['mosfet_gds']
    per_dev_gds_2 = {}
    for i in range(dpg_2):
        dev_vals = np.log10(np.maximum(gds_2[i::dpg_2], 1e-20))
        valid = dev_vals > -15
        per_dev_gds_2[f'M{i}'] = dev_vals[valid]

    bp = ax.boxplot([per_dev_gds_2[f'M{i}'] for i in range(dpg_2)],
                    positions=np.arange(dpg_2), widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.set_xticklabels([f'M{i}' for i in range(dpg_2)], rotation=45, fontsize=7)
    ax.set_ylabel('log10(gds) [S]')
    ax.set_title(f'2-stage: Per-device gds ({dpg_2} MOSFETs)')

    # Per-device gds boxplot - 3-stage
    ax = axes[1, 2]
    gds_3 = data_3stage['mosfet_gds']
    per_dev_gds_3 = {}
    for i in range(dpg_3):
        dev_vals = np.log10(np.maximum(gds_3[i::dpg_3], 1e-20))
        valid = dev_vals > -15
        per_dev_gds_3[f'M{i}'] = dev_vals[valid]

    bp = ax.boxplot([per_dev_gds_3[f'M{i}'] for i in range(dpg_3)],
                    positions=np.arange(dpg_3), widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.6)
    ax.set_xticklabels([f'M{i}' for i in range(dpg_3)], rotation=45, fontsize=7)
    ax.set_ylabel('log10(gds) [S]')
    ax.set_title(f'3-stage: Per-device gds ({dpg_3} MOSFETs)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

def print_stats(data, name):
    """Print distribution statistics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    log_gm = data['log_gm']
    log_gds = data['log_gds']
    dpg = data['devs_per_graph']

    print(f"  Devices per graph: {dpg}")
    print(f"  Num graphs: {data['num_graphs']}")
    print(f"  Total MOSFET samples: {len(data['mosfet_gm'])}")

    print(f"\n  --- Overall gm (log10 space) ---")
    print(f"  Mean:   {np.mean(log_gm):.4f}")
    print(f"  Std:    {np.std(log_gm):.4f}")
    print(f"  Min:    {np.min(log_gm):.4f}")
    print(f"  Max:    {np.max(log_gm):.4f}")
    print(f"  Range:  {np.ptp(log_gm):.4f} decades")
    print(f"  IQR:    {np.percentile(log_gm, 75) - np.percentile(log_gm, 25):.4f} decades")

    print(f"\n  --- Overall gds (log10 space) ---")
    print(f"  Mean:   {np.mean(log_gds):.4f}")
    print(f"  Std:    {np.std(log_gds):.4f}")
    print(f"  Min:    {np.min(log_gds):.4f}")
    print(f"  Max:    {np.max(log_gds):.4f}")
    print(f"  Range:  {np.ptp(log_gds):.4f} decades")
    print(f"  IQR:    {np.percentile(log_gds, 75) - np.percentile(log_gds, 25):.4f} decades")

    # Per-device stats
    print(f"\n  --- Per-device gm (log10 space) ---")
    print(f"  {'Device':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    mosfet_gm = data['mosfet_gm']
    for i in range(dpg):
        dev_gm = np.log10(np.maximum(mosfet_gm[i::dpg], 1e-20))
        valid = dev_gm > -15
        dev_gm = dev_gm[valid]
        if len(dev_gm) > 0:
            print(f"  M{i:<7} {np.mean(dev_gm):>8.3f} {np.std(dev_gm):>8.3f} {np.min(dev_gm):>8.3f} {np.max(dev_gm):>8.3f} {np.ptp(dev_gm):>8.3f}")

    print(f"\n  --- Per-device gds (log10 space) ---")
    print(f"  {'Device':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    mosfet_gds = data['mosfet_gds']
    for i in range(dpg):
        dev_gds = np.log10(np.maximum(mosfet_gds[i::dpg], 1e-20))
        valid = dev_gds > -15
        dev_gds = dev_gds[valid]
        if len(dev_gds) > 0:
            print(f"  M{i:<7} {np.mean(dev_gds):>8.3f} {np.std(dev_gds):>8.3f} {np.min(dev_gds):>8.3f} {np.max(dev_gds):>8.3f} {np.ptp(dev_gds):>8.3f}")

    # Intra-graph range (how different are gm values within a single graph?)
    print(f"\n  --- Intra-graph diversity ---")
    gm_ranges = []
    gds_ranges = []
    for g in range(min(data['num_graphs'], 1000)):
        start = g * dpg
        end = start + dpg
        g_gm = np.log10(np.maximum(mosfet_gm[start:end], 1e-20))
        g_gds = np.log10(np.maximum(mosfet_gds[start:end], 1e-20))
        valid_gm = g_gm[g_gm > -15]
        valid_gds = g_gds[g_gds > -15]
        if len(valid_gm) > 1:
            gm_ranges.append(np.ptp(valid_gm))
            gds_ranges.append(np.ptp(valid_gds))

    print(f"  gm intra-graph range:  mean={np.mean(gm_ranges):.3f}  std={np.std(gm_ranges):.3f} decades")
    print(f"  gds intra-graph range: mean={np.mean(gds_ranges):.3f}  std={np.std(gds_ranges):.3f} decades")


def load_from_raw(dataset_path, max_samples=1000):
    """Load gm/gds from raw dataset_train.pkl (for datasets without prebatched val)."""
    import torch
    pkl_path = Path(dataset_path) / 'dataset_val.pkl'
    if not pkl_path.exists():
        pkl_path = Path(dataset_path) / 'dataset_train.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    data = data[:max_samples]

    all_gm = []
    all_gds = []
    for d in data:
        g = d['graph'] if isinstance(d, dict) else d
        all_gm.append(g.mosfet_gm.cpu())
        all_gds.append(g.mosfet_gds.cpu())

    mosfet_gm = torch.cat(all_gm).numpy()
    mosfet_gds = torch.cat(all_gds).numpy()

    g0 = data[0]['graph'] if isinstance(data[0], dict) else data[0]
    devs_per_graph = len(g0.mosfet_gm)

    valid_gm = mosfet_gm > 1e-15
    valid_gds = mosfet_gds > 1e-15

    return {
        'mosfet_gm': mosfet_gm,
        'mosfet_gds': mosfet_gds,
        'log_gm': np.log10(mosfet_gm[valid_gm]),
        'log_gds': np.log10(mosfet_gds[valid_gds]),
        'num_graphs': len(data),
        'devs_per_graph': devs_per_graph,
    }


if __name__ == '__main__':
    dataset_2stage = 'datasets/opamp_5k_ss_v1'
    dataset_3stage_v3 = 'datasets/opamp_3stage_fan_smc_v3'
    dataset_3stage_v4 = 'datasets/opamp_3stage_fan_smc_v4'

    print("Loading 2-stage val batch...")
    batch_2 = load_val_batch(dataset_2stage)
    data_2 = extract_per_device_ss(batch_2)

    print("Loading 3-stage v3 (narrow) val batch...")
    batch_3v3 = load_val_batch(dataset_3stage_v3)
    data_3v3 = extract_per_device_ss(batch_3v3)

    print("Loading 3-stage v4 (wide) from raw...")
    data_3v4 = load_from_raw(dataset_3stage_v4)

    print_stats(data_2, "2-Stage Opamp (opamp_5k_ss_v1)")
    print_stats(data_3v3, "3-Stage v3 NARROW (opamp_3stage_fan_smc_v3)")
    print_stats(data_3v4, "3-Stage v4 WIDE (opamp_3stage_fan_smc_v4)")

    output_path = 'datasets/opamp_3stage_fan_smc_v4/ss_distribution_comparison.png'
    plot_distributions(data_2, data_3v4, output_path)

    # Also plot v3 vs v4 comparison
    output_path2 = 'datasets/opamp_3stage_fan_smc_v4/ss_v3_vs_v4.png'
    plot_distributions(data_3v3, data_3v4, output_path2)
