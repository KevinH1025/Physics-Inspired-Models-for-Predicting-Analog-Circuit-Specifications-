#!/usr/bin/env python3
"""Plot distribution of gm and gds values from the prebatched dataset.

Creates a 2x2 figure:
  Top-left:     Histogram of raw log10(gm) values
  Top-right:    Histogram of raw log10(gds) values
  Bottom-left:  Histogram of z-scored gm values
  Bottom-right: Histogram of z-scored gds values

Usage:
    python scripts/plot_gm_gds_distribution.py [--dataset PATH]
"""
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_all_variants(split_dir: Path):
    """Load all variant_*.pkl files from a split directory."""
    batches = []
    for vf in sorted(split_dir.glob('variant_*.pkl')):
        with open(vf, 'rb') as f:
            batches.extend(pickle.load(f))
    return batches


def collect_log_gm_gds(batches):
    """Collect raw log10(gm) and log10(gds) from drain-masked nodes."""
    all_log_gm = []
    all_log_gds = []
    for batch in batches:
        mask = batch.mosfet_drain_mask
        if mask.any():
            all_log_gm.append(batch.node_log_gm[mask].cpu().numpy())
            all_log_gds.append(batch.node_log_gds[mask].cpu().numpy())
    return np.concatenate(all_log_gm), np.concatenate(all_log_gds)


def main():
    parser = argparse.ArgumentParser(description='Plot gm/gds distributions')
    parser.add_argument('--dataset', type=str,
                        default='datasets/opamp_3stage_fan_smc_v8_5k_nofil',
                        help='Path to prebatched dataset directory')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'

    # --- Load training data to compute normalization stats ---
    print(f"Loading training data from {train_dir} ...")
    train_batches = load_all_variants(train_dir)
    print(f"  Loaded {len(train_batches)} training batches")

    train_log_gm, train_log_gds = collect_log_gm_gds(train_batches)
    gm_mean, gm_std = float(train_log_gm.mean()), float(train_log_gm.std())
    gds_mean, gds_std = float(train_log_gds.mean()), float(train_log_gds.std())

    print(f"  gm  stats: mean={gm_mean:.4f}, std={gm_std:.4f}")
    print(f"  gds stats: mean={gds_mean:.4f}, std={gds_std:.4f}")
    print(f"  Total training drain nodes: {len(train_log_gm)}")

    # --- Load validation data too ---
    print(f"Loading validation data from {val_dir} ...")
    val_batches = load_all_variants(val_dir)
    print(f"  Loaded {len(val_batches)} validation batches")

    val_log_gm, val_log_gds = collect_log_gm_gds(val_batches)

    # --- Combine all data for plotting ---
    all_log_gm = np.concatenate([train_log_gm, val_log_gm])
    all_log_gds = np.concatenate([train_log_gds, val_log_gds])
    print(f"Total drain nodes (train+val): {len(all_log_gm)}")

    # Compute z-scored versions using training stats
    all_z_gm = (all_log_gm - gm_mean) / gm_std
    all_z_gds = (all_log_gds - gds_mean) / gds_std

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    hist_kwargs = dict(bins=120, alpha=0.75, edgecolor='white', linewidth=0.3)
    color_gm = '#2980b9'
    color_gds = '#c0392b'

    # Top-left: raw log10(gm)
    ax = axes[0, 0]
    ax.hist(all_log_gm, color=color_gm, **hist_kwargs)
    ax.axvline(gm_mean, color='#e74c3c', linewidth=2, linestyle='--', label='mean')
    ax.set_xlabel('log10(gm) [S]')
    ax.set_ylabel('Count')
    ax.set_title('Raw log10(gm) Distribution', fontweight='bold')
    ax.annotate(f'mean = {gm_mean:.3f}\nstd  = {gm_std:.3f}',
                xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # Top-right: raw log10(gds)
    ax = axes[0, 1]
    ax.hist(all_log_gds, color=color_gds, **hist_kwargs)
    ax.axvline(gds_mean, color='#2980b9', linewidth=2, linestyle='--', label='mean')
    ax.set_xlabel('log10(gds) [S]')
    ax.set_ylabel('Count')
    ax.set_title('Raw log10(gds) Distribution', fontweight='bold')
    ax.annotate(f'mean = {gds_mean:.3f}\nstd  = {gds_std:.3f}',
                xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # Bottom-left: z-scored gm
    ax = axes[1, 0]
    ax.hist(all_z_gm, color=color_gm, **hist_kwargs)
    ax.axvline(0, color='#e74c3c', linewidth=2, linestyle='--', label='mean (0)')
    z_gm_mean = float(all_z_gm.mean())
    z_gm_std = float(all_z_gm.std())
    ax.set_xlabel('z-scored gm')
    ax.set_ylabel('Count')
    ax.set_title('Z-scored gm Distribution', fontweight='bold')
    ax.annotate(f'mean = {z_gm_mean:.3f}\nstd  = {z_gm_std:.3f}',
                xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # Bottom-right: z-scored gds
    ax = axes[1, 1]
    ax.hist(all_z_gds, color=color_gds, **hist_kwargs)
    ax.axvline(0, color='#2980b9', linewidth=2, linestyle='--', label='mean (0)')
    z_gds_mean = float(all_z_gds.mean())
    z_gds_std = float(all_z_gds.std())
    ax.set_xlabel('z-scored gds')
    ax.set_ylabel('Count')
    ax.set_title('Z-scored gds Distribution', fontweight='bold')
    ax.annotate(f'mean = {z_gds_mean:.3f}\nstd  = {z_gds_std:.3f}',
                xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.suptitle('gm / gds Value Distributions (3-stage opamp, 5k dataset)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = dataset_path / 'gm_gds_distribution.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to {out_path}")


if __name__ == '__main__':
    main()
