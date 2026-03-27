#!/usr/bin/env python3
"""Analyze SS (gm/gds) prediction errors by region, device, and magnitude."""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import (
    load_prebatched_variant,
    add_ss_node_targets,
    compute_ss_normalization,
    normalize_batches_ss,
    normalize_batches_vdc,
    normalize_batches_current,
    compute_vdc_normalization,
    compute_current_normalization,
    attach_normalization_stats,
    add_vth_node_targets,
    add_region_node_targets,
    add_mosfet_gt_vov,
)


REGION_NAMES = {0: 'cutoff', 1: 'triode', 2: 'saturation'}


def extract_device_names(batch):
    """Extract MOSFET device names from node_names for the first graph."""
    if not hasattr(batch, 'node_names') or not batch.node_names:
        return [f'M{i}' for i in range(24)]

    names = batch.node_names[0]  # first graph
    ptr = batch.mosfet_ptr
    n_mosfets = (ptr[1] - ptr[0]).item()
    mosfet_info = batch.mosfet_info[:n_mosfets]

    # Map drain terminal index -> device name
    device_names = []
    for i in range(n_mosfets):
        drain_idx = mosfet_info[i, 1].item()
        if drain_idx < len(names):
            name = names[drain_idx]
            # Extract Xm{num} from e.g. "Xm12_drain"
            m = re.match(r'(Xm?\d+)_', name)
            device_names.append(m.group(1) if m else f'M{i}')
        else:
            device_names.append(f'M{i}')
    return device_names


def run_analysis(checkpoint_path, dataset_path, device='cuda'):
    dataset_path = Path(dataset_path)
    exp_dir = Path(checkpoint_path).parent

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model, config, stats = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    # Load training data for normalization
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'

    print("Loading training variants for normalization...")
    all_train_variants = []
    for vid in range(10):
        vpath = train_dir / f'variant_{vid}.pkl'
        if not vpath.exists():
            break
        all_train_variants.append(load_prebatched_variant(train_dir, variant_id=vid))

    train_batches = all_train_variants[0]
    val_batches = load_prebatched_variant(val_dir, variant_id=0)

    # Replicate normalization pipeline from train_v3.py
    vdc_mean, vdc_std = compute_vdc_normalization(dataset_path, train_batches, 'zscore', 1.8)
    for vb in all_train_variants:
        normalize_batches_vdc(vb, vdc_mean, vdc_std)
    normalize_batches_vdc(val_batches, vdc_mean, vdc_std)

    # Vov (before current normalization)
    for vb in all_train_variants:
        add_mosfet_gt_vov(vb)
    add_mosfet_gt_vov(val_batches)

    # Current normalization
    current_mean, current_std = compute_current_normalization(all_train_variants)
    for vb in all_train_variants:
        normalize_batches_current(vb, current_mean, current_std)
    normalize_batches_current(val_batches, current_mean, current_std)

    # Attach stats
    for vb in all_train_variants:
        attach_normalization_stats(vb, vdc_mean, vdc_std, current_mean, current_std)
    attach_normalization_stats(val_batches, vdc_mean, vdc_std, current_mean, current_std)

    # SS targets
    for vb in all_train_variants:
        add_ss_node_targets(vb)
    add_ss_node_targets(val_batches)

    # Vth and region targets
    for vb in all_train_variants:
        add_vth_node_targets(vb)
        add_region_node_targets(vb)
    add_vth_node_targets(val_batches)
    add_region_node_targets(val_batches)

    # SS normalization
    ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std = compute_ss_normalization(all_train_variants)
    for vb in all_train_variants:
        normalize_batches_ss(vb, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)
    normalize_batches_ss(val_batches, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)

    print(f"SS norm: gm mean={ss_gm_mean:.3f} std={ss_gm_std:.3f} | gds mean={ss_gds_mean:.3f} std={ss_gds_std:.3f}")

    # Get device names from first batch
    device_names = extract_device_names(val_batches[0])
    n_devices = len(device_names)
    print(f"Devices ({n_devices}): {', '.join(device_names)}")

    # Move val batches to device and run inference
    print("Running inference on validation set...")
    all_gm_pred_log = []
    all_gm_true_log = []
    all_gds_pred_log = []
    all_gds_true_log = []
    all_regions = []
    all_device_idx = []
    all_is_nmos = []
    all_gm_true_raw = []
    all_gds_true_raw = []

    with torch.no_grad():
        for batch in val_batches:
            batch = batch.to(device)
            out_dict = model(batch)

            gm_pred_node = out_dict.get('mosfet_gm_pred')
            gds_pred_node = out_dict.get('mosfet_gds_pred')

            if gm_pred_node is None:
                print("ERROR: model does not produce mosfet_gm_pred")
                return

            mosfet_info = batch.mosfet_info
            n_mosfets = mosfet_info.shape[0]
            n_graphs = batch.num_graphs
            mosfets_per_graph = n_mosfets // n_graphs

            # mosfet_gm_pred is already per-MOSFET [num_mosfets] from tower arch
            # Ground truth: use raw per-MOSFET values → log10 (avoids drain_mask mismatch)
            gm_true_log = torch.log10(batch.mosfet_gm.to(device).clamp(min=1e-15))
            gds_true_log = torch.log10(batch.mosfet_gds.to(device).clamp(min=1e-15))

            # Denormalize predictions from z-score to log10 space
            gm_pred_log = gm_pred_node * ss_gm_std + ss_gm_mean
            gds_pred_log = gds_pred_node * ss_gds_std + ss_gds_mean

            # Device index within each graph (0-23)
            dev_idx = torch.arange(n_mosfets, device=device) % mosfets_per_graph

            all_gm_pred_log.append(gm_pred_log.cpu())
            all_gm_true_log.append(gm_true_log.cpu())
            all_gds_pred_log.append(gds_pred_log.cpu())
            all_gds_true_log.append(gds_true_log.cpu())
            all_regions.append(batch.mosfet_region_labels.cpu())
            all_device_idx.append(dev_idx.cpu())
            all_is_nmos.append(mosfet_info[:, 6].cpu())
            all_gm_true_raw.append(batch.mosfet_gm.cpu())
            all_gds_true_raw.append(batch.mosfet_gds.cpu())

    # Concatenate
    gm_pred = torch.cat(all_gm_pred_log)
    gm_true = torch.cat(all_gm_true_log)
    gds_pred = torch.cat(all_gds_pred_log)
    gds_true = torch.cat(all_gds_true_log)
    regions = torch.cat(all_regions)
    dev_idx = torch.cat(all_device_idx)
    is_nmos = torch.cat(all_is_nmos)
    gm_raw = torch.cat(all_gm_true_raw)
    gds_raw = torch.cat(all_gds_true_raw)

    # Errors in log10 space
    gm_err = (gm_pred - gm_true).abs()
    gds_err = (gds_pred - gds_true).abs()
    gm_signed = gm_pred - gm_true
    gds_signed = gds_pred - gds_true

    # Relative errors in linear space
    gm_rel = (10**gm_pred - 10**gm_true).abs() / (10**gm_true).clamp(min=1e-15) * 100
    gds_rel = (10**gds_pred - 10**gds_true).abs() / (10**gds_true).clamp(min=1e-15) * 100

    total = len(gm_pred)
    print(f"\nTotal MOSFET samples: {total}")
    print(f"Overall gm MAE: {gm_err.mean():.4f} log10 | gds MAE: {gds_err.mean():.4f} log10")
    print(f"Overall gm <10%: {(gm_rel < 10).float().mean()*100:.1f}% | gds <10%: {(gds_rel < 10).float().mean()*100:.1f}%")

    # ========== TABLE 1: Error by operating region ==========
    print("\n" + "="*80)
    print("TABLE 1: Error by Operating Region")
    print("="*80)
    print(f"{'Region':<12} {'Count':>6} {'%':>5} | {'gm MAE':>8} {'gm<10%':>7} {'gm<20%':>7} | {'gds MAE':>8} {'gds<10%':>7} {'gds<20%':>7}")
    print("-"*80)
    for r in [0, 1, 2]:
        mask = regions == r
        n = mask.sum().item()
        if n == 0:
            continue
        pct = n / total * 100
        gm_m = gm_err[mask].mean().item()
        gds_m = gds_err[mask].mean().item()
        gm10 = (gm_rel[mask] < 10).float().mean().item() * 100
        gm20 = (gm_rel[mask] < 20).float().mean().item() * 100
        gds10 = (gds_rel[mask] < 10).float().mean().item() * 100
        gds20 = (gds_rel[mask] < 20).float().mean().item() * 100
        print(f"{REGION_NAMES[r]:<12} {n:>6} {pct:>4.1f}% | {gm_m:>8.4f} {gm10:>6.1f}% {gm20:>6.1f}% | {gds_m:>8.4f} {gds10:>6.1f}% {gds20:>6.1f}%")

    # ========== TABLE 2: Error by device ==========
    print("\n" + "="*100)
    print("TABLE 2: Error by Device (sorted by gds MAE)")
    print("="*100)
    print(f"{'Device':<8} {'Type':<5} {'Sat%':>5} {'Tri%':>5} {'Cut%':>5} | {'gm MAE':>8} {'gm<10%':>7} {'gmBias':>7} | {'gds MAE':>8} {'gds<10%':>7} {'gdsBias':>7}")
    print("-"*100)

    device_stats = []
    for d in range(n_devices):
        mask = dev_idx == d
        n = mask.sum().item()
        if n == 0:
            continue
        nmos = is_nmos[mask][0].item()
        dtype = 'NMOS' if nmos else 'PMOS'

        # Region distribution for this device
        reg = regions[mask]
        sat_pct = (reg == 2).float().mean().item() * 100
        tri_pct = (reg == 1).float().mean().item() * 100
        cut_pct = (reg == 0).float().mean().item() * 100

        gm_m = gm_err[mask].mean().item()
        gds_m = gds_err[mask].mean().item()
        gm10 = (gm_rel[mask] < 10).float().mean().item() * 100
        gds10 = (gds_rel[mask] < 10).float().mean().item() * 100
        gm_bias = gm_signed[mask].mean().item()
        gds_bias = gds_signed[mask].mean().item()

        device_stats.append((d, device_names[d], dtype, sat_pct, tri_pct, cut_pct,
                           gm_m, gm10, gm_bias, gds_m, gds10, gds_bias))

    # Sort by gds MAE descending
    device_stats.sort(key=lambda x: x[9], reverse=True)
    for d, name, dtype, sat, tri, cut, gm_m, gm10, gm_b, gds_m, gds10, gds_b in device_stats:
        print(f"{name:<8} {dtype:<5} {sat:>4.0f}% {tri:>4.0f}% {cut:>4.0f}% | "
              f"{gm_m:>8.4f} {gm10:>6.1f}% {gm_b:>+7.4f} | "
              f"{gds_m:>8.4f} {gds10:>6.1f}% {gds_b:>+7.4f}")

    # ========== TABLE 3: Error by magnitude bucket ==========
    print("\n" + "="*80)
    print("TABLE 3: Error by Magnitude Bucket (log10 space)")
    print("="*80)

    buckets = [(-7, -6), (-6, -5), (-5, -4), (-4, -3), (-3, -2)]

    print(f"\n{'Bucket':>12} | {'Count':>6} | {'gm MAE':>8} {'gm<10%':>7} {'gmMedRel':>8} | {'gds MAE':>8} {'gds<10%':>7} {'gdsMedRel':>8}")
    print("-"*80)

    for lo, hi in buckets:
        # gm buckets
        gm_mask = (gm_true >= lo) & (gm_true < hi)
        gds_mask = (gds_true >= lo) & (gds_true < hi)

        gm_n = gm_mask.sum().item()
        gds_n = gds_mask.sum().item()

        gm_m = gm_err[gm_mask].mean().item() if gm_n > 0 else 0
        gm10 = (gm_rel[gm_mask] < 10).float().mean().item() * 100 if gm_n > 0 else 0
        gm_med = gm_rel[gm_mask].median().item() if gm_n > 0 else 0

        gds_m = gds_err[gds_mask].mean().item() if gds_n > 0 else 0
        gds10 = (gds_rel[gds_mask] < 10).float().mean().item() * 100 if gds_n > 0 else 0
        gds_med = gds_rel[gds_mask].median().item() if gds_n > 0 else 0

        print(f"[{lo},{hi})  | gm:{gm_n:>5} | {gm_m:>8.4f} {gm10:>6.1f}% {gm_med:>7.1f}% | "
              f"gds:{gds_n:>5} | {gds_m:>8.4f} {gds10:>6.1f}% {gds_med:>7.1f}%")

    # ========== TABLE 4: Systematic bias ==========
    print("\n" + "="*80)
    print("TABLE 4: Systematic Bias (mean signed error in log10)")
    print("="*80)
    print(f"{'':>12} | {'gm bias':>8} {'gm std':>8} | {'gds bias':>8} {'gds std':>8}")
    print("-"*60)
    print(f"{'Overall':>12} | {gm_signed.mean():>+8.4f} {gm_signed.std():>8.4f} | {gds_signed.mean():>+8.4f} {gds_signed.std():>8.4f}")
    for r in [0, 1, 2]:
        mask = regions == r
        if mask.sum() == 0:
            continue
        print(f"{REGION_NAMES[r]:>12} | {gm_signed[mask].mean():>+8.4f} {gm_signed[mask].std():>8.4f} | "
              f"{gds_signed[mask].mean():>+8.4f} {gds_signed[mask].std():>8.4f}")

    # ========== TABLE 5: Worst outliers ==========
    print("\n" + "="*80)
    print("TABLE 5: Top 20 Worst gds Predictions")
    print("="*80)
    worst_gds = gds_err.argsort(descending=True)[:20]
    print(f"{'Device':<8} {'Region':<10} {'True':>8} {'Pred':>8} {'Err':>6} {'RelErr%':>8}")
    print("-"*55)
    for idx in worst_gds:
        d = dev_idx[idx].item()
        r = regions[idx].item()
        print(f"{device_names[d]:<8} {REGION_NAMES.get(r,'?'):<10} {gds_true[idx]:>8.3f} {gds_pred[idx]:>8.3f} "
              f"{gds_err[idx]:>6.3f} {gds_rel[idx]:>7.1f}%")

    # ========== PLOTS ==========
    print("\nGenerating plots...")

    # Plot 1: Scatter pred vs actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {0: 'red', 1: 'orange', 2: 'blue'}

    # Subsample for plotting (max 5000 points)
    n_plot = min(5000, len(gm_pred))
    idx_plot = torch.randperm(len(gm_pred))[:n_plot]

    for ax, pred, true, title in [
        (axes[0], gm_pred, gm_true, 'gm (log10)'),
        (axes[1], gds_pred, gds_true, 'gds (log10)'),
    ]:
        for r in [0, 1, 2]:
            mask = regions[idx_plot] == r
            if mask.sum() == 0:
                continue
            ax.scatter(true[idx_plot][mask], pred[idx_plot][mask],
                      c=colors[r], alpha=0.3, s=5, label=REGION_NAMES[r])

        lims = [min(true.min(), pred.min()).item(), max(true.max(), pred.max()).item()]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('True (log10)')
        ax.set_ylabel('Predicted (log10)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')

    plt.tight_layout()
    scatter_path = exp_dir / 'ss_analysis_scatter.png'
    plt.savefig(scatter_path, dpi=150)
    print(f"Saved scatter plot: {scatter_path}")
    plt.close()

    # Plot 2: Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, err, title in [
        (axes[0], gm_err, 'gm error distribution'),
        (axes[1], gds_err, 'gds error distribution'),
    ]:
        ax.hist(err.numpy(), bins=80, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(err.mean(), color='red', linestyle='--', label=f'mean={err.mean():.4f}')
        ax.axvline(err.median(), color='green', linestyle='--', label=f'median={err.median():.4f}')
        ax.set_xlabel('Absolute Error (log10)')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    hist_path = exp_dir / 'ss_analysis_hist.png'
    plt.savefig(hist_path, dpi=150)
    print(f"Saved histogram: {hist_path}")
    plt.close()

    # Plot 3: Per-device error bar chart
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Re-sort by device index for the bar chart
    device_stats_sorted = sorted(device_stats, key=lambda x: x[0])
    names = [s[1] for s in device_stats_sorted]
    gm_maes = [s[6] for s in device_stats_sorted]
    gds_maes = [s[9] for s in device_stats_sorted]
    dtypes = [s[2] for s in device_stats_sorted]
    bar_colors = ['#2196F3' if t == 'NMOS' else '#E91E63' for t in dtypes]

    x = np.arange(len(names))
    axes[0].bar(x, gm_maes, color=bar_colors, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('gm MAE (log10)')
    axes[0].set_title('gm MAE by Device (blue=NMOS, pink=PMOS)')
    axes[0].axhline(np.mean(gm_maes), color='gray', linestyle='--', alpha=0.5)

    axes[1].bar(x, gds_maes, color=bar_colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('gds MAE (log10)')
    axes[1].set_title('gds MAE by Device (blue=NMOS, pink=PMOS)')
    axes[1].axhline(np.mean(gds_maes), color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    device_path = exp_dir / 'ss_analysis_per_device.png'
    plt.savefig(device_path, dpi=150)
    print(f"Saved per-device plot: {device_path}")
    plt.close()

    # ========== VOLTAGE PER-NET AND CURRENT PER-DEVICE ==========
    print("\nComputing voltage per-net and current per-device errors...")

    all_v_pred_per_net = []   # list of [n_nets] per batch
    all_v_true_per_net = []
    all_i_pred_per_dev = []   # list of [n_devices] per batch
    all_i_true_per_dev = []

    with torch.no_grad():
        for batch in val_batches:
            batch = batch.to(device)
            out_dict = model(batch)
            v_pred_all = out_dict['node_voltages']  # [N_total] z-scored
            n_graphs = batch.num_graphs
            ptr = batch.ptr

            for g in range(n_graphs):
                start, end = ptr[g].item(), ptr[g + 1].item()
                n_per = end - start

                # --- Voltage per net ---
                # train_mask selects net nodes we predict (exclude known: vdd/gnd/input)
                tmask_g = batch.train_mask[start:end]
                v_pred_g = v_pred_all[start:end][tmask_g]  # z-scored
                # vdc is stacked per graph: need to slice correctly
                # vdc is indexed by train_indices which are per-graph
                # Simpler: denormalize full predictions, use node_voltage_targets
                v_pred_raw = v_pred_all[start:end] * vdc_std + vdc_mean
                v_true_raw = batch.node_voltage_targets[start:end]

                # output_node_mask marks the internal net nodes we care about
                omask_g = batch.output_node_mask[start:end]
                # Also exclude known voltage nodes
                kmask_g = batch.known_voltage_mask[start:end]
                net_mask = omask_g & ~kmask_g

                v_pred_nets = v_pred_raw[net_mask]
                v_true_nets = v_true_raw[net_mask]
                all_v_pred_per_net.append(v_pred_nets.cpu())
                all_v_true_per_net.append(v_true_nets.cpu())

                # --- Current per device ---
                i_pred_all = out_dict.get('node_currents')
                if i_pred_all is not None:
                    mosfet_ptr = batch.mosfet_ptr
                    m_start = mosfet_ptr[g].item()
                    m_end = mosfet_ptr[g + 1].item()
                    mi_g = batch.mosfet_info[m_start:m_end]

                    # Current is predicted per terminal; use drain terminal
                    drain_indices = mi_g[:, 1] + start  # global indices
                    i_pred_drain = i_pred_all[drain_indices]  # z-scored log10
                    i_pred_raw = i_pred_drain * current_std + current_mean  # log10(|I|)

                    # Ground truth current at drain terminals
                    i_true_log = batch.node_current_targets[drain_indices]  # log10(|I|+eps) z-scored
                    i_true_raw = i_true_log * current_std + current_mean

                    all_i_pred_per_dev.append(i_pred_raw.cpu())
                    all_i_true_per_dev.append(i_true_raw.cpu())

    # Stack per-net voltage errors: [n_samples, n_nets]
    if all_v_pred_per_net:
        n_nets = all_v_pred_per_net[0].shape[0]
        v_pred_stack = torch.stack(all_v_pred_per_net)  # [n_samples, n_nets]
        v_true_stack = torch.stack(all_v_true_per_net)
        v_err_mV = (v_pred_stack - v_true_stack).abs() * 1000  # mV
        v_mae_per_net = v_err_mV.mean(dim=0)  # [n_nets]

        # Get net names from first batch
        first_batch = val_batches[0]
        if hasattr(first_batch, 'node_names') and first_batch.node_names:
            all_names = first_batch.node_names[0]
            n_per = (first_batch.ptr[1] - first_batch.ptr[0]).item()
            omask = first_batch.output_node_mask[:n_per]
            kmask = first_batch.known_voltage_mask[:n_per]
            net_mask_names = omask & ~kmask
            net_names = [all_names[i] for i in range(n_per) if net_mask_names[i]]
        else:
            net_names = [f'net_{i}' for i in range(n_nets)]

        # Print table
        print(f"\n{'='*60}")
        print("VOLTAGE MAE PER NET (mV)")
        print(f"{'='*60}")
        print(f"{'Net':<20} {'MAE (mV)':>10} {'Median (mV)':>12}")
        print(f"{'-'*60}")
        v_med_per_net = v_err_mV.median(dim=0).values
        sorted_idx = v_mae_per_net.argsort(descending=True)
        for i in sorted_idx:
            print(f"{net_names[i]:<20} {v_mae_per_net[i]:>10.2f} {v_med_per_net[i]:>12.2f}")
        print(f"{'Overall':<20} {v_mae_per_net.mean():>10.2f} {v_med_per_net.mean():>12.2f}")

        # Plot 4: Voltage MAE per net
        fig, ax = plt.subplots(figsize=(16, 5))
        x_net = np.arange(n_nets)
        # Sort by net index for consistent ordering
        ax.bar(x_net, v_mae_per_net.numpy(), color='#4CAF50', alpha=0.8)
        ax.set_xticks(x_net)
        ax.set_xticklabels(net_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Voltage MAE (mV)')
        ax.set_title('Voltage MAE per Net')
        ax.axhline(v_mae_per_net.mean().item(), color='gray', linestyle='--', alpha=0.5,
                    label=f'mean={v_mae_per_net.mean():.1f} mV')
        ax.legend(fontsize=8)
        plt.tight_layout()
        vnet_path = exp_dir / 'voltage_per_net.png'
        plt.savefig(vnet_path, dpi=150)
        print(f"Saved voltage per-net plot: {vnet_path}")
        plt.close()

    # Stack per-device current errors: [n_samples, n_devices]
    if all_i_pred_per_dev:
        n_devs = all_i_pred_per_dev[0].shape[0]
        i_pred_stack = torch.stack(all_i_pred_per_dev)  # [n_samples, n_devices]
        i_true_stack = torch.stack(all_i_true_per_dev)
        # MAE in linear space (µA): 10^pred - 10^true
        i_pred_lin = 10 ** i_pred_stack  # Amps
        i_true_lin = 10 ** i_true_stack
        i_err_uA = (i_pred_lin - i_true_lin).abs() * 1e6  # µA
        i_mae_per_dev = i_err_uA.mean(dim=0)  # [n_devices]

        # Also compute relative error
        i_rel_per_dev = ((i_pred_lin - i_true_lin).abs() / i_true_lin.clamp(min=1e-15) * 100).mean(dim=0)

        # Print table
        print(f"\n{'='*60}")
        print("CURRENT MAE PER DEVICE (µA)")
        print(f"{'='*60}")
        print(f"{'Device':<10} {'Type':<5} {'MAE (µA)':>10} {'MedRel%':>10}")
        print(f"{'-'*60}")
        i_med_rel = ((i_pred_lin - i_true_lin).abs() / i_true_lin.clamp(min=1e-15) * 100).median(dim=0).values
        sorted_idx = i_mae_per_dev.argsort(descending=True)
        for i in sorted_idx:
            dtype_str = 'NMOS' if is_nmos[i].item() else 'PMOS'
            print(f"{device_names[i]:<10} {dtype_str:<5} {i_mae_per_dev[i]:>10.2f} {i_med_rel[i]:>10.1f}%")
        print(f"{'Overall':<10} {'':5} {i_mae_per_dev.mean():>10.2f} {i_med_rel.mean():>10.1f}%")

        # Plot 5: Current MAE per device
        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        x_dev = np.arange(n_devs)
        dev_colors = ['#2196F3' if is_nmos[i].item() else '#E91E63' for i in range(n_devs)]

        axes[0].bar(x_dev, i_mae_per_dev.numpy(), color=dev_colors, alpha=0.8)
        axes[0].set_xticks(x_dev)
        axes[0].set_xticklabels([device_names[i] for i in range(n_devs)], rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Current MAE (µA)')
        axes[0].set_title('Current MAE per Device (blue=NMOS, pink=PMOS)')
        axes[0].axhline(i_mae_per_dev.mean().item(), color='gray', linestyle='--', alpha=0.5)

        axes[1].bar(x_dev, i_med_rel.numpy(), color=dev_colors, alpha=0.8)
        axes[1].set_xticks(x_dev)
        axes[1].set_xticklabels([device_names[i] for i in range(n_devs)], rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Current Median Rel Error (%)')
        axes[1].set_title('Current Median Relative Error per Device (blue=NMOS, pink=PMOS)')
        axes[1].axhline(i_med_rel.mean().item(), color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        idev_path = exp_dir / 'current_per_device.png'
        plt.savefig(idev_path, dpi=150)
        print(f"Saved current per-device plot: {idev_path}")
        plt.close()

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze SS prediction errors')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_analysis(args.checkpoint, args.dataset, args.device)
