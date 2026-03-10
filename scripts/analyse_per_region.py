#!/usr/bin/env python3
"""
Per-region error analysis: break down model predictions by MOSFET operating region.

Reports voltage, current, gm, and gds errors separately for:
  - Saturation (region=2)
  - Triode (region=1)
  - Cutoff (region=0)

Usage:
    python scripts/analyse_per_region.py --checkpoint datasets/opamp_5k_ss_v1/best_model.pt
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import (
    load_prebatched_variant,
    normalize_batches_vdc,
    normalize_batches_current,
    add_ss_node_targets,
    compute_ss_normalization,
    normalize_batches_ss,
)

REGION_NAMES = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation', -1: 'Unknown'}


def analyse_per_region(model, batches, device, vdc_mean, vdc_std,
                       current_mean, current_std, ss_gm_mean, ss_gm_std,
                       ss_gds_mean, ss_gds_std):
    """Run inference and collect per-region errors for every MOSFET."""
    model.eval()

    # Accumulators keyed by region label
    voltage_errors = defaultdict(list)   # drain-net voltage error (mV)
    current_errors = defaultdict(list)   # drain current error (uA)
    gm_errors = defaultdict(list)        # log10(gm) MAE (in log10 space)
    gds_errors = defaultdict(list)       # log10(gds) MAE (in log10 space)
    gm_pct_errors = defaultdict(list)    # gm relative % error
    gds_pct_errors = defaultdict(list)   # gds relative % error
    region_counts = defaultdict(int)

    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            out_dict = model(batch)

            full_v_pred = out_dict['node_voltages']          # [num_nodes] normalized
            full_i_pred = out_dict.get('node_currents')      # [num_nodes] normalized or None
            ss_gm_pred = out_dict.get('mosfet_gm_pred')      # [num_nodes] normalized or None
            ss_gds_pred = out_dict.get('mosfet_gds_pred')     # [num_nodes] normalized or None

            mosfet_info = batch.mosfet_info
            region_labels = batch.mosfet_region_labels if hasattr(batch, 'mosfet_region_labels') else None
            ptr = batch.ptr
            num_nodes = batch.x.shape[0]
            num_mosfets = mosfet_info.shape[0]
            num_graphs = ptr.shape[0] - 1
            mosfets_per_graph = num_mosfets // num_graphs

            # Global indices
            mosfet_graph_idx = torch.arange(num_mosfets, device=device) // mosfets_per_graph
            node_offsets = ptr[mosfet_graph_idx]
            drain_term_idx = mosfet_info[:, 1].long() + node_offsets  # terminal node (for current/SS)
            drain_net_idx = mosfet_info[:, 4].long() + node_offsets   # net node (for voltage)

            # --- Voltage at drain NET nodes ---
            # Build full target tensor: vdc is stored only at train_mask positions
            full_targets = torch.full((num_nodes,), float('nan'), device=device)
            train_mask = batch.train_mask
            train_indices = train_mask.nonzero(as_tuple=True)[0]
            full_targets[train_indices] = batch.vdc.flatten()

            v_pred_net = full_v_pred[drain_net_idx]
            v_target_net = full_targets[drain_net_idx]

            # Only keep MOSFETs whose drain net has a valid target (is a prediction node)
            has_v_target = ~torch.isnan(v_target_net)

            # Denormalize to mV
            v_pred_mv = (v_pred_net * vdc_std + vdc_mean) * 1000
            v_target_mv = (v_target_net * vdc_std + vdc_mean) * 1000
            v_errors_mv = (v_pred_mv - v_target_mv).abs()

            # --- Current at drain TERMINAL nodes ---
            if full_i_pred is not None and hasattr(batch, 'node_current_targets'):
                i_pred_drain = full_i_pred[drain_term_idx]
                i_target_drain = batch.node_current_targets[drain_term_idx]
                has_curr = batch.has_current_mask[drain_term_idx] if hasattr(batch, 'has_current_mask') else torch.ones(num_mosfets, dtype=torch.bool, device=device)
                # Denormalize: log z-score -> amps -> uA
                i_pred_real = torch.pow(10, i_pred_drain * current_std + current_mean)
                i_target_real = torch.pow(10, i_target_drain * current_std + current_mean)
                i_errors_ua = (i_pred_real - i_target_real).abs() * 1e6
                has_current = True
            else:
                has_current = False

            # --- gm / gds at drain TERMINAL nodes ---
            has_ss = ss_gm_pred is not None and hasattr(batch, 'node_log_gm')
            if has_ss:
                gm_pred_drain = ss_gm_pred[drain_term_idx]   # normalized
                gds_pred_drain = ss_gds_pred[drain_term_idx]
                gm_target_drain = batch.node_log_gm[drain_term_idx]   # normalized
                gds_target_drain = batch.node_log_gds[drain_term_idx]

                # Denormalize to log10 space
                gm_pred_log = gm_pred_drain * ss_gm_std + ss_gm_mean
                gm_target_log = gm_target_drain * ss_gm_std + ss_gm_mean
                gds_pred_log = gds_pred_drain * ss_gds_std + ss_gds_mean
                gds_target_log = gds_target_drain * ss_gds_std + ss_gds_mean

                # Absolute error in log10 space (decades of error)
                gm_log_err = (gm_pred_log - gm_target_log).abs()
                gds_log_err = (gds_pred_log - gds_target_log).abs()

                # Relative error in linear space: |10^pred - 10^target| / 10^target * 100
                gm_pred_lin = torch.pow(10, gm_pred_log)
                gm_target_lin = torch.pow(10, gm_target_log)
                gds_pred_lin = torch.pow(10, gds_pred_log)
                gds_target_lin = torch.pow(10, gds_target_log)

                gm_pct = (gm_pred_lin - gm_target_lin).abs() / gm_target_lin.clamp(min=1e-15) * 100
                gds_pct = (gds_pred_lin - gds_target_lin).abs() / gds_target_lin.clamp(min=1e-15) * 100

            # --- Group by region ---
            if region_labels is None:
                continue

            for r in [0, 1, 2]:
                mask = (region_labels.to(device) == r)
                if not mask.any():
                    continue
                n = mask.sum().item()
                region_counts[r] += n

                # Voltage: only for MOSFETs with valid drain net target
                v_valid = mask & has_v_target
                if v_valid.any():
                    voltage_errors[r].extend(v_errors_mv[v_valid].cpu().tolist())

                # Current: only for MOSFETs with valid current target
                if has_current:
                    c_valid = mask & has_curr
                    if c_valid.any():
                        current_errors[r].extend(i_errors_ua[c_valid].cpu().tolist())

                if has_ss:
                    gm_errors[r].extend(gm_log_err[mask].cpu().tolist())
                    gds_errors[r].extend(gds_log_err[mask].cpu().tolist())
                    gm_pct_errors[r].extend(gm_pct[mask].cpu().tolist())
                    gds_pct_errors[r].extend(gds_pct[mask].cpu().tolist())

    return {
        'voltage_errors': voltage_errors,
        'current_errors': current_errors,
        'gm_errors': gm_errors,
        'gds_errors': gds_errors,
        'gm_pct_errors': gm_pct_errors,
        'gds_pct_errors': gds_pct_errors,
        'region_counts': region_counts,
    }


def print_results(results):
    """Pretty-print per-region error breakdown."""
    counts = results['region_counts']
    total = sum(counts.values())

    print(f"\n{'='*70}")
    print(f"  PER-REGION ERROR ANALYSIS")
    print(f"{'='*70}")

    # Region distribution
    print(f"\n  Region Distribution:")
    for r in [2, 1, 0]:
        n = counts.get(r, 0)
        pct = n / total * 100 if total > 0 else 0
        print(f"    {REGION_NAMES[r]:12s}: {n:6d} MOSFETs ({pct:5.1f}%)")
    print(f"    {'Total':12s}: {total:6d}")

    # Voltage errors
    print(f"\n  Voltage Error at Drain Net (mV):")
    print(f"    {'Region':12s} {'Count':>7s} {'MAE':>8s} {'Median':>8s} {'P90':>8s} {'P95':>8s}")
    print(f"    {'-'*55}")
    for r in [2, 1, 0]:
        errs = results['voltage_errors'].get(r, [])
        if errs:
            errs = np.array(errs)
            print(f"    {REGION_NAMES[r]:12s} {len(errs):7d} {errs.mean():8.2f} {np.median(errs):8.2f} {np.percentile(errs, 90):8.2f} {np.percentile(errs, 95):8.2f}")

    # Current errors
    if any(results['current_errors'].values()):
        print(f"\n  Current Error at Drain (uA):")
        print(f"    {'Region':12s} {'Count':>7s} {'MAE':>8s} {'Median':>8s} {'P90':>8s} {'P95':>8s}")
        print(f"    {'-'*55}")
        for r in [2, 1, 0]:
            errs = results['current_errors'].get(r, [])
            if errs:
                errs = np.array(errs)
                print(f"    {REGION_NAMES[r]:12s} {len(errs):7d} {errs.mean():8.2f} {np.median(errs):8.2f} {np.percentile(errs, 90):8.2f} {np.percentile(errs, 95):8.2f}")

    # gm errors
    if any(results['gm_errors'].values()):
        print(f"\n  gm Error (decades, log10 space):")
        print(f"    {'Region':12s} {'Count':>7s} {'MAE':>8s} {'Median':>8s} {'P90':>8s}")
        print(f"    {'-'*47}")
        for r in [2, 1, 0]:
            errs = results['gm_errors'].get(r, [])
            if errs:
                errs = np.array(errs)
                print(f"    {REGION_NAMES[r]:12s} {len(errs):7d} {errs.mean():8.4f} {np.median(errs):8.4f} {np.percentile(errs, 90):8.4f}")

        print(f"\n  gm Relative Error (%):")
        print(f"    {'Region':12s} {'Count':>7s} {'Median':>8s} {'P75':>8s} {'P90':>8s}")
        print(f"    {'-'*47}")
        for r in [2, 1, 0]:
            errs = results['gm_pct_errors'].get(r, [])
            if errs:
                errs = np.array(errs)
                # Clip outliers for cleaner display
                print(f"    {REGION_NAMES[r]:12s} {len(errs):7d} {np.median(errs):8.1f} {np.percentile(errs, 75):8.1f} {np.percentile(errs, 90):8.1f}")

    # gds errors
    if any(results['gds_errors'].values()):
        print(f"\n  gds Error (decades, log10 space):")
        print(f"    {'Region':12s} {'Count':>7s} {'MAE':>8s} {'Median':>8s} {'P90':>8s}")
        print(f"    {'-'*47}")
        for r in [2, 1, 0]:
            errs = results['gds_errors'].get(r, [])
            if errs:
                errs = np.array(errs)
                print(f"    {REGION_NAMES[r]:12s} {len(errs):7d} {errs.mean():8.4f} {np.median(errs):8.4f} {np.percentile(errs, 90):8.4f}")

        print(f"\n  gds Relative Error (%):")
        print(f"    {'Region':12s} {'Count':>7s} {'Median':>8s} {'P75':>8s} {'P90':>8s}")
        print(f"    {'-'*47}")
        for r in [2, 1, 0]:
            errs = results['gds_pct_errors'].get(r, [])
            if errs:
                errs = np.array(errs)
                print(f"    {REGION_NAMES[r]:12s} {len(errs):7d} {np.median(errs):8.1f} {np.percentile(errs, 75):8.1f} {np.percentile(errs, 90):8.1f}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Per-region error analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset dir (default: parent of checkpoint)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_dir = Path(args.dataset) if args.dataset else checkpoint_path.parent
    device = args.device

    print(f"Loading checkpoint: {checkpoint_path}")
    model, config, stats = load_checkpoint(str(checkpoint_path), device=device)

    # Load data
    split_dir = dataset_dir / args.split
    print(f"Loading {args.split} data from: {split_dir}")
    batches = load_prebatched_variant(split_dir, variant_id=0, device=device)
    print(f"  Loaded {len(batches)} batches")

    # Normalization stats
    vdc_stats = stats.get('vdc', {})
    current_stats = stats.get('current', stats.get('curr', {}))
    vdc_mean = vdc_stats.get('mean', 0.0)
    vdc_std = vdc_stats.get('std', 1.0)
    current_mean = current_stats.get('mean', 0.0)
    current_std = current_stats.get('std', 1.0)

    # Normalize voltage/current targets
    normalize_batches_vdc(batches, vdc_mean, vdc_std)
    sample = batches[0]
    if hasattr(sample, 'node_current_targets') and sample.node_current_targets is not None:
        normalize_batches_current(batches, current_mean, current_std)

    # SS targets + normalization
    add_ss_node_targets(batches)
    ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std = compute_ss_normalization([batches])
    normalize_batches_ss(batches, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)
    print(f"  SS norm: gm(mean={ss_gm_mean:.3f}, std={ss_gm_std:.3f}), gds(mean={ss_gds_mean:.3f}, std={ss_gds_std:.3f})")

    # Run analysis
    print(f"\nRunning per-region analysis...")
    results = analyse_per_region(
        model, batches, device,
        vdc_mean, vdc_std, current_mean, current_std,
        ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std,
    )

    print_results(results)


if __name__ == '__main__':
    main()
