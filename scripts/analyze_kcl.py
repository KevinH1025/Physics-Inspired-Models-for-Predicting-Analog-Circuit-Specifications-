#!/usr/bin/env python3
"""Analyze per-net KCL violations: GT filter drop counts and per-net breakdown."""

import sys
import os
import pickle
import argparse
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.config import load_config, parse_training_config
from src.training.data_loading import load_prebatched_variant
from src.training.checkpoint import create_model_from_args
from src.training.losses import compute_kcl_loss


def get_node_names(dataset_path):
    """Load node names from dataset.pkl."""
    with open(os.path.join(dataset_path, 'dataset.pkl'), 'rb') as f:
        ds = pickle.load(f)
    return ds.node_names if hasattr(ds, 'node_names') else ds[0]['graph'].node_names


def compute_per_net_kcl(currents, edge_index, num_terminals, train_mask, ptr,
                        terminal_current_sign, current_mean, current_std,
                        kcl_include_mask, node_names):
    """Compute per-net KCL violations for all graphs in a batch.

    Returns list of dicts: [{net_name: {'rel_violation': float, 'abs_sum': float, 'n_terms': int}}, ...]
    """
    device = currents.device
    num_nodes = train_mask.size(0)

    # Build batch index
    graph_sizes = ptr[1:] - ptr[:-1]
    batch_idx = torch.repeat_interleave(torch.arange(len(graph_sizes), device=device), graph_sizes)
    local_idx = torch.arange(num_nodes, device=device) - ptr[batch_idx]

    n_terms = num_terminals if isinstance(num_terminals, int) else num_terminals[0].item()
    terminal_mask = local_idx < n_terms
    internal_net_mask = train_mask & ~terminal_mask

    num_graphs = len(graph_sizes)
    all_results = []

    for g in range(num_graphs):
        start = ptr[g].item()
        end = ptr[g + 1].item()
        results = {}

        for li in range(n_terms, end - start):
            gi = start + li
            net_name = node_names[li] if li < len(node_names) else f"net_{li}"

            if not train_mask[gi]:
                continue

            src, dst = edge_index
            mask = (dst == gi) & (src >= start) & (src < start + n_terms)
            term_indices = src[mask]

            if len(term_indices) == 0:
                continue

            total_signed = 0.0
            total_abs = 0.0
            n_included = 0

            for ti in term_indices:
                if kcl_include_mask is not None and not kcl_include_mask[ti]:
                    continue

                z = currents[ti].item()
                raw = 10 ** (z * current_std + current_mean)
                sign = terminal_current_sign[ti].item() if terminal_current_sign is not None else 1.0

                total_signed += raw * sign
                total_abs += raw
                n_included += 1

            if total_abs > 1e-9 and n_included >= 2:
                rel_viol = abs(total_signed) / total_abs
                results[net_name] = {
                    'rel_violation': rel_viol,
                    'abs_sum': total_abs,
                    'n_terms': n_included,
                }

        all_results.append(results)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--experiment', required=True, help='Experiment name under dataset/experiments/')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--with-model', action='store_true', help='Also run model predictions')
    args_cli = parser.parse_args()

    config = load_config(args_cli.config)
    args = parse_training_config(config)

    dataset_path = args['dataset']
    exp_dir = os.path.join(dataset_path, 'experiments', args_cli.experiment)

    node_names = get_node_names(dataset_path)
    n_terms = None
    for i, name in enumerate(node_names):
        if not any(c in name for c in ['_drain', '_gate', '_source', '_bulk', '_p', '_n']):
            n_terms = i
            break
    if n_terms is None:
        n_terms = 54  # fallback

    print(f"Dataset: {dataset_path}")
    print(f"Node names ({len(node_names)}): terminals={n_terms}, internal nets={len(node_names)-n_terms}")
    print(f"Internal nets: {node_names[n_terms:]}")
    print()

    # Load data
    print(f"Loading {args_cli.split} data...")
    split_dir = os.path.join(dataset_path, args_cli.split)
    all_batches = load_prebatched_variant(split_dir, variant_id=0, device=None)

    # Compute current normalization from train set
    train_dir = os.path.join(dataset_path, 'train')
    train_batches = load_prebatched_variant(train_dir, variant_id=0, device=None)

    # Compute log10 normalization stats from raw train currents
    log_eps = 1e-12
    all_log_currents = []
    for batch in train_batches:
        if hasattr(batch, 'node_current_targets') and hasattr(batch, 'has_current_mask'):
            mask = batch.has_current_mask
            if mask.any():
                raw = batch.node_current_targets[mask].abs()
                all_log_currents.append(torch.log10(raw + log_eps))
    all_log_currents = torch.cat(all_log_currents)
    current_mean = all_log_currents.mean().item()
    current_std = all_log_currents.std().item()
    print(f"Current normalization: mean={current_mean:.4f}, std={current_std:.4f}")

    # Normalize val currents in-place: z = (log10(|I| + eps) - mean) / std
    for batch in all_batches:
        if hasattr(batch, 'node_current_targets') and batch.node_current_targets is not None:
            raw = batch.node_current_targets.abs()
            batch.node_current_targets = (torch.log10(raw + log_eps) - current_mean) / current_std

    # Optionally load model
    model = None
    if args_cli.with_model:
        ckpt_path = os.path.join(exp_dir, 'best_model.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            src_cfg = ckpt.get('config', {})
            in_dim = ckpt['model_state_dict']['input_linear.weight'].shape[1]

            # Reconstruct args as namespace from checkpoint config
            from types import SimpleNamespace
            model_args = SimpleNamespace(**src_cfg)
            model, _ = create_model_from_args(model_args, in_dim, 'cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f"Loaded model from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
        else:
            print(f"Warning: no checkpoint at {ckpt_path}, skipping model predictions")

    print()

    # === FILTER PIPELINE TRACE (first graph) ===
    print("=" * 90)
    print("FILTER PIPELINE (graph 0 of first batch)")
    print("=" * 90)
    batch = all_batches[0]
    start = batch.ptr[0].item()
    end_g = batch.ptr[1].item()
    n_t = batch.num_terminals if isinstance(batch.num_terminals, int) else batch.num_terminals[0].item()
    src_e, dst_e = batch.edge_index
    _kcl_inc = batch.kcl_include_mask if hasattr(batch, 'kcl_include_mask') else None
    _term_sign = batch.terminal_current_sign if hasattr(batch, 'terminal_current_sign') else None

    all_internal = []
    for li in range(n_t, end_g - start):
        gi = start + li
        if batch.train_mask[gi]:
            all_internal.append((li, node_names[li] if li < len(node_names) else f"net_{li}"))

    print(f"  Total internal nets (train_mask=True): {len(all_internal)}")

    kcl_nets = []
    for li, net_name in all_internal:
        gi = start + li
        edge_mask = (dst_e == gi) & (src_e >= start) & (src_e < start + n_t)
        term_idx = src_e[edge_mask]

        terms_info = []
        n_incl = 0
        pos_count = 0
        neg_count = 0
        for ti in term_idx:
            local_ti = ti.item() - start
            tname = node_names[local_ti] if local_ti < len(node_names) else f"t_{local_ti}"
            incl = _kcl_inc[ti].item() if _kcl_inc is not None else True
            sign = _term_sign[ti].item() if _term_sign is not None else 0
            terms_info.append((tname, sign, incl))
            if incl:
                n_incl += 1
                if sign > 0: pos_count += 1
                if sign < 0: neg_count += 1

        has_both = pos_count > 0 and neg_count > 0
        enough = n_incl >= 2
        passes = has_both and enough

        status = "SKIP"
        kind = ""
        if passes:
            kind = "2t" if n_incl == 2 else f"{n_incl}t"
            status = f"KCL-{kind}"
            kcl_nets.append((li, net_name, n_incl, terms_info))

        term_str = ", ".join(
            f"{t[0]}({'+' if t[1]>0 else '-' if t[1]<0 else '?'}{'incl' if t[2] else 'EXCL'})"
            for t in terms_info
        )
        skip_reason = ""
        if not passes:
            if not enough: skip_reason = " (< 2 included terms)"
            elif not has_both: skip_reason = " (no sign diversity)"
        print(f"  {net_name:<16} [{status:>6}] total_terms={len(term_idx):>2} incl={n_incl:>2}{skip_reason}")
        print(f"    {term_str}")

    print(f"\n  Summary: {len(kcl_nets)} nets supervised out of {len(all_internal)} internal nets")
    print()

    # === RAW CURRENT DUMP (first 3 samples) ===
    print("=" * 90)
    print("RAW CURRENT DUMP (first 3 samples, before normalization)")
    print("=" * 90)
    # We need un-normalized currents for this. Reload first batch raw.
    raw_batches = load_prebatched_variant(os.path.join(dataset_path, args_cli.split), variant_id=0, device=None)
    raw_batch = raw_batches[0]

    n_samples_to_show = min(3, (raw_batch.ptr.size(0) - 1))
    for g in range(n_samples_to_show):
        g_start = raw_batch.ptr[g].item()
        g_end = raw_batch.ptr[g + 1].item()
        print(f"\n  --- Sample {g} ---")
        for li, net_name, n_incl_terms, terms_info in kcl_nets:
            gi = g_start + li
            edge_mask = (src_e == src_e)  # rebuild for this graph
            _src, _dst = raw_batch.edge_index
            emask = (_dst == gi) & (_src >= g_start) & (_src < g_start + n_t)
            tidx = _src[emask]

            total_signed = 0.0
            total_abs = 0.0
            parts = []
            for ti in tidx:
                local_ti = ti.item() - g_start
                tname = node_names[local_ti] if local_ti < len(node_names) else f"t_{local_ti}"
                incl = _kcl_inc[ti].item() if _kcl_inc is not None else True
                if not incl:
                    parts.append(f"{tname}(EXCL)")
                    continue
                sign = _term_sign[ti].item() if _term_sign is not None else 1.0
                raw_I = raw_batch.node_current_targets[ti].item()
                signed_I = raw_I * sign
                total_signed += signed_I
                total_abs += raw_I
                parts.append(f"{tname}={raw_I:.3e}({'+' if sign>0 else '-'}1)")

            rel = abs(total_signed) / (total_abs + 1e-30) * 100
            print(f"  {net_name:<14} {' '.join(parts)}")
            print(f"    {'':14} sum={total_signed:+.3e}  abs={total_abs:.3e}  rel={rel:.2f}%")

    del raw_batches, raw_batch
    print()

    # Analyze per-net KCL
    gt_stats = defaultdict(list)  # net_name -> list of rel_violations
    model_stats = defaultdict(list)
    gt_filter_stats = defaultdict(lambda: {'pass': 0, 'fail': 0})

    for batch in all_batches:
        gt_currents = batch.node_current_targets
        kcl_include = batch.kcl_include_mask if hasattr(batch, 'kcl_include_mask') else None
        term_sign = batch.terminal_current_sign if hasattr(batch, 'terminal_current_sign') else None

        # GT per-net analysis
        gt_results = compute_per_net_kcl(
            gt_currents, batch.edge_index, batch.num_terminals, batch.train_mask, batch.ptr,
            term_sign, current_mean, current_std, kcl_include, node_names
        )

        for graph_result in gt_results:
            for net_name, info in graph_result.items():
                gt_stats[net_name].append(info)
                # Check if GT filter would drop this (only for multi-term nets)
                if info['n_terms'] > 2:
                    if info['rel_violation'] < 0.1:
                        gt_filter_stats[net_name]['pass'] += 1
                    else:
                        gt_filter_stats[net_name]['fail'] += 1

        # Model per-net analysis
        if model is not None:
            with torch.no_grad():
                out = model(batch)
                pred_currents = out['node_currents']

            model_results = compute_per_net_kcl(
                pred_currents, batch.edge_index, batch.num_terminals, batch.train_mask, batch.ptr,
                term_sign, current_mean, current_std, kcl_include, node_names
            )

            for graph_result in model_results:
                for net_name, info in graph_result.items():
                    model_stats[net_name].append(info)

    # Print results
    print("=" * 90)
    print("PER-NET KCL ANALYSIS (Ground Truth)")
    print("=" * 90)
    print(f"{'Net':<16} {'#Terms':>6} {'#Samples':>8} {'Mean%':>8} {'Median%':>8} {'Max%':>8} {'<1%':>6} {'<10%':>6} {'>50%':>6}")
    print("-" * 90)

    for net_name in sorted(gt_stats.keys()):
        violations = [s['rel_violation'] for s in gt_stats[net_name]]
        n_terms = gt_stats[net_name][0]['n_terms']
        v = np.array(violations) * 100
        n = len(v)
        print(f"{net_name:<16} {n_terms:>6} {n:>8} {v.mean():>7.2f}% {np.median(v):>7.2f}% {v.max():>7.2f}% "
              f"{(v < 1).sum():>5} {(v < 10).sum():>5} {(v > 50).sum():>5}")

    print()
    print("=" * 90)
    print("GT FILTER DROP STATS (multi-term nets only, threshold=10%)")
    print("=" * 90)
    print(f"{'Net':<16} {'Pass':>8} {'Fail':>8} {'Drop%':>8}")
    print("-" * 50)

    total_pass = 0
    total_fail = 0
    for net_name in sorted(gt_filter_stats.keys()):
        p = gt_filter_stats[net_name]['pass']
        f = gt_filter_stats[net_name]['fail']
        total_pass += p
        total_fail += f
        pct = f / (p + f) * 100 if (p + f) > 0 else 0
        print(f"{net_name:<16} {p:>8} {f:>8} {pct:>7.1f}%")

    if total_pass + total_fail > 0:
        print(f"{'TOTAL':<16} {total_pass:>8} {total_fail:>8} {total_fail/(total_pass+total_fail)*100:>7.1f}%")

    if model_stats:
        print()
        print("=" * 90)
        print("PER-NET KCL ANALYSIS (Model Predictions)")
        print("=" * 90)
        print(f"{'Net':<16} {'#Terms':>6} {'#Samples':>8} {'Mean%':>8} {'Median%':>8} {'Max%':>8} {'<1%':>6} {'<10%':>6} {'>50%':>6}")
        print("-" * 90)

        for net_name in sorted(model_stats.keys()):
            violations = [s['rel_violation'] for s in model_stats[net_name]]
            n_terms = model_stats[net_name][0]['n_terms']
            v = np.array(violations) * 100
            n = len(v)
            print(f"{net_name:<16} {n_terms:>6} {n:>8} {v.mean():>7.2f}% {np.median(v):>7.2f}% {v.max():>7.2f}% "
                  f"{(v < 1).sum():>5} {(v < 10).sum():>5} {(v > 50).sum():>5}")

    # Also check which nets are being supervised by has_both_signs filter
    print()
    print("=" * 90)
    print("SIGN ANALYSIS (which nets have both +/- signs)")
    print("=" * 90)
    batch = all_batches[0]
    start = batch.ptr[0].item()
    n_t = batch.num_terminals if isinstance(batch.num_terminals, int) else batch.num_terminals[0].item()
    src, dst = batch.edge_index
    kcl_include = batch.kcl_include_mask if hasattr(batch, 'kcl_include_mask') else None
    term_sign = batch.terminal_current_sign if hasattr(batch, 'terminal_current_sign') else None

    for li in range(n_t, batch.ptr[1].item() - start):
        gi = start + li
        if not batch.train_mask[gi]:
            continue
        mask = (dst == gi) & (src >= start) & (src < start + n_t)
        term_indices = src[mask]

        net_name = node_names[li]
        signs = []
        names = []
        for ti in term_indices:
            included = True
            if kcl_include is not None and not kcl_include[ti]:
                included = False
            local_ti = ti.item() - start
            tname = node_names[local_ti]
            s = term_sign[ti].item() if term_sign is not None else '?'
            signs.append(s)
            names.append(f"{tname}({'incl' if included else 'excl'})")

        has_pos = any(s > 0 for s in signs if isinstance(s, (int, float)))
        has_neg = any(s < 0 for s in signs if isinstance(s, (int, float)))
        status = "KCL" if (has_pos and has_neg and len([s for i, s in enumerate(signs)]) >= 2) else "SKIP"

        print(f"  {net_name:<16} [{status}] terms={len(term_indices):>2}  signs={signs}  {', '.join(names)}")

    # === GT KCL FLOOR (normalized loss space) ===
    print()
    print("=" * 90)
    print("GT KCL FLOOR (same loss metric as training, using GT currents)")
    print("=" * 90)

    gt_two_losses = []
    gt_multi_losses = []
    total_two = 0
    total_multi = 0
    total_multi_dropped = 0

    for batch in all_batches:
        gt_c = batch.node_current_targets
        _ki = batch.kcl_include_mask if hasattr(batch, 'kcl_include_mask') else None
        _ts = batch.terminal_current_sign if hasattr(batch, 'terminal_current_sign') else None

        _, _, stats = compute_kcl_loss(
            node_currents=gt_c,
            edge_index=batch.edge_index,
            num_terminals=batch.num_terminals,
            train_mask=batch.train_mask,
            ptr=batch.ptr,
            terminal_current_sign=_ts,
            current_mean=current_mean,
            current_std=current_std,
            gt_currents=gt_c,
            kcl_include_mask=_ki,
            return_stats=True,
        )
        if 'two_term_loss' in stats:
            gt_two_losses.append(stats['two_term_loss'])
            total_two += stats.get('n_two_term', 0)
        if 'multi_term_loss' in stats:
            gt_multi_losses.append(stats['multi_term_loss'])
        total_multi += stats.get('n_multi_total', 0)
        total_multi_dropped += stats.get('n_multi_dropped', 0)

    if gt_two_losses:
        print(f"  2-term GT floor: {np.mean(gt_two_losses):.3e}  (across {total_two} net instances)")
    else:
        print(f"  2-term GT floor: N/A (no 2-term nets)")

    if gt_multi_losses:
        print(f"  3-term GT floor: {np.mean(gt_multi_losses):.3e}  (after GT filter)")
    else:
        print(f"  3-term GT floor: N/A (no valid 3-term nets)")

    print(f"  GT filter drops: {total_multi_dropped}/{total_multi} ({100*total_multi_dropped/(total_multi+1e-9):.1f}%)")
    print()
    print("If GT floor is near-zero, the loss formulation is correct and the data is consistent.")
    print("If GT floor is large, there's a data or formulation issue.")


if __name__ == '__main__':
    main()
