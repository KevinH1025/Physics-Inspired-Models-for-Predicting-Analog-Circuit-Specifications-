#!/usr/bin/env python3
"""
Compute relative errors for trained models and append results to training.log.

Usage:
    python scripts/compute_relative_errors.py --dataset datasets/opamp_2stage_5k_ss_v3 \
        --experiments 2stage_baseline_dc 2stage_soft_2t_w05 2stage_soft_3t_w10 \
                      2stage_soft_all_w05 2stage_kcl_blend03 2stage_ss_gmphy
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import (
    load_prebatched_variant, get_prediction_mask,
    normalize_batches_vdc, normalize_batches_current,
    compute_ss_normalization, normalize_batches_ss,
)
from src.training.metrics import denormalize_voltage, denormalize_current


def compute_relative_errors(experiment_dir, dataset_dir, device='cuda'):
    """Load model, run val inference, compute relative errors."""
    checkpoint_path = experiment_dir / 'best_model.pt'
    model, config, stats = load_checkpoint(str(checkpoint_path), device=device)
    model.eval()

    # Get normalization stats
    vdc_mean = stats.get('vdc', {}).get('mean', 0.0)
    vdc_std = stats.get('vdc', {}).get('std', 1.0)
    current_mean = stats.get('current', stats.get('curr', {})).get('mean', 0.0)
    current_std = stats.get('current', stats.get('curr', {})).get('std', 1.0)

    predict_currents = config.get('predict_currents', False)

    # Load val data (raw, not normalized)
    val_batches = load_prebatched_variant(dataset_dir / 'val', variant_id=0)

    # Save raw targets before normalization for relative error computation
    raw_vdc_per_batch = []
    raw_current_per_batch = []
    for batch in val_batches:
        mask = get_prediction_mask(batch)
        raw_vdc_per_batch.append(batch.vdc.flatten().clone())  # raw volts
        if predict_currents and hasattr(batch, 'node_current_targets') and batch.node_current_targets is not None:
            raw_current_per_batch.append(batch.node_current_targets.clone())
        else:
            raw_current_per_batch.append(None)

    # Normalize in-place (same as training script does)
    normalize_batches_vdc(val_batches, vdc_mean, vdc_std)
    if predict_currents:
        normalize_batches_current(val_batches, current_mean, current_std)

    # SS normalization: compute from training data if model has SS head
    has_ss = hasattr(val_batches[0], 'node_log_gm') and val_batches[0].node_log_gm is not None
    ss_enabled = config.get('ss_head_config', {}).get('enabled', False)
    ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std = 0.0, 1.0, 0.0, 1.0
    if has_ss and ss_enabled:
        train_batches = load_prebatched_variant(dataset_dir / 'train', variant_id=0)
        ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std = compute_ss_normalization([train_batches])
        normalize_batches_ss(val_batches, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)
        del train_batches

    all_v_rel = []
    all_v_abs_mv = []
    all_i_rel = []
    all_i_abs_ua = []
    all_gm_rel = []
    all_gds_rel = []

    with torch.inference_mode():
        for i, batch in enumerate(val_batches):
            batch = batch.to(device)
            out_dict = model(batch)

            # Voltage — denormalize predictions, compare to raw targets
            mask = get_prediction_mask(batch)
            pred_norm = out_dict['node_voltages'][mask]
            pred_mv, _ = denormalize_voltage(pred_norm, pred_norm, vdc_mean, vdc_std)
            pred_v = pred_mv / 1000.0  # to volts

            target_v = raw_vdc_per_batch[i].to(device)  # raw volts
            target_mv = target_v * 1000.0

            abs_err_mv = (pred_mv - target_mv).abs()
            rel_err_v = abs_err_mv / torch.clamp(target_mv.abs(), min=10.0)  # min 10mV denom

            all_v_abs_mv.extend(abs_err_mv.cpu().tolist())
            all_v_rel.extend((rel_err_v * 100).cpu().tolist())

            # Current
            if predict_currents and 'node_currents' in out_dict and out_dict['node_currents'] is not None:
                out_currents = out_dict['node_currents']
                current_mask = batch.has_current_mask
                if current_mask is not None and current_mask.any():
                    c_pred_norm = out_currents[current_mask]

                    # Denormalize predicted currents
                    c_pred_orig = torch.pow(10, c_pred_norm * current_std + current_mean)

                    # Raw target currents (already in amps)
                    c_target_orig = raw_current_per_batch[i].to(device)[current_mask].abs()

                    abs_err_a = (c_pred_orig - c_target_orig).abs()
                    abs_err_ua = abs_err_a * 1e6
                    # Only compute relative error for currents above 1nA floor
                    # (below that is leakage — relative error is meaningless)
                    above_floor = c_target_orig > 1e-9
                    if above_floor.any():
                        rel_err_i = abs_err_a[above_floor] / c_target_orig[above_floor]
                        all_i_rel.extend((rel_err_i * 100).cpu().tolist())

                    all_i_abs_ua.extend(abs_err_ua.cpu().tolist())

            # SS (gm/gds)
            gm_pred = out_dict.get('mosfet_gm_pred')
            gds_pred = out_dict.get('mosfet_gds_pred')
            drain_mask = getattr(batch, 'mosfet_drain_mask', None)
            if gm_pred is not None and drain_mask is not None and drain_mask.any():
                gm_pred_log = gm_pred[drain_mask] * ss_gm_std + ss_gm_mean
                gm_tgt_log = batch.node_log_gm[drain_mask] * ss_gm_std + ss_gm_mean
                gds_pred_log = gds_pred[drain_mask] * ss_gds_std + ss_gds_mean
                gds_tgt_log = batch.node_log_gds[drain_mask] * ss_gds_std + ss_gds_mean
                gm_rel = ((torch.pow(10, gm_pred_log) - torch.pow(10, gm_tgt_log)).abs() / torch.pow(10, gm_tgt_log).clamp(min=1e-15) * 100)
                gds_rel = ((torch.pow(10, gds_pred_log) - torch.pow(10, gds_tgt_log)).abs() / torch.pow(10, gds_tgt_log).clamp(min=1e-15) * 100)
                all_gm_rel.extend(gm_rel.cpu().tolist())
                all_gds_rel.extend(gds_rel.cpu().tolist())

    all_v_abs_mv = np.array(all_v_abs_mv)
    all_v_rel = np.array(all_v_rel)
    all_i_abs_ua = np.array(all_i_abs_ua) if all_i_abs_ua else np.array([])
    all_i_rel = np.array(all_i_rel) if all_i_rel else np.array([])

    results = {
        'v_mae_mv': all_v_abs_mv.mean(),
        'v_rel_median': np.median(all_v_rel),
        'v_rel_mean': all_v_rel.mean(),
        'v_rel_acc': {t: (all_v_rel < t).mean() * 100 for t in [1, 5, 10, 20]},
        'v_abs_acc': {t: (all_v_abs_mv < t).mean() * 100 for t in [80, 50, 20, 10]},
        'i_mae_ua': all_i_abs_ua.mean() if len(all_i_abs_ua) > 0 else 0.0,
        'i_rel_median': np.median(all_i_rel) if len(all_i_rel) > 0 else 0.0,
        'i_rel_mean': all_i_rel.mean() if len(all_i_rel) > 0 else 0.0,
        'i_rel_acc': {t: (all_i_rel < t).mean() * 100 for t in [1, 5, 10, 20]} if len(all_i_rel) > 0 else {t: 0.0 for t in [1, 5, 10, 20]},
        'i_abs_acc': {t: (all_i_abs_ua < t).mean() * 100 for t in [50, 20, 5, 2]} if len(all_i_abs_ua) > 0 else {t: 0.0 for t in [50, 20, 5, 2]},
        'ss_metrics': None,
    }

    if all_gm_rel:
        gm_rel_arr = np.array(all_gm_rel)
        gds_rel_arr = np.array(all_gds_rel)
        results['ss_metrics'] = {
            'gm_acc': {t: float((gm_rel_arr < t).mean() * 100) for t in [10, 20]},
            'gds_acc': {t: float((gds_rel_arr < t).mean() * 100) for t in [10, 20]},
            'gm_median': float(np.median(gm_rel_arr)),
            'gds_median': float(np.median(gds_rel_arr)),
        }

    return results


def format_results(r):
    """Format results as text block to append to training.log."""
    lines = []
    lines.append('')
    lines.append('==================================================')
    lines.append('=== RELATIVE ERROR ANALYSIS ===')
    lines.append('==================================================')
    lines.append(f'Best Val MAE: {r["v_mae_mv"]:.2f}mV (median rel: {r["v_rel_median"]:.2f}%, mean rel: {r["v_rel_mean"]:.2f}%)')
    lines.append(f'Best Val Current MAE: {r["i_mae_ua"]:.1f}µA (median rel: {r["i_rel_median"]:.2f}%, mean rel: {r["i_rel_mean"]:.2f}%)')

    vr = r['v_rel_acc']
    ir = r['i_rel_acc']
    lines.append(f'Voltage Rel Acc @1%: {vr[1]:5.2f}% | Current Rel Acc @1%: {ir[1]:5.2f}%')
    lines.append(f'Voltage Rel Acc @5%: {vr[5]:5.2f}% | Current Rel Acc @5%: {ir[5]:5.2f}%')
    lines.append(f'Voltage Rel Acc@10%: {vr[10]:5.2f}% | Current Rel Acc@10%: {ir[10]:5.2f}%')
    lines.append(f'Voltage Rel Acc@20%: {vr[20]:5.2f}% | Current Rel Acc@20%: {ir[20]:5.2f}%')

    ss = r.get('ss_metrics')
    if ss is not None:
        lines.append(f'gm  Acc@10%: {ss["gm_acc"][10]:5.1f}% (median: {ss["gm_median"]:.1f}%) | gds Acc@10%: {ss["gds_acc"][10]:5.1f}% (median: {ss["gds_median"]:.1f}%)')
        lines.append(f'gm  Acc@20%: {ss["gm_acc"][20]:5.1f}% | gds Acc@20%: {ss["gds_acc"][20]:5.1f}%')

    lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Compute relative errors for trained models')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    parser.add_argument('--experiments', nargs='+', required=True, help='Experiment names')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    for exp_name in args.experiments:
        exp_dir = dataset_dir / 'experiments' / exp_name
        log_path = exp_dir / 'training.log'

        if not (exp_dir / 'best_model.pt').exists():
            print(f'  SKIP {exp_name}: no best_model.pt')
            continue

        print(f'Processing {exp_name}...')
        results = compute_relative_errors(exp_dir, dataset_dir, device=args.device)

        result_text = format_results(results)
        print(result_text)

        # Append to training.log
        with open(log_path, 'a') as f:
            f.write(result_text)
        print(f'  Appended to {log_path}')
        print()


if __name__ == '__main__':
    main()
