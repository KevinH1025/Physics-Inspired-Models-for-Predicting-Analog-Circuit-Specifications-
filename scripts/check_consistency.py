#!/usr/bin/env python3
"""Check current consistency: I_drain == I_source for MOSFETs, I_p == I_n for resistors."""
import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import (
    load_prebatched_variant, normalize_batches_vdc, normalize_batches_current,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def analyze(model_name, checkpoint_path):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model, config, stats = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    data_path = Path('datasets/opamp_3stage_fan_smc_v8_5k_nofil')
    batches = load_prebatched_variant(data_path / 'val', variant_id=0, device=device)

    curr_stats = stats.get('current', stats.get('curr', {}))
    curr_mean = curr_stats.get('mean', 0.0)
    curr_std = curr_stats.get('std', 1.0)
    vdc_stats = stats.get('vdc', {})

    normalize_batches_vdc(batches, vdc_stats.get('mean', 0.0), vdc_stats.get('std', 1.0))
    normalize_batches_current(batches, curr_mean, curr_std)

    mosfet_d_pred, mosfet_s_pred = [], []
    res_p_pred, res_n_pred = [], []
    mosfet_d_gt, mosfet_s_gt = [], []
    res_p_gt, res_n_gt = [], []

    with torch.no_grad():
        for batch in batches:
            result = model(batch)
            cp = result.get('node_currents', result.get('current_pred')).cpu()
            gt = batch.node_current_targets.cpu()
            mi = batch.mosfet_info.cpu()

            mosfet_d_pred.append(cp[mi[:, 1]])
            mosfet_s_pred.append(cp[mi[:, 2]])
            mosfet_d_gt.append(gt[mi[:, 1]])
            mosfet_s_gt.append(gt[mi[:, 2]])

            if hasattr(batch, 'resistor_info') and batch.resistor_info is not None and batch.resistor_info.numel() > 0:
                ri = batch.resistor_info.cpu()
                p_idx, n_idx = ri[:, 0].long(), ri[:, 1].long()
                res_p_pred.append(cp[p_idx])
                res_n_pred.append(cp[n_idx])
                res_p_gt.append(gt[p_idx])
                res_n_gt.append(gt[n_idx])

    dp = torch.cat(mosfet_d_pred).numpy()
    sp = torch.cat(mosfet_s_pred).numpy()
    dg = torch.cat(mosfet_d_gt).numpy()
    sg = torch.cat(mosfet_s_gt).numpy()

    print(f"\n--- MOSFET: I_drain vs I_source (N={len(dp)}) ---")
    print(f"  GT z-score diff:   mean={np.abs(dg-sg).mean():.6f}, max={np.abs(dg-sg).max():.6f}")
    pdiff = np.abs(dp - sp)
    print(f"  Pred z-score diff: mean={pdiff.mean():.4f}, std={pdiff.std():.4f}, max={pdiff.max():.4f}")

    d_real = 10**(dp * curr_std + curr_mean)
    s_real = 10**(sp * curr_std + curr_mean)
    rel = np.abs(d_real - s_real) / (np.maximum(d_real, s_real) + 1e-15)
    print(f"  Real relative |Id-Is|/max:")
    print(f"    mean={rel.mean()*100:.2f}%, median={np.median(rel)*100:.2f}%, max={rel.max()*100:.2f}%")
    print(f"    <1%: {(rel<0.01).mean()*100:.1f}%  <5%: {(rel<0.05).mean()*100:.1f}%  <10%: {(rel<0.10).mean()*100:.1f}%")

    if res_p_pred:
        pp = torch.cat(res_p_pred).numpy()
        np_ = torch.cat(res_n_pred).numpy()
        pg = torch.cat(res_p_gt).numpy()
        ng = torch.cat(res_n_gt).numpy()

        print(f"\n--- Resistor: I_p vs I_n (N={len(pp)}) ---")
        print(f"  GT z-score diff:   mean={np.abs(pg-ng).mean():.6f}, max={np.abs(pg-ng).max():.6f}")
        rdiff = np.abs(pp - np_)
        print(f"  Pred z-score diff: mean={rdiff.mean():.4f}, std={rdiff.std():.4f}, max={rdiff.max():.4f}")

        p_real = 10**(pp * curr_std + curr_mean)
        n_real = 10**(np_ * curr_std + curr_mean)
        rel_r = np.abs(p_real - n_real) / (np.maximum(p_real, n_real) + 1e-15)
        print(f"  Real relative |Ip-In|/max:")
        print(f"    mean={rel_r.mean()*100:.2f}%, median={np.median(rel_r)*100:.2f}%, max={rel_r.max()*100:.2f}%")
        print(f"    <1%: {(rel_r<0.01).mean()*100:.1f}%  <5%: {(rel_r<0.05).mean()*100:.1f}%  <10%: {(rel_r<0.10).mean()*100:.1f}%")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

base = Path('datasets/opamp_3stage_fan_smc_v8_5k_nofil/experiments/tower')
analyze("w512 (no KCL)", base / "fp32_bulk_voltctx_w512/best_model.pt")
analyze("w512 + KCL=1.0", base / "w512_kcl_1_0/best_model.pt")
