#!/usr/bin/env python3
"""Analyze tower model: embedding quality, prediction consistency, per-device errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import (
    load_prebatched_variant, normalize_batches_vdc, normalize_batches_current,
    add_ss_node_targets,
)


def extract_tower_embeddings(model, batch):
    """Extract embeddings from tower architecture using hooks."""
    model.eval()
    captured = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # For ModuleList layers, capture the input to the first layer (= tower input)
            # and output of the last layer
            captured[name] = output
        return hook_fn

    hooks = []
    # Hook the last layer of each tower to get its output
    hooks.append(model.backbone[-1].register_forward_hook(make_hook('backbone_last')))
    hooks.append(model.state_tower[-1].register_forward_hook(make_hook('state_last')))
    if hasattr(model, 'sensitivity_tower') and model.sensitivity_tower is not None:
        hooks.append(model.sensitivity_tower[-1].register_forward_hook(make_hook('sens_last')))

    with torch.no_grad():
        result = model(batch)

    for h in hooks:
        h.remove()

    return {
        'backbone': captured.get('backbone_last', None),
        'state': captured.get('state_last', None),
        'sensitivity': captured.get('sens_last', None),
        'result': result,
    }


def analyze_model(model_name, checkpoint_path, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    model, config, stats = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    data_path = Path('datasets/opamp_3stage_fan_smc_v8_5k_nofil')
    batches = load_prebatched_variant(data_path / 'val', variant_id=0, device=device)

    vdc_stats = stats.get('vdc', {})
    curr_stats = stats.get('current', stats.get('curr', {}))
    curr_mean, curr_std = curr_stats.get('mean', 0.0), curr_stats.get('std', 1.0)

    normalize_batches_vdc(batches, vdc_stats.get('mean', 0.0), vdc_stats.get('std', 1.0))
    normalize_batches_current(batches, curr_mean, curr_std)
    for b in batches:
        add_ss_node_targets([b])

    # Collect per-MOSFET data
    all_sens_emb = []       # sensitivity tower embedding at drain
    all_state_emb = []      # state tower embedding at drain
    all_backbone_emb = []   # backbone embedding at drain
    all_gm_gt = []
    all_gds_gt = []
    all_gm_pred = []
    all_gds_pred = []
    all_region = []
    all_is_nmos = []
    all_device_id = []
    all_drain_curr_pred = []
    all_source_curr_pred = []
    all_drain_curr_gt = []
    all_v_pred = []
    all_v_gt = []

    with torch.no_grad():
        for batch in batches:
            emb = extract_tower_embeddings(model, batch)
            result = emb['result']
            mi = batch.mosfet_info
            num_graphs = batch.ptr.shape[0] - 1
            mosfets_per_graph = mi.shape[0] // num_graphs

            # Node offsets for batched graph
            mosfet_graph_idx = torch.arange(mi.shape[0], device=device) // mosfets_per_graph
            offsets = batch.ptr[mosfet_graph_idx]

            drain_global = mi[:, 1].long() + offsets
            source_global = mi[:, 2].long() + offsets
            gate_global = mi[:, 0].long() + offsets

            # Embeddings at drain terminals
            if emb['sensitivity'] is not None:
                all_sens_emb.append(emb['sensitivity'][drain_global].cpu())
            all_state_emb.append(emb['state'][drain_global].cpu())
            all_backbone_emb.append(emb['backbone'][drain_global].cpu())

            # gm/gds targets
            all_gm_gt.append(batch.node_log_gm[drain_global].cpu())
            all_gds_gt.append(batch.node_log_gds[drain_global].cpu())

            # gm/gds predictions
            if 'mosfet_gm_pred' in result:
                all_gm_pred.append(result['mosfet_gm_pred'].cpu())
            if 'mosfet_gds_pred' in result:
                all_gds_pred.append(result['mosfet_gds_pred'].cpu())

            # Current predictions at drain vs source
            if 'node_currents' in result:
                all_drain_curr_pred.append(result['node_currents'][drain_global].cpu())
                all_source_curr_pred.append(result['node_currents'][source_global].cpu())
                all_drain_curr_gt.append(batch.node_current_targets[drain_global].cpu())

            # Voltage predictions
            if 'node_voltages' in result:
                train_mask = batch.train_mask
                v_pred = result['node_voltages'][train_mask].cpu()
                v_gt = batch.vdc.cpu() if hasattr(batch, 'vdc') else batch.node_voltage_targets[train_mask].cpu()
                all_v_pred.append(v_pred)
                all_v_gt.append(v_gt)

            # Region and device info
            if hasattr(batch, 'mosfet_region_labels'):
                all_region.append(batch.mosfet_region_labels.cpu())
            all_is_nmos.append(mi[:, 6].bool().cpu())
            all_device_id.append((torch.arange(mi.shape[0]) % mosfets_per_graph).cpu())

    # Concatenate
    sens = torch.cat(all_sens_emb).numpy() if all_sens_emb else None
    state = torch.cat(all_state_emb).numpy()
    backbone = torch.cat(all_backbone_emb).numpy()
    gm_gt = torch.cat(all_gm_gt).numpy()
    gds_gt = torch.cat(all_gds_gt).numpy()
    region = torch.cat(all_region).numpy() if all_region else None
    is_nmos = torch.cat(all_is_nmos).numpy()
    device_id = torch.cat(all_device_id).numpy()

    gm_pred = torch.cat(all_gm_pred).numpy() if all_gm_pred else None
    gds_pred = torch.cat(all_gds_pred).numpy() if all_gds_pred else None
    drain_cp = torch.cat(all_drain_curr_pred).numpy() if all_drain_curr_pred else None
    source_cp = torch.cat(all_source_curr_pred).numpy() if all_source_curr_pred else None
    drain_cg = torch.cat(all_drain_curr_gt).numpy() if all_drain_curr_gt else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collected {len(gm_gt)} MOSFET drain embeddings")

    # ===== 1. Linear Probe: embedding → gm/gds =====
    print("\n--- Linear Probe (Ridge regression) ---")
    n = len(gm_gt)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    emb_pairs = [('backbone', backbone), ('state', state)]
    if sens is not None:
        emb_pairs.append(('sensitivity', sens))
    for emb_name, emb_data in emb_pairs:
        for target_name, target in [('gm', gm_gt), ('gds', gds_gt)]:
            reg = Ridge(alpha=1.0)
            reg.fit(emb_data[train_idx], target[train_idx])
            r2 = reg.score(emb_data[test_idx], target[test_idx])
            print(f"  {emb_name:15s} → {target_name}: R² = {r2:.4f}")

    # ===== 2. Per-device gm/gds error analysis =====
    if gm_pred is not None and gds_pred is not None:
        print("\n--- Per-Device Error (log10 MAE) ---")
        # Compute SS normalization to convert to log10 MAE
        n_devices = int(device_id.max()) + 1
        print(f"  {'Device':>8s}  {'gm_MAE':>8s}  {'gds_MAE':>8s}  {'Region':>20s}")
        for d in range(n_devices):
            mask_d = device_id == d
            gm_err = np.abs(gm_pred[mask_d] - gm_gt[mask_d]).mean()
            gds_err = np.abs(gds_pred[mask_d] - gds_gt[mask_d]).mean()
            if region is not None:
                r = region[mask_d]
                sat_frac = (r == 2).mean() * 100
                tri_frac = (r == 1).mean() * 100
                cut_frac = (r == 0).mean() * 100
                region_str = f"sat={sat_frac:.0f}% tri={tri_frac:.0f}% cut={cut_frac:.0f}%"
            else:
                region_str = ""
            nmos_str = "NMOS" if is_nmos[mask_d].any() else "PMOS"
            print(f"  M{d:02d} {nmos_str}  {gm_err:.4f}    {gds_err:.4f}    {region_str}")

    # ===== 3. Current consistency per device =====
    if drain_cp is not None and source_cp is not None:
        print("\n--- Current Consistency (I_drain vs I_source) per Device ---")
        d_real = 10**(drain_cp * curr_std + curr_mean)
        s_real = 10**(source_cp * curr_std + curr_mean)
        rel = np.abs(d_real - s_real) / (np.maximum(d_real, s_real) + 1e-15)
        print(f"  {'Device':>8s}  {'mean_diff':>10s}  {'median_diff':>12s}  {'<5%':>6s}")
        for d in range(n_devices):
            mask_d = device_id == d
            r = rel[mask_d]
            nmos_str = "NMOS" if is_nmos[mask_d].any() else "PMOS"
            print(f"  M{d:02d} {nmos_str}  {r.mean()*100:8.2f}%   {np.median(r)*100:8.2f}%    {(r<0.05).mean()*100:5.1f}%")

    # ===== 4. t-SNE of drain embeddings =====
    tsne_emb = sens if sens is not None else state
    tsne_label = "Sensitivity Tower" if sens is not None else "State Tower"
    print(f"\n--- Computing t-SNE ({tsne_label}, 2000 samples) ---")
    subsample = min(2000, len(tsne_emb))
    idx_sub = np.random.choice(len(tsne_emb), subsample, replace=False)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_2d = tsne.fit_transform(tsne_emb[idx_sub])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f't-SNE of {tsne_label} Drain Embeddings — {model_name}', fontsize=14)

    # By device
    ax = axes[0, 0]
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=device_id[idx_sub], cmap='tab20', s=5, alpha=0.6)
    ax.set_title('Colored by Device (M00-M23)')

    # By region
    ax = axes[0, 1]
    region_names = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation'}
    colors = {0: 'red', 1: 'green', 2: 'blue'}
    if region is not None:
        for r_val, r_name in region_names.items():
            mask_r = region[idx_sub] == r_val
            ax.scatter(emb_2d[mask_r, 0], emb_2d[mask_r, 1], c=colors[r_val], s=5, alpha=0.6, label=r_name)
        ax.legend()
    ax.set_title('Colored by Operating Region')

    # By NMOS/PMOS
    ax = axes[0, 2]
    ax.scatter(emb_2d[is_nmos[idx_sub], 0], emb_2d[is_nmos[idx_sub], 1], c='blue', s=5, alpha=0.6, label='NMOS')
    ax.scatter(emb_2d[~is_nmos[idx_sub], 0], emb_2d[~is_nmos[idx_sub], 1], c='red', s=5, alpha=0.6, label='PMOS')
    ax.legend()
    ax.set_title('Colored by Device Type')

    # By gm
    ax = axes[1, 0]
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=gm_gt[idx_sub], cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(scatter, ax=ax)
    ax.set_title('Colored by log10(gm)')

    # By gds
    ax = axes[1, 1]
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=gds_gt[idx_sub], cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(scatter, ax=ax)
    ax.set_title('Colored by log10(gds)')

    # By gds error
    if gds_pred is not None:
        ax = axes[1, 2]
        gds_err = np.abs(gds_pred[idx_sub] - gds_gt[idx_sub])
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=gds_err, cmap='hot_r', s=5, alpha=0.6, vmax=np.percentile(gds_err, 95))
        plt.colorbar(scatter, ax=ax)
        ax.set_title('Colored by gds Error (|pred-gt|)')

    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_tsne.png', dpi=150)
    print(f"  Saved {output_dir / 'sensitivity_tsne.png'}")

    # ===== 5. Per-device t-SNE (selected devices) =====
    print("\n--- Per-device t-SNE ---")
    # Pick 4 devices with different characteristics
    devices_to_plot = [0, 5, 11, 23]  # mix of NMOS/PMOS, different roles
    fig, axes = plt.subplots(1, len(devices_to_plot), figsize=(5*len(devices_to_plot), 5))

    for i, dev in enumerate(devices_to_plot):
        mask_d = device_id == dev
        dev_emb = tsne_emb[mask_d]
        dev_region = region[mask_d] if region is not None else None

        if len(dev_emb) > 500:
            sub = np.random.choice(len(dev_emb), 500, replace=False)
            dev_emb = dev_emb[sub]
            dev_region = dev_region[sub] if dev_region is not None else None

        tsne_d = TSNE(n_components=2, perplexity=min(30, len(dev_emb)//4), random_state=42)
        emb_d = tsne_d.fit_transform(dev_emb)

        ax = axes[i]
        nmos_str = "NMOS" if is_nmos[mask_d].any() else "PMOS"
        if dev_region is not None:
            for r_val, r_name in region_names.items():
                m = dev_region == r_val
                frac = m.mean() * 100
                ax.scatter(emb_d[m, 0], emb_d[m, 1], c=colors[r_val], s=10, alpha=0.6,
                          label=f'{r_name} ({frac:.0f}%)')
            ax.legend(fontsize=8)
        ax.set_title(f'M{dev:02d} ({nmos_str})')

    plt.suptitle(f'Per-Device t-SNE — {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_device_tsne.png', dpi=150)
    print(f"  Saved {output_dir / 'per_device_tsne.png'}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':
    base = Path('datasets/opamp_3stage_fan_smc_v8_5k_nofil/experiments/tower')

    analyze_model(
        "w512 (no KCL)",
        base / "fp32_bulk_voltctx_w512/best_model.pt",
        base / "fp32_bulk_voltctx_w512/analysis"
    )
    analyze_model(
        "w512 + KCL=1.0",
        base / "w512_kcl_1_0/best_model.pt",
        base / "w512_kcl_1_0/analysis"
    )
