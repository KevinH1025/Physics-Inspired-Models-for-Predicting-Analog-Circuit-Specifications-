#!/usr/bin/env python3
"""
Probe learned embeddings of a trained GNN for interpretability analysis.

Analyses:
  1. Linear probe: drain embeddings → gm/gds (does backbone encode device physics?)
  2. t-SNE: drain embeddings colored by gm, operating region, NMOS/PMOS
  3. Layer-wise probing: gm/gds R² vs network depth (when does physics emerge?)
  5. Graph-level t-SNE: mean-pooled node embeddings colored by I_BIAS, DC gain, sat fraction
  6. Per-device t-SNE: single device across samples, colored by operating region

Usage:
    python scripts/probe_embeddings.py --name baseline_sat_only
    python scripts/probe_embeddings.py --checkpoint datasets/opamp_5k_ss_v1/best_model.pt
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import (
    load_prebatched_variant,
    add_ss_node_targets,
    compute_ss_normalization,
)
from src.gnn.architectures.tower_genconv import TowerGENConv


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(model, batch, device):
    """Extract embeddings at multiple levels without modifying the model.

    Supports both DeepGENConv and TowerGENConv architectures.

    Returns dict with:
        final_repr: [num_nodes, mlp_input_dim] — what heads see (with skip)
        layer_outputs: list of [num_nodes, hidden_dim] per GNN layer
    """
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        x_in = model._get_input_features(batch)
        x = model.input_linear(x_in)

        if isinstance(model, TowerGENConv):
            # --- Tower architecture path ---
            batch_vec = batch.batch if hasattr(batch, 'batch') else None
            num_graphs = model._get_num_graphs(batch, batch_vec) if batch_vec is not None else 1
            edge_attr = getattr(batch, 'edge_attr', None) if model.use_edge_features else None

            vn_emb = None
            if model.virtual_node is not None:
                vn_emb = model.virtual_node.init_embedding(num_graphs)

            # Backbone layers (with VN)
            backbone_outputs, vn_emb = model._run_layers(
                model.backbone, x, batch.edge_index, edge_attr,
                batch=batch_vec, num_graphs=num_graphs,
                use_vn=True, vn_emb=vn_emb,
            )
            backbone_hidden = model.backbone_jk(backbone_outputs)
            backbone_hidden = model.backbone[0].act(model.backbone[0].norm(backbone_hidden))

            # State tower layers
            state_outputs, _ = model._run_layers(
                model.state_tower, backbone_hidden, batch.edge_index, edge_attr,
            )
            state_repr = model._finalize_repr(
                model.state_jk, state_outputs, model.state_tower[0], x_in,
            )

            # Sensitivity tower layers (if exists)
            if model.has_sensitivity_tower:
                sens_input = backbone_hidden
                if model.state_conditioned_sens:
                    state_hidden = model.state_jk(state_outputs)
                    state_hidden = model.state_tower[0].act(model.state_tower[0].norm(state_hidden))
                    state_for_sens = state_hidden.detach() if model.detach_state_for_sens else state_hidden
                    sens_input = model.state_sens_proj(torch.cat([backbone_hidden, state_for_sens], dim=-1))

                sens_outputs, _ = model._run_layers(
                    model.sensitivity_tower, sens_input, batch.edge_index, edge_attr,
                )
                sens_repr = model._finalize_repr(
                    model.sensitivity_jk, sens_outputs, model.sensitivity_tower[0], x_in,
                )
            else:
                sens_outputs = []
                sens_repr = None

            # Y-branch: extract gm/gds branch representations if they exist
            gm_repr = None
            gds_repr = None
            gm_branch_outputs = []
            gds_branch_outputs = []
            if model.has_sensitivity_tower and getattr(model, 'gm_branch', None) is not None:
                branch_input = sens_outputs[-1]  # last hidden state from shared tower

                gm_branch_outputs, _ = model._run_layers(
                    model.gm_branch, branch_input, batch.edge_index, edge_attr,
                )
                gm_repr = model._finalize_repr(
                    model.gm_branch_jk, gm_branch_outputs, model.gm_branch[0], x_in,
                )

                gds_branch_outputs, _ = model._run_layers(
                    model.gds_branch, branch_input, batch.edge_index, edge_attr,
                )
                gds_repr = model._finalize_repr(
                    model.gds_branch_jk, gds_branch_outputs, model.gds_branch[0], x_in,
                )

            # Backbone layer outputs (sequential, shared)
            layer_outputs = list(backbone_outputs)
            layer_labels = [f'BB{i}' for i in range(len(backbone_outputs))]

            # State and sensitivity towers are PARALLEL from backbone — store separately
            state_layer_outputs = state_outputs[1:]  # skip input (= backbone_hidden)
            sens_layer_outputs = sens_outputs[1:] if sens_outputs else []

            # Y-branch: store branch layer outputs separately (parallel from sensitivity)
            gm_branch_layer_outputs = []
            gds_branch_layer_outputs = []
            if gm_branch_outputs:
                gm_branch_layer_outputs = gm_branch_outputs[1:]   # skip input (= sens_outputs[-1])
                gds_branch_layer_outputs = gds_branch_outputs[1:]

            # final_repr = sensitivity tower output if available, else state tower output
            # For Y-branch: gm_repr/gds_repr are what the heads actually see
            final_repr = sens_repr if sens_repr is not None else state_repr

        else:
            # --- Original DeepGENConv path ---
            batch_idx = None
            num_graphs = None
            if model.virtual_node is not None:
                batch_idx = batch.batch if hasattr(batch, 'batch') else torch.zeros(
                    x.size(0), dtype=torch.long, device=x.device
                )
                num_graphs = model._get_num_graphs(batch, batch_idx)

            layer_outputs, _ = model._message_passing(
                x, batch.edge_index, batch_idx, num_graphs
            )

            backbone = model._apply_jk(layer_outputs)
            backbone = model.layers[0].act(model.layers[0].norm(backbone))
            if model.skip_connection:
                final_repr = torch.cat([backbone, x_in], dim=-1)
            else:
                final_repr = backbone

    result = {
        'final_repr': final_repr,       # [N, mlp_input_dim]
        'layer_outputs': layer_outputs,  # list of [N, hidden_dim] — backbone only for tower
    }
    if isinstance(model, TowerGENConv):
        result['layer_labels'] = layer_labels
        result['state_layers'] = state_layer_outputs      # parallel from backbone
        result['sens_layers'] = sens_layer_outputs        # parallel from backbone
        # Y-branch specific: branch representations (what gm/gds heads actually see)
        if gm_repr is not None:
            result['gm_final_repr'] = gm_repr
            result['gds_final_repr'] = gds_repr
            result['gm_branch_layers'] = gm_branch_layer_outputs
            result['gds_branch_layers'] = gds_branch_layer_outputs
    return result


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_drain_data(model, batches, device):
    """Collect drain embeddings + targets from all batches."""
    drain_final = []
    drain_per_layer = None           # backbone layers
    drain_state_layers = None        # state tower layers (parallel)
    drain_sens_layers = None         # sensitivity tower layers (parallel)
    drain_gm_repr = []               # Y-branch: gm branch output at drain
    drain_gds_repr = []              # Y-branch: gds branch output at drain
    drain_gm_branch_layers = None
    drain_gds_branch_layers = None
    drain_gm = []
    drain_gds = []
    drain_is_nmos = []
    drain_region = []
    drain_device_id = []
    layer_labels = None

    for batch in batches:
        batch = batch.to(device)
        add_ss_node_targets([batch])

        emb = extract_embeddings(model, batch, device)
        mask = batch.mosfet_drain_mask
        if not mask.any():
            continue

        drain_final.append(emb['final_repr'][mask].cpu())
        drain_gm.append(batch.node_log_gm[mask].cpu())

        # Capture layer labels (once)
        if layer_labels is None and 'layer_labels' in emb:
            layer_labels = emb['layer_labels']

        # Y-branch: collect branch-specific drain embeddings
        if 'gm_final_repr' in emb:
            drain_gm_repr.append(emb['gm_final_repr'][mask].cpu())
            drain_gds_repr.append(emb['gds_final_repr'][mask].cpu())

            if drain_gm_branch_layers is None:
                drain_gm_branch_layers = [[] for _ in emb['gm_branch_layers']]
                drain_gds_branch_layers = [[] for _ in emb['gds_branch_layers']]
            for i, lo in enumerate(emb['gm_branch_layers']):
                drain_gm_branch_layers[i].append(lo[mask].cpu())
            for i, lo in enumerate(emb['gds_branch_layers']):
                drain_gds_branch_layers[i].append(lo[mask].cpu())
        drain_gds.append(batch.node_log_gds[mask].cpu())

        # Per-layer drain embeddings (backbone only)
        if drain_per_layer is None:
            drain_per_layer = [[] for _ in range(len(emb['layer_outputs']))]
        for i, lo in enumerate(emb['layer_outputs']):
            drain_per_layer[i].append(lo[mask].cpu())

        # State/sensitivity tower layers (parallel branches, tower arch only)
        if 'state_layers' in emb:
            if drain_state_layers is None:
                drain_state_layers = [[] for _ in emb['state_layers']]
                drain_sens_layers = [[] for _ in emb['sens_layers']] if emb['sens_layers'] else None
            for i, lo in enumerate(emb['state_layers']):
                drain_state_layers[i].append(lo[mask].cpu())
            if drain_sens_layers is not None:
                for i, lo in enumerate(emb['sens_layers']):
                    drain_sens_layers[i].append(lo[mask].cpu())

        # NMOS/PMOS and region — scatter from per-MOSFET to drain nodes
        mosfet_info = batch.mosfet_info
        region_labels = batch.mosfet_region_labels if hasattr(batch, 'mosfet_region_labels') else None
        ptr = batch.ptr
        num_nodes = batch.x.shape[0]
        num_mosfets = mosfet_info.shape[0]
        num_graphs = ptr.shape[0] - 1
        mosfets_per_graph = num_mosfets // num_graphs

        mosfet_graph_idx = torch.arange(num_mosfets, device=device) // mosfets_per_graph
        node_offsets = ptr[mosfet_graph_idx]
        drain_idx_global = mosfet_info[:, 1].long() + node_offsets

        # is_nmos per node
        node_is_nmos = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_is_nmos[drain_idx_global] = mosfet_info[:, 6].bool()

        # region per node
        node_region = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        if region_labels is not None:
            node_region[drain_idx_global] = region_labels.to(device)

        # device identity (M1=0, M2=1, ..., M8=7) per drain node
        mosfet_local_id = torch.arange(num_mosfets, device=device) % mosfets_per_graph
        node_device_id = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_device_id[drain_idx_global] = mosfet_local_id

        drain_is_nmos.append(node_is_nmos[mask].cpu())
        drain_region.append(node_region[mask].cpu())
        drain_device_id.append(node_device_id[mask].cpu())

    result = {
        'final': torch.cat(drain_final),
        'gm': torch.cat(drain_gm),
        'gds': torch.cat(drain_gds),
        'is_nmos': torch.cat(drain_is_nmos),
        'region': torch.cat(drain_region),
        'device_id': torch.cat(drain_device_id),
    }
    if drain_per_layer is not None:
        result['per_layer'] = [torch.cat(l) for l in drain_per_layer]
    if layer_labels is not None:
        result['layer_labels'] = layer_labels
    # Tower parallel branches: state and sensitivity tower layers
    if drain_state_layers is not None:
        result['state_layers'] = [torch.cat(l) for l in drain_state_layers]
    if drain_sens_layers is not None:
        result['sens_layers'] = [torch.cat(l) for l in drain_sens_layers]
    # Y-branch: branch-specific drain embeddings
    if drain_gm_repr:
        result['gm_final'] = torch.cat(drain_gm_repr)
        result['gds_final'] = torch.cat(drain_gds_repr)
    if drain_gm_branch_layers is not None:
        result['gm_branch_layers'] = [torch.cat(l) for l in drain_gm_branch_layers]
        result['gds_branch_layers'] = [torch.cat(l) for l in drain_gds_branch_layers]
    return result


# ---------------------------------------------------------------------------
# Data collection: graph-level
# ---------------------------------------------------------------------------

def collect_graph_level_data(model, batches, device):
    """Collect mean-pooled node embeddings + graph-level metadata."""
    graph_embs = []
    graph_ibias = []
    graph_dc_gain = []
    graph_sat_frac = []
    graph_triode_frac = []
    graph_cutoff_frac = []

    for batch in batches:
        batch = batch.to(device)
        emb = extract_embeddings(model, batch, device)
        final = emb['final_repr']  # [num_nodes, dim]

        ptr = batch.ptr
        num_graphs = ptr.shape[0] - 1
        num_mosfets = batch.mosfet_info.shape[0]
        mosfets_per_graph = num_mosfets // num_graphs

        for g_idx in range(num_graphs):
            start, end = ptr[g_idx].item(), ptr[g_idx + 1].item()

            # Mean-pool all node embeddings for this graph
            graph_embs.append(final[start:end].mean(dim=0).cpu())

            # I_BIAS: feat[8] is same for all nodes in a graph
            graph_ibias.append(batch.x[start, 8].item())

            # AC DC gain (if available)
            if hasattr(batch, 'ac_dc_gain'):
                gain = batch.ac_dc_gain
                if gain.dim() == 0:
                    graph_dc_gain.append(gain.item())
                else:
                    graph_dc_gain.append(gain[g_idx].item())

            # Region fractions from mosfet_region_labels
            if hasattr(batch, 'mosfet_region_labels'):
                m_start = g_idx * mosfets_per_graph
                m_end = m_start + mosfets_per_graph
                regions = batch.mosfet_region_labels[m_start:m_end]
                n_dev = regions.shape[0]
                graph_sat_frac.append((regions == 2).sum().item() / n_dev)
                graph_triode_frac.append((regions == 1).sum().item() / n_dev)
                graph_cutoff_frac.append((regions == 0).sum().item() / n_dev)

    result = {
        'emb': torch.stack(graph_embs),
        'ibias': np.array(graph_ibias),
    }
    if graph_dc_gain:
        result['dc_gain'] = np.array(graph_dc_gain)
    if graph_sat_frac:
        result['sat_frac'] = np.array(graph_sat_frac)
        result['triode_frac'] = np.array(graph_triode_frac)
        result['cutoff_frac'] = np.array(graph_cutoff_frac)
    return result


# ---------------------------------------------------------------------------
# Analysis 1: Linear probe for gm/gds
# ---------------------------------------------------------------------------

def run_linear_probe(train_data, val_data, output_dir):
    """Fit Ridge probes from drain embeddings to gm/gds, plot results."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    results = {}

    has_ybranch = 'gm_final' in train_data

    for col, target_name in enumerate(['gm', 'gds']):
        # Y-branch: use branch-specific embeddings (what the heads actually see)
        embed_key = f'{target_name}_final' if has_ybranch else 'final'
        X_train = train_data[embed_key].numpy()
        y_train = train_data[target_name].numpy()
        X_val = val_data[embed_key].numpy()
        y_val = val_data[target_name].numpy()

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_val)

        r2 = r2_score(y_val, y_pred)
        mae = np.mean(np.abs(y_val - y_pred))
        results[target_name] = {'r2': r2, 'mae': mae}

        # Scatter plot: predicted vs actual
        ax = axes[0, col]
        ax.scatter(y_val, y_pred, s=2, alpha=0.3, c='steelblue')
        lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
        ax.set_xlabel(f'Actual log10({target_name})')
        ax.set_ylabel(f'Predicted log10({target_name})')
        ax.set_title(f'{target_name} Linear Probe  (R²={r2:.3f}, MAE={mae:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Error histogram
        ax = axes[1, col]
        errors = np.abs(y_val - y_pred)
        ax.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(np.log10(1.5), color='green', linestyle='--', linewidth=1.5, label=f'1.5x ({np.log10(1.5):.2f})')
        ax.axvline(np.log10(2.0), color='orange', linestyle='--', linewidth=1.5, label=f'2x ({np.log10(2.0):.2f})')
        within_1_5x = 100 * np.mean(errors < np.log10(1.5))
        within_2x = 100 * np.mean(errors < np.log10(2.0))
        ax.set_xlabel(f'|error| in log10({target_name})')
        ax.set_ylabel('Count')
        ax.set_title(f'{target_name} Error Distribution  (1.5x={within_1_5x:.0f}%, 2x={within_2x:.0f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    title = 'Linear Probe: Branch Embeddings → gm/gds (Y-branch)' if has_ybranch else 'Linear Probe: Drain Embeddings → gm/gds'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_linear.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'probe_linear.png'}")
    return results


# ---------------------------------------------------------------------------
# Analysis 2: t-SNE visualization
# ---------------------------------------------------------------------------

def run_tsne(val_data, output_dir, max_points=5000, has_ss=True):
    """t-SNE of drain embeddings colored by device identity, operating region, NMOS/PMOS, gm, gds."""
    from sklearn.manifold import TSNE

    X = val_data['final'].numpy()
    is_nmos = val_data['is_nmos'].numpy()
    region = val_data['region'].numpy()
    device_id = val_data['device_id'].numpy()
    gm = val_data['gm'].numpy()
    gds = val_data['gds'].numpy()

    # Subsample if too large
    n = len(X)
    if n > max_points:
        idx = np.random.default_rng(42).choice(n, max_points, replace=False)
        X, is_nmos, region, device_id = X[idx], is_nmos[idx], region[idx], device_id[idx]
        gm, gds = gm[idx], gds[idx]

    print(f"  Running t-SNE on {len(X)} drain embeddings ({X.shape[1]}-dim)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    n_rows = 2 if has_ss else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    # Panel 1: Device identity
    unique_ids = sorted(set(device_id[device_id >= 0]))
    n_devices = len(unique_ids)
    device_cmap = plt.colormaps.get_cmap('tab20' if n_devices > 10 else 'tab10').resampled(max(n_devices, 2))
    for i, d_id in enumerate(unique_ids):
        m = device_id == d_id
        if m.any():
            axes[0, 0].scatter(X_2d[m, 0], X_2d[m, 1], c=[device_cmap(i)],
                            s=4, alpha=0.5, label=f'M{d_id}')
    axes[0, 0].legend(markerscale=4, fontsize=6, ncol=max(1, n_devices // 6))
    axes[0, 0].set_title(f'Colored by Device ({n_devices} devices)')

    # Panel 2: Operating region
    region_names = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation', -1: 'Unknown'}
    region_colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', -1: '#95a5a6'}
    for r_val in sorted(set(region)):
        m = region == r_val
        if m.any():
            axes[0, 1].scatter(X_2d[m, 0], X_2d[m, 1], c=region_colors.get(r_val, 'gray'),
                            s=4, alpha=0.5, label=region_names.get(r_val, f'R={r_val}'))
    axes[0, 1].legend(markerscale=4)
    axes[0, 1].set_title('Colored by Operating Region')

    # Panel 3: NMOS vs PMOS
    for val, label, color in [(True, 'NMOS', '#3498db'), (False, 'PMOS', '#e74c3c')]:
        m = is_nmos == val
        if m.any():
            axes[0, 2].scatter(X_2d[m, 0], X_2d[m, 1], c=color, s=4, alpha=0.5, label=label)
    axes[0, 2].legend(markerscale=4)
    axes[0, 2].set_title('Colored by Device Type')

    if has_ss:
        # Panel 4: gm (continuous colormap)
        sc_gm = axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=gm, cmap='viridis',
                                    s=4, alpha=0.5)
        plt.colorbar(sc_gm, ax=axes[1, 0], label='log10(gm)')
        axes[1, 0].set_title('Colored by gm')

        # Panel 5: gds (continuous colormap)
        sc_gds = axes[1, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=gds, cmap='viridis',
                                     s=4, alpha=0.5)
        plt.colorbar(sc_gds, ax=axes[1, 1], label='log10(gds)')
        axes[1, 1].set_title('Colored by gds')

        # Panel 6: gm/gds ratio
        gm_over_gds = gm - gds  # in log space, subtraction = ratio
        sc_ratio = axes[1, 2].scatter(X_2d[:, 0], X_2d[:, 1], c=gm_over_gds, cmap='coolwarm',
                                       s=4, alpha=0.5)
        plt.colorbar(sc_ratio, ax=axes[1, 2], label='log10(gm/gds)')
        axes[1, 2].set_title('Colored by gm/gds ratio')

    for ax in axes.flat:
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.suptitle('t-SNE of Drain Node Embeddings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'probe_tsne.png'}")


# ---------------------------------------------------------------------------
# Analysis 3: Layer-wise probing
# ---------------------------------------------------------------------------

def _probe_layers(layers_train, layers_val, y_train, y_val):
    """Probe a list of layer embeddings, return list of R² values."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    r2s = []
    for i in range(len(layers_train)):
        probe = Ridge(alpha=1.0)
        probe.fit(layers_train[i].numpy(), y_train)
        r2s.append(r2_score(y_val, probe.predict(layers_val[i].numpy())))
    return r2s


def run_layerwise_probe(train_data, val_data, output_dir):
    """Fit Ridge probe at each GEN layer to track when physics emerges.

    For tower models: backbone is shared, then state/sensitivity towers branch
    in parallel. Plot reflects this parallel structure.
    For Y-branch: sensitivity further splits into gm/gds branches.
    """
    if 'per_layer' not in train_data or 'per_layer' not in val_data:
        print("  Skipping layer-wise probe (no per-layer data)")
        return None

    has_tower = 'state_layers' in train_data
    has_sens = 'sens_layers' in train_data
    has_ybranch = 'gm_branch_layers' in train_data
    layer_labels = train_data.get('layer_labels', [str(i) for i in range(len(train_data['per_layer']))])

    y_gm_tr = train_data['gm'].numpy()
    y_gm_va = val_data['gm'].numpy()
    y_gds_tr = train_data['gds'].numpy()
    y_gds_va = val_data['gds'].numpy()

    # --- Backbone layers (always present) ---
    bb_gm_r2 = _probe_layers(train_data['per_layer'], val_data['per_layer'], y_gm_tr, y_gm_va)
    bb_gds_r2 = _probe_layers(train_data['per_layer'], val_data['per_layer'], y_gds_tr, y_gds_va)
    n_bb = len(bb_gm_r2)

    # --- State tower layers (parallel from backbone, tower only) ---
    st_gm_r2, st_gds_r2 = [], []
    if has_tower:
        st_gm_r2 = _probe_layers(train_data['state_layers'], val_data['state_layers'], y_gm_tr, y_gm_va)
        st_gds_r2 = _probe_layers(train_data['state_layers'], val_data['state_layers'], y_gds_tr, y_gds_va)

    # --- Sensitivity tower layers (parallel from backbone, tower only) ---
    ss_gm_r2, ss_gds_r2 = [], []
    if has_sens:
        ss_gm_r2 = _probe_layers(train_data['sens_layers'], val_data['sens_layers'], y_gm_tr, y_gm_va)
        ss_gds_r2 = _probe_layers(train_data['sens_layers'], val_data['sens_layers'], y_gds_tr, y_gds_va)

    # --- Y-branch layers (parallel from sensitivity) ---
    gm_br_r2, gds_br_r2 = [], []
    if has_ybranch:
        gm_br_r2 = _probe_layers(train_data['gm_branch_layers'], val_data['gm_branch_layers'], y_gm_tr, y_gm_va)
        gds_br_r2 = _probe_layers(train_data['gds_branch_layers'], val_data['gds_branch_layers'], y_gds_tr, y_gds_va)

    # --- Build results dict ---
    results = {
        'bb_labels': layer_labels, 'bb_gm_r2': bb_gm_r2, 'bb_gds_r2': bb_gds_r2,
        'st_gm_r2': st_gm_r2, 'st_gds_r2': st_gds_r2,
        'ss_gm_r2': ss_gm_r2, 'ss_gds_r2': ss_gds_r2,
    }
    if has_ybranch:
        results['gm_branch_r2'] = gm_br_r2
        results['gds_branch_r2'] = gds_br_r2

    # --- Plot ---
    if has_tower:
        # Tower architecture: backbone on left, parallel towers on right
        fig, (ax_bb, ax_towers) = plt.subplots(1, 2, figsize=(16, 6),
                                                gridspec_kw={'width_ratios': [n_bb, max(len(st_gm_r2), len(ss_gm_r2)) + (len(gm_br_r2) if has_ybranch else 0) + 1]})

        # Left panel: backbone
        x_bb = list(range(n_bb))
        ax_bb.plot(x_bb, bb_gm_r2, 'o-', color='#3498db', markersize=6, linewidth=2, label='gm R²')
        ax_bb.plot(x_bb, bb_gds_r2, 's-', color='#e74c3c', markersize=6, linewidth=2, label='gds R²')
        ax_bb.set_xticks(x_bb)
        ax_bb.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
        ax_bb.set_xlabel('Backbone (shared)', fontsize=12)
        ax_bb.set_ylabel('Linear Probe R²', fontsize=12)
        ax_bb.set_title('Backbone Layers', fontsize=13, fontweight='bold')
        ax_bb.legend(fontsize=10)
        ax_bb.grid(True, alpha=0.3)
        ax_bb.set_ylim([-0.05, 1.05])

        # Right panel: parallel towers
        # State tower
        n_st = len(st_gm_r2)
        n_ss = len(ss_gm_r2)
        x_st = list(range(n_st))
        st_labels = [f'ST{i}' for i in range(n_st)]
        ax_towers.plot(x_st, st_gm_r2, 'o-', color='#3498db', markersize=8, linewidth=2, label='State → gm R²')
        ax_towers.plot(x_st, st_gds_r2, 's-', color='#e74c3c', markersize=8, linewidth=2, label='State → gds R²')

        # Sensitivity tower (offset on x-axis to show parallel)
        if n_ss > 0:
            x_ss = [i + n_st + 1 for i in range(n_ss)]  # gap of 1 to separate
            ss_labels = [f'SS{i}' for i in range(n_ss)]
            ax_towers.plot(x_ss, ss_gm_r2, '^', color='#3498db', markersize=8, linewidth=2, linestyle='--', label='Sens → gm R²')
            ax_towers.plot(x_ss, ss_gds_r2, 'D', color='#e74c3c', markersize=8, linewidth=2, linestyle='--', label='Sens → gds R²')
        else:
            x_ss = []
            ss_labels = []

        # Y-branch (offset further)
        br_labels = []
        if has_ybranch and (gm_br_r2 or gds_br_r2):
            x_br_start = (x_ss[-1] if x_ss else x_st[-1]) + 2
            n_br = max(len(gm_br_r2), len(gds_br_r2))
            x_br = [x_br_start + i for i in range(n_br)]
            br_labels = [f'BR{i}' for i in range(n_br)]
            if gm_br_r2:
                ax_towers.plot(x_br[:len(gm_br_r2)], gm_br_r2, 'v', color='#3498db', markersize=8, linewidth=2, linestyle=':', label='gm branch → gm R²')
            if gds_br_r2:
                ax_towers.plot(x_br[:len(gds_br_r2)], gds_br_r2, 'P', color='#e74c3c', markersize=8, linewidth=2, linestyle=':', label='gds branch → gds R²')

        # Separator between tower groups
        if x_ss:
            sep1 = (x_st[-1] + x_ss[0]) / 2
            ax_towers.axvline(sep1, color='gray', linestyle=':', alpha=0.5)
        if has_ybranch and br_labels:
            sep2 = (x_ss[-1] + x_br[0]) / 2 if x_ss else (x_st[-1] + x_br[0]) / 2
            ax_towers.axvline(sep2, color='gray', linestyle=':', alpha=0.5)

        # Backbone final R² as baseline
        ax_towers.axhline(bb_gm_r2[-1], color='#3498db', linestyle='-.', alpha=0.3, linewidth=1)
        ax_towers.axhline(bb_gds_r2[-1], color='#e74c3c', linestyle='-.', alpha=0.3, linewidth=1)

        all_x = x_st + x_ss + (x_br if has_ybranch and br_labels else [])
        all_labels = st_labels + ss_labels + br_labels
        ax_towers.set_xticks(all_x)
        ax_towers.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
        ax_towers.set_xlabel('Tower Layers (State ‖ Sensitivity' + (' ‖ Branches)' if has_ybranch else ')'), fontsize=12)
        ax_towers.set_ylabel('Linear Probe R²', fontsize=12)
        ax_towers.set_title('Parallel Tower Layers', fontsize=13, fontweight='bold')
        ax_towers.legend(fontsize=8, ncol=2)
        ax_towers.grid(True, alpha=0.3)
        ax_towers.set_ylim([-0.05, 1.05])

    else:
        # DeepGENConv: single sequential plot
        fig, ax_bb = plt.subplots(figsize=(10, 5))
        x_bb = list(range(n_bb))
        ax_bb.plot(x_bb, bb_gm_r2, 'o-', color='#3498db', markersize=6, linewidth=2, label='gm R²')
        ax_bb.plot(x_bb, bb_gds_r2, 's-', color='#e74c3c', markersize=6, linewidth=2, label='gds R²')
        ax_bb.set_xlabel('Layer (0 = input projection)', fontsize=12)
        ax_bb.set_ylabel('Linear Probe R²', fontsize=12)
        ax_bb.set_title('Layer-wise Probing: When Does Physics Knowledge Emerge?', fontsize=13, fontweight='bold')
        ax_bb.legend(fontsize=11)
        ax_bb.grid(True, alpha=0.3)
        ax_bb.set_ylim([-0.05, 1.05])
        ax_bb.set_xticks(x_bb)
        if layer_labels:
            ax_bb.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)

    # Annotate peaks
    all_gm = bb_gm_r2 + st_gm_r2 + ss_gm_r2 + (gm_br_r2 if has_ybranch else [])
    all_gds = bb_gds_r2 + st_gds_r2 + ss_gds_r2 + (gds_br_r2 if has_ybranch else [])
    all_names = list(layer_labels) + [f'ST{i}' for i in range(len(st_gm_r2))] + [f'SS{i}' for i in range(len(ss_gm_r2))]
    if has_ybranch:
        all_names += [f'BR{i}' for i in range(max(len(gm_br_r2), len(gds_br_r2)))]

    gm_peak_name = all_names[np.argmax(all_gm)]
    gds_peak_name = all_names[np.argmax(all_gds)]
    results['gm_peak'] = gm_peak_name
    results['gds_peak'] = gds_peak_name
    results['gm_best'] = max(all_gm)
    results['gds_best'] = max(all_gds)

    plt.tight_layout()
    plt.savefig(output_dir / 'probe_layerwise.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'probe_layerwise.png'}")
    return results


# ---------------------------------------------------------------------------
# Analysis 4: PCA visualization
# ---------------------------------------------------------------------------

def run_pca(val_data, output_dir, max_points=5000):
    """PCA of drain embeddings: scree plot + PC1/PC2 colored by device identity, region, NMOS/PMOS."""
    from sklearn.decomposition import PCA

    X = val_data['final'].numpy()
    is_nmos = val_data['is_nmos'].numpy()
    region = val_data['region'].numpy()
    device_id = val_data['device_id'].numpy()

    # Subsample if too large
    n = len(X)
    if n > max_points:
        idx = np.random.default_rng(42).choice(n, max_points, replace=False)
        X, is_nmos, region, device_id = X[idx], is_nmos[idx], region[idx], device_id[idx]

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    # --- Figure 1: PC1 vs PC2 colored 3 ways ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Device identity
    unique_ids = sorted(set(device_id[device_id >= 0]))
    n_devices = len(unique_ids)
    device_cmap = plt.colormaps.get_cmap('tab20' if n_devices > 10 else 'tab10').resampled(max(n_devices, 2))
    for i, d_id in enumerate(unique_ids):
        m = device_id == d_id
        if m.any():
            axes[0].scatter(X_pca[m, 0], X_pca[m, 1], c=[device_cmap(i)],
                            s=4, alpha=0.5, label=f'M{d_id}')
    axes[0].legend(markerscale=4, fontsize=6, ncol=max(1, n_devices // 6))
    axes[0].set_title(f'Colored by Device ({n_devices} devices)')

    # Panel 2: Operating region
    region_names = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation', -1: 'Unknown'}
    region_colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', -1: '#95a5a6'}
    for r_val in sorted(set(region)):
        m = region == r_val
        if m.any():
            axes[1].scatter(X_pca[m, 0], X_pca[m, 1], c=region_colors.get(r_val, 'gray'),
                            s=4, alpha=0.5, label=region_names.get(r_val, f'R={r_val}'))
    axes[1].legend(markerscale=4)
    axes[1].set_title('Colored by Operating Region')

    # Panel 3: NMOS vs PMOS
    for val, label, color in [(True, 'NMOS', '#3498db'), (False, 'PMOS', '#e74c3c')]:
        m = is_nmos == val
        if m.any():
            axes[2].scatter(X_pca[m, 0], X_pca[m, 1], c=color, s=4, alpha=0.5, label=label)
    axes[2].legend(markerscale=4)
    axes[2].set_title('Colored by Device Type')

    for ax in axes:
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('PCA of Drain Node Embeddings (PC1 vs PC2)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'probe_pca.png'}")

    # --- Figure 2: Scree plot ---
    fig, ax1 = plt.subplots(figsize=(8, 5))

    n_comp = len(pca.explained_variance_ratio_)
    cumvar = pca.explained_variance_ratio_.cumsum() * 100
    ax1.bar(range(1, n_comp + 1), pca.explained_variance_ratio_ * 100,
            color='steelblue', alpha=0.8, label='Individual')
    ax1.plot(range(1, n_comp + 1), cumvar, 'ro-', markersize=5, label='Cumulative')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('PCA Scree Plot — Drain Node Embeddings', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, n_comp + 1))

    plt.tight_layout()
    plt.savefig(output_dir / 'probe_pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'probe_pca_analysis.png'}")

    return {
        'explained_variance': pca.explained_variance_ratio_,
    }


# ---------------------------------------------------------------------------
# Analysis 5: Graph-level t-SNE
# ---------------------------------------------------------------------------

def run_graph_tsne(graph_data, output_dir, max_points=3000):
    """t-SNE of mean-pooled graph embeddings colored by I_BIAS, DC gain, saturation fraction."""
    from sklearn.manifold import TSNE

    X = graph_data['emb'].numpy()
    ibias = graph_data['ibias']
    dc_gain = graph_data.get('dc_gain')
    sat_frac = graph_data.get('sat_frac')
    triode_frac = graph_data.get('triode_frac')

    n = len(X)
    if n > max_points:
        idx = np.random.default_rng(42).choice(n, max_points, replace=False)
        X = X[idx]
        ibias = ibias[idx]
        if dc_gain is not None:
            dc_gain = dc_gain[idx]
        if sat_frac is not None:
            sat_frac = sat_frac[idx]
            triode_frac = triode_frac[idx]

    print(f"  Running graph-level t-SNE on {len(X)} circuits ({X.shape[1]}-dim)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    n_panels = 2 + (1 if dc_gain is not None else 0) + (1 if sat_frac is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    panel = 0

    # Panel: I_BIAS
    sc = axes[panel].scatter(X_2d[:, 0], X_2d[:, 1], c=ibias, cmap='viridis', s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[panel], label='log10(I_BIAS) + 5')
    axes[panel].set_title('Colored by I_BIAS')
    panel += 1

    # Panel: Saturation fraction
    if sat_frac is not None:
        sc = axes[panel].scatter(X_2d[:, 0], X_2d[:, 1], c=sat_frac, cmap='RdYlGn',
                                  s=8, alpha=0.6, vmin=0, vmax=1)
        plt.colorbar(sc, ax=axes[panel], label='Fraction in saturation')
        axes[panel].set_title('Colored by Saturation Fraction')
        panel += 1

    # Panel: DC gain
    if dc_gain is not None:
        # Clip extreme gains for visualization
        gain_clipped = np.clip(dc_gain, -20, 150)
        sc = axes[panel].scatter(X_2d[:, 0], X_2d[:, 1], c=gain_clipped, cmap='coolwarm',
                                  s=8, alpha=0.6)
        plt.colorbar(sc, ax=axes[panel], label='DC Gain (dB)')
        axes[panel].set_title('Colored by DC Gain')
        panel += 1

    # Panel: Triode fraction
    if triode_frac is not None:
        sc = axes[panel].scatter(X_2d[:, 0], X_2d[:, 1], c=triode_frac, cmap='YlOrRd',
                                  s=8, alpha=0.6, vmin=0, vmax=0.5)
        plt.colorbar(sc, ax=axes[panel], label='Fraction in triode')
        axes[panel].set_title('Colored by Triode Fraction')
        panel += 1

    for ax in axes:
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.suptitle('Graph-Level t-SNE (Mean-Pooled Node Embeddings)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_graph_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'probe_graph_tsne.png'}")


# ---------------------------------------------------------------------------
# Analysis 6: Per-device t-SNE
# ---------------------------------------------------------------------------

def run_per_device_tsne(val_data, output_dir, devices_to_plot=None, max_points=3000):
    """t-SNE of a single device's drain embedding across all samples, colored by region."""
    from sklearn.manifold import TSNE

    device_id = val_data['device_id'].numpy()
    region = val_data['region'].numpy()
    X_all = val_data['final'].numpy()

    unique_ids = sorted(set(device_id[device_id >= 0]))
    if devices_to_plot is None:
        # Auto-select: pick devices with most region diversity
        diversity = {}
        for d_id in unique_ids:
            m = device_id == d_id
            regions_d = region[m]
            # Score = entropy-like: fraction not in majority class
            counts = np.bincount(regions_d[regions_d >= 0], minlength=3)
            total = counts.sum()
            if total > 0:
                majority = counts.max() / total
                diversity[d_id] = 1.0 - majority
        # Pick top 4 most diverse devices
        devices_to_plot = sorted(diversity, key=diversity.get, reverse=True)[:4]

    n_dev = len(devices_to_plot)
    fig, axes = plt.subplots(1, n_dev, figsize=(7 * n_dev, 6))
    if n_dev == 1:
        axes = [axes]

    region_names = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation', -1: 'Unknown'}
    region_colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', -1: '#95a5a6'}

    for i, d_id in enumerate(devices_to_plot):
        m = device_id == d_id
        X = X_all[m]
        reg = region[m]

        # Subsample if needed
        if len(X) > max_points:
            idx = np.random.default_rng(42).choice(len(X), max_points, replace=False)
            X, reg = X[idx], reg[idx]

        print(f"  Running per-device t-SNE for M{d_id} ({len(X)} points)...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(X) // 4), random_state=42, max_iter=1000)
        X_2d = tsne.fit_transform(X)

        # Count per region for legend
        for r_val in sorted(set(reg)):
            rm = reg == r_val
            if rm.any():
                count = rm.sum()
                pct = 100 * count / len(reg)
                axes[i].scatter(X_2d[rm, 0], X_2d[rm, 1],
                                c=region_colors.get(r_val, 'gray'),
                                s=8, alpha=0.5,
                                label=f'{region_names.get(r_val, f"R={r_val}")} ({pct:.0f}%)')

        axes[i].legend(markerscale=3, fontsize=9)
        axes[i].set_title(f'M{d_id} — Colored by Region')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')

    plt.suptitle('Per-Device t-SNE (Single Device Across Samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_per_device_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'probe_per_device_tsne.png'}")


# ---------------------------------------------------------------------------
# Data collection: per-MOSFET multi-terminal (gate + drain + source)
# ---------------------------------------------------------------------------

def collect_mosfet_terminal_data(model, batches, device):
    """Collect concatenated gate+drain+source embeddings per MOSFET."""
    mosfet_embs = []  # [G||D||S] concatenated
    mosfet_region = []
    mosfet_is_nmos = []
    mosfet_device_id = []

    for batch in batches:
        batch = batch.to(device)
        emb = extract_embeddings(model, batch, device)
        final = emb['final_repr']  # [num_nodes, hidden]

        mosfet_info = batch.mosfet_info
        ptr = batch.ptr
        num_mosfets = mosfet_info.shape[0]
        num_graphs = ptr.shape[0] - 1
        mosfets_per_graph = num_mosfets // num_graphs

        mosfet_graph_idx = torch.arange(num_mosfets, device=device) // mosfets_per_graph
        node_offsets = ptr[mosfet_graph_idx]

        # Terminal indices (columns 0=gate_t, 1=drain_t, 2=source_t)
        gate_idx = mosfet_info[:, 0].long() + node_offsets
        drain_idx = mosfet_info[:, 1].long() + node_offsets
        source_idx = mosfet_info[:, 2].long() + node_offsets

        # Concatenate G||D||S embeddings
        gate_emb = final[gate_idx]    # [num_mosfets, hidden]
        drain_emb = final[drain_idx]  # [num_mosfets, hidden]
        source_emb = final[source_idx]  # [num_mosfets, hidden]
        gds_emb = torch.cat([gate_emb, drain_emb, source_emb], dim=1)  # [num_mosfets, 3*hidden]
        mosfet_embs.append(gds_emb.cpu())

        # Region labels
        region_labels = batch.mosfet_region_labels if hasattr(batch, 'mosfet_region_labels') else None
        if region_labels is not None:
            mosfet_region.append(region_labels.cpu())
        else:
            mosfet_region.append(torch.full((num_mosfets,), -1, dtype=torch.long))

        mosfet_is_nmos.append(mosfet_info[:, 6].bool().cpu())
        mosfet_device_id.append((torch.arange(num_mosfets, device=device) % mosfets_per_graph).cpu())

    return {
        'emb': torch.cat(mosfet_embs),
        'region': torch.cat(mosfet_region),
        'is_nmos': torch.cat(mosfet_is_nmos),
        'device_id': torch.cat(mosfet_device_id),
    }


def run_mosfet_terminal_tsne(mosfet_data, output_dir, max_points=5000):
    """t-SNE of concatenated gate+drain+source embeddings per MOSFET."""
    from sklearn.manifold import TSNE

    X = mosfet_data['emb'].numpy()
    region = mosfet_data['region'].numpy()
    is_nmos = mosfet_data['is_nmos'].numpy()
    device_id = mosfet_data['device_id'].numpy()

    # Subsample if needed
    if len(X) > max_points:
        idx = np.random.default_rng(42).choice(len(X), max_points, replace=False)
        X, region, is_nmos, device_id = X[idx], region[idx], is_nmos[idx], device_id[idx]

    print(f"  Running multi-terminal t-SNE on {len(X)} MOSFETs ({X.shape[1]}-dim = 3×hidden)...")
    tsne = TSNE(n_components=2, perplexity=min(50, len(X) // 4), random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel 1: Colored by device
    unique_ids = sorted(set(device_id))
    cmap = plt.cm.get_cmap('tab20', len(unique_ids))
    for i, d_id in enumerate(unique_ids):
        m = device_id == d_id
        axes[0].scatter(X_2d[m, 0], X_2d[m, 1], c=[cmap(i)], s=5, alpha=0.4, label=f'M{d_id}')
    axes[0].set_title('Colored by Device')
    axes[0].legend(markerscale=3, fontsize=6, ncol=2, loc='best')

    # Panel 2: Colored by region
    region_names = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation', -1: 'Unknown'}
    region_colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', -1: '#95a5a6'}
    for r_val in sorted(set(region)):
        m = region == r_val
        if m.any():
            axes[1].scatter(X_2d[m, 0], X_2d[m, 1], c=region_colors.get(r_val, 'gray'),
                            s=5, alpha=0.4, label=region_names.get(r_val, f'R={r_val}'))
    axes[1].set_title('Colored by Operating Region')
    axes[1].legend(markerscale=3, fontsize=10)

    # Panel 3: Colored by NMOS/PMOS
    for is_n, label, color in [(True, 'NMOS', '#e74c3c'), (False, 'PMOS', '#3498db')]:
        m = is_nmos == is_n
        if m.any():
            axes[2].scatter(X_2d[m, 0], X_2d[m, 1], c=color, s=5, alpha=0.4, label=label)
    axes[2].set_title('Colored by Device Type')
    axes[2].legend(markerscale=3, fontsize=10)

    for ax in axes:
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.2)

    plt.suptitle('t-SNE of MOSFET Gate+Drain+Source Embeddings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_mosfet_gds_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'probe_mosfet_gds_tsne.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Probe GNN embeddings for interpretability')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to best_model.pt')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset dir (default: parent of checkpoint)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (reads/saves from experiments/<name>/)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Resolve paths: --name sets experiment dir under dataset
    if args.name:
        dataset_dir = Path(args.dataset) if args.dataset else Path('datasets/opamp_5k_ss_v1')
        experiment_dir = dataset_dir / 'experiments' / args.name
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else experiment_dir / 'best_model.pt'
        output_dir = experiment_dir
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        dataset_dir = Path(args.dataset) if args.dataset else checkpoint_path.parent
        output_dir = checkpoint_path.parent
    else:
        parser.error('Either --checkpoint or --name is required')
        return
    device = args.device

    print("=" * 60)
    print("  EMBEDDING PROBE ANALYSIS")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Dataset:    {dataset_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Device:     {device}")
    print()

    # Load model
    print("Loading model...")
    model, config, stats = load_checkpoint(str(checkpoint_path), device=device)
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Load data
    print("Loading data...")
    train_batches = load_prebatched_variant(dataset_dir / 'train', variant_id=0)
    val_batches = load_prebatched_variant(dataset_dir / 'val', variant_id=0)

    # Ensure SS targets exist
    add_ss_node_targets(train_batches)
    add_ss_node_targets(val_batches)
    print(f"  Train: {len(train_batches)} batches, Val: {len(val_batches)} batches")

    # Collect drain embeddings
    print("\nExtracting drain embeddings...")
    train_data = collect_drain_data(model, train_batches, device)
    val_data = collect_drain_data(model, val_batches, device)
    print(f"  Train drains: {len(train_data['gm'])}, Val drains: {len(val_data['gm'])}")

    # Detect if SS head is enabled (check both nested and flat config formats)
    has_ss = False
    if isinstance(config, dict):
        ss_cfg = config.get('model', {}).get('heads', {}).get('ss', {})
        if not ss_cfg:
            ss_cfg = config.get('ss_head_config', {})
        has_ss = ss_cfg.get('enabled', False)
    print(f"  SS head: {'enabled' if has_ss else 'disabled'}")

    if has_ss:
        # Analysis 1: Linear probe (only meaningful with SS)
        print("\n--- Analysis 1: Linear Probe (drain → gm/gds) ---")
        probe_results = run_linear_probe(train_data, val_data, output_dir)
        for name, res in probe_results.items():
            print(f"  {name}: R²={res['r2']:.3f}  MAE={res['mae']:.3f} log10")
    else:
        print("\n--- Skipping Linear Probe (SS head disabled) ---")
        probe_results = None

    # Analysis 2: t-SNE
    print("\n--- Analysis 2: t-SNE Visualization ---")
    run_tsne(val_data, output_dir, has_ss=has_ss)

    if has_ss:
        # Analysis 3: Layer-wise probe (only meaningful with SS)
        print("\n--- Analysis 3: Layer-wise Probing ---")
        layer_results = run_layerwise_probe(train_data, val_data, output_dir)
        if layer_results:
            print(f"  Overall peak — gm: {layer_results['gm_peak']} (R²={layer_results['gm_best']:.3f}), gds: {layer_results['gds_peak']} (R²={layer_results['gds_best']:.3f})")
            # Print per-section summaries
            bb_labels = layer_results['bb_labels']
            for section, gm_r2s, gds_r2s, labels in [
                ('Backbone', layer_results['bb_gm_r2'], layer_results['bb_gds_r2'], bb_labels),
                ('State tower', layer_results['st_gm_r2'], layer_results['st_gds_r2'], [f'ST{i}' for i in range(len(layer_results['st_gm_r2']))]),
                ('Sensitivity tower', layer_results['ss_gm_r2'], layer_results['ss_gds_r2'], [f'SS{i}' for i in range(len(layer_results['ss_gm_r2']))]),
            ]:
                if gm_r2s:
                    best_gm_i = np.argmax(gm_r2s)
                    best_gds_i = np.argmax(gds_r2s)
                    print(f"  {section}: gm best={labels[best_gm_i]} R²={gm_r2s[best_gm_i]:.3f}, gds best={labels[best_gds_i]} R²={gds_r2s[best_gds_i]:.3f}")
            if 'gm_branch_r2' in layer_results:
                print(f"  --- Y-branch layers (parallel from sensitivity) ---")
                for i in range(max(len(layer_results['gm_branch_r2']), len(layer_results['gds_branch_r2']))):
                    gm_r2 = layer_results['gm_branch_r2'][i] if i < len(layer_results['gm_branch_r2']) else None
                    gds_r2 = layer_results['gds_branch_r2'][i] if i < len(layer_results['gds_branch_r2']) else None
                    gm_str = f"gm R²={gm_r2:.3f}" if gm_r2 is not None else ""
                    gds_str = f"gds R²={gds_r2:.3f}" if gds_r2 is not None else ""
                    print(f"  BR{i}: {gm_str}  {gds_str}")
    else:
        print("\n--- Skipping Layer-wise Probing (SS head disabled) ---")

    # Analysis 4: PCA
    print("\n--- Analysis 4: PCA Visualization ---")
    pca_results = run_pca(val_data, output_dir)
    if pca_results:
        cumvar = pca_results['explained_variance'].cumsum()
        print(f"  Top 3 PCs explain {cumvar[2]*100:.1f}% of variance")
        print(f"  PC1 explains {pca_results['explained_variance'][0]*100:.1f}% alone")

    # Analysis 5: Graph-level t-SNE
    print("\n--- Analysis 5: Graph-Level t-SNE ---")
    print("  Collecting graph-level embeddings...")
    graph_data = collect_graph_level_data(model, val_batches, device)
    print(f"  {len(graph_data['ibias'])} graph embeddings collected")
    run_graph_tsne(graph_data, output_dir)

    # Analysis 6: Per-device t-SNE
    print("\n--- Analysis 6: Per-Device t-SNE ---")
    run_per_device_tsne(val_data, output_dir)

    # Analysis 7: Multi-terminal (G+D+S) t-SNE
    print("\n--- Analysis 7: MOSFET Gate+Drain+Source t-SNE ---")
    print("  Collecting multi-terminal embeddings...")
    mosfet_data = collect_mosfet_terminal_data(model, val_batches, device)
    print(f"  {len(mosfet_data['emb'])} MOSFET embeddings ({mosfet_data['emb'].shape[1]}-dim)")
    run_mosfet_terminal_tsne(mosfet_data, output_dir)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if probe_results:
        print(f"  gm  probe R² = {probe_results['gm']['r2']:.3f}  (MAE = {probe_results['gm']['mae']:.3f})")
        print(f"  gds probe R² = {probe_results['gds']['r2']:.3f}  (MAE = {probe_results['gds']['mae']:.3f})")
    else:
        print("  SS head disabled — gm/gds probing skipped")
    print()
    print(f"  Plots saved to: {output_dir}")
    print(f"    - probe_tsne.png")
    if has_ss:
        print(f"    - probe_linear.png")
        print(f"    - probe_layerwise.png")
    print(f"    - probe_pca.png")
    print(f"    - probe_pca_analysis.png")
    print(f"    - probe_graph_tsne.png")
    print(f"    - probe_per_device_tsne.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
