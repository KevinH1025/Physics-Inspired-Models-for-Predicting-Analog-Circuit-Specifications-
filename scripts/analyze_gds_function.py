#!/usr/bin/env python3
"""Check if G+D+S embeddings cluster by circuit function (mirror, diff pair, cascode, etc.)."""
import sys
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import load_prebatched_variant, add_ss_node_targets
from scripts.probe_embeddings import extract_embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Circuit function grouping based on actual netlist topology
FUNCTION_MAP = {
    # Current mirrors (gate tied to same bias node, sourcing/sinking copies of reference)
    0:  'mirror',      # Diode-connected bias ref (mirror master)
    1:  'mirror',      # Mirror copy → VB4
    2:  'mirror',      # Mirror copy → DM_1
    3:  'mirror',      # Mirror copy → VB3
    4:  'mirror',      # Tail current (mirror copy, 4x)
    5:  'mirror',      # CMFB diode (mirror master)
    6:  'mirror',      # CMFB mirror copy
    7:  'mirror',      # Mirror copy → Stage2

    # Differential pair
    8:  'diff_pair',   # Diff pair INV
    9:  'diff_pair',   # Diff pair NINV

    # Cascode (stacked devices for high output impedance)
    12: 'cascode',     # Cascode top for VB4
    13: 'cascode',     # Cascode top for DM_1
    15: 'cascode',     # Cascode top left (Stage1 load)
    16: 'cascode',     # Cascode top right (Stage1 load)

    # Current sources (bottom devices in cascode stacks, gate=vb4)
    10: 'current_src', # Bottom cascode left (Stage1)
    11: 'current_src', # Bottom cascode right (Stage1)
    17: 'current_src', # Bottom for VB4 gen
    18: 'current_src', # Bottom for DM_1

    # Diode-connected (gate=drain, sets voltage reference)
    14: 'diode',       # Diode-connected → VB3

    # Common-source gain stage
    19: 'cs_gain',     # Stage 2 PMOS gain

    # Active load (diode + mirror for gain stage)
    20: 'active_load', # Stage 2 diode load
    21: 'active_load', # Stage 2 mirror load

    # Output drivers (push-pull)
    22: 'output',      # PMOS output driver
    23: 'output',      # NMOS output driver
}

# Also test: connection-based grouping (devices sharing the same gate node)
GATE_NODE_MAP = {
    0:  'nbias_diode',  # gate=drain (diode)
    1:  'nbias', 2: 'nbias', 3: 'nbias', 4: 'nbias',  # gate=nbias
    5:  'voutn_diode',  # gate=drain (diode)
    6:  'voutn', 7: 'voutn',  # gate=voutn
    8:  'vinn',   # gate=vinn
    9:  'vinp',   # gate=vinp
    10: 'vb3', 11: 'vb3',  # gate=vb3 (cascode bias)
    12: 'vb3', 13: 'vb3',
    14: 'vb3_diode',    # gate=drain (diode)
    15: 'vb4', 16: 'vb4',  # gate=vb4
    17: 'vb4', 18: 'vb4',
    19: 'net050',  # gate=Stage1 output
    20: 'net043_diode',  # gate=drain (diode)
    21: 'net043',  # gate=net043 (mirrors M20)
    22: 'net050',  # gate=Stage1 output (same as M19!)
    23: 'net049',  # gate=Stage2 output
}

# Device type for reference
DEVICE_INFO = {
    0:  ('PMOS', 'mirror'),  1:  ('PMOS', 'mirror'),  2:  ('PMOS', 'mirror'),
    3:  ('PMOS', 'mirror'),  4:  ('PMOS', 'mirror'),  5:  ('PMOS', 'mirror'),
    6:  ('PMOS', 'mirror'),  7:  ('PMOS', 'mirror'),
    8:  ('PMOS', 'diff_pair'), 9: ('PMOS', 'diff_pair'),
    10: ('NMOS', 'current_src'), 11: ('NMOS', 'current_src'),
    12: ('NMOS', 'cascode'), 13: ('NMOS', 'cascode'),
    14: ('NMOS', 'diode'),
    15: ('NMOS', 'cascode'), 16: ('NMOS', 'cascode'),
    17: ('NMOS', 'current_src'), 18: ('NMOS', 'current_src'),
    19: ('PMOS', 'cs_gain'),
    20: ('NMOS', 'active_load'), 21: ('NMOS', 'active_load'),
    22: ('PMOS', 'output'), 23: ('NMOS', 'output'),
}

def group_similarity(cos_sim, grouping):
    """Compute avg within-group and between-group cosine similarity."""
    groups = {}
    for d, g in grouping.items():
        if g not in groups:
            groups[g] = []
        groups[g].append(d)

    within_sims = []
    between_sims = []
    group_keys = sorted(groups.keys())

    for g1 in group_keys:
        devs1 = groups[g1]
        for i in devs1:
            for j in devs1:
                if i < j:
                    within_sims.append(cos_sim[i, j])
            for g2 in group_keys:
                if g1 < g2:
                    for j in groups[g2]:
                        between_sims.append(cos_sim[i, j])

    return np.mean(within_sims) if within_sims else 0, np.mean(between_sims) if between_sims else 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='datasets/opamp_3stage_fan_smc_v8_5k_nofil/experiments/tower/fp32_bulk_voltctx_w512/best_model.pt')
    parser.add_argument('--dataset', type=str,
                        default='datasets/opamp_3stage_fan_smc_v8_5k_nofil')
    args = parser.parse_args()
    ckpt = args.checkpoint
    dataset = args.dataset

    print("Loading model...")
    model, config, stats = load_checkpoint(ckpt, device=device)
    model.eval()

    print("Loading data...")
    val_batches = load_prebatched_variant(Path(dataset) / 'val', variant_id=0)
    add_ss_node_targets(val_batches)

    print("Extracting G+D+S embeddings...")
    mosfet_embs = []
    mosfet_device_id = []

    for batch in val_batches:
        batch = batch.to(device)
        emb = extract_embeddings(model, batch, device)
        final = emb['final_repr']

        mosfet_info = batch.mosfet_info
        ptr = batch.ptr
        num_mosfets = mosfet_info.shape[0]
        num_graphs = ptr.shape[0] - 1
        mosfets_per_graph = num_mosfets // num_graphs

        mosfet_graph_idx = torch.arange(num_mosfets, device=device) // mosfets_per_graph
        node_offsets = ptr[mosfet_graph_idx]

        gate_idx = mosfet_info[:, 0].long() + node_offsets
        drain_idx = mosfet_info[:, 1].long() + node_offsets
        source_idx = mosfet_info[:, 2].long() + node_offsets

        gds_emb = torch.cat([final[gate_idx], final[drain_idx], final[source_idx]], dim=1)
        mosfet_embs.append(gds_emb.cpu())
        mosfet_device_id.append((torch.arange(num_mosfets, device=device) % mosfets_per_graph).cpu())

    embs = torch.cat(mosfet_embs).numpy()
    dev_ids = torch.cat(mosfet_device_id).numpy()

    # Compute centroids
    centroids = {}
    for d in range(24):
        centroids[d] = embs[dev_ids == d].mean(axis=0)

    centroid_mat = np.stack([centroids[d] for d in range(24)])
    cos_sim = 1 - cdist(centroid_mat, centroid_mat, metric='cosine')

    # =============================================
    # All groupings comparison
    # =============================================
    print("\n" + "="*80)
    print("CLUSTERING BY DIFFERENT GROUPINGS")
    print("="*80)

    # Type
    type_map = {d: DEVICE_INFO[d][0] for d in range(24)}
    w, b = group_similarity(cos_sim, type_map)
    print(f"\n{'Grouping':<30} {'Within':>8} {'Between':>8} {'Ratio':>7}")
    print("-"*60)
    print(f"{'NMOS/PMOS':<30} {w:>8.3f} {b:>8.3f} {w/b:>7.2f}")

    # Stage
    stage_map = {
        0:'Bias',1:'Bias',2:'Bias',3:'Bias',4:'Bias',5:'Bias',6:'Bias',7:'Bias',
        8:'Stage1',9:'Stage1',10:'Stage1',11:'Stage1',
        12:'Bias',13:'Bias',14:'Bias',15:'Stage1',16:'Stage1',17:'Bias',18:'Bias',
        19:'Stage2',20:'Stage2',21:'Stage2',22:'Stage3',23:'Stage3',
    }
    w, b = group_similarity(cos_sim, stage_map)
    print(f"{'Circuit Stage':<30} {w:>8.3f} {b:>8.3f} {w/b:>7.2f}")

    # W/L group
    wl_map = {
        0:'BIASCM_P',1:'BIASCM_P',2:'BIASCM_P',3:'BIASCM_P',
        4:'BIASCM_P',5:'BIASCM_P',6:'BIASCM_P',7:'BIASCM_P',
        8:'GM1',9:'GM1',
        10:'BIASCM_N',11:'BIASCM_N',12:'BIASCM_N',13:'BIASCM_N',14:'BIASCM_N',
        15:'BIASCM_N',16:'BIASCM_N',17:'BIASCM_N',18:'BIASCM_N',
        19:'GM2',20:'LOAD2',21:'LOAD2',22:'GMF2',23:'GM3',
    }
    w, b = group_similarity(cos_sim, wl_map)
    print(f"{'W/L Sizing Group':<30} {w:>8.3f} {b:>8.3f} {w/b:>7.2f}")

    # Signal/Bias
    sig_map = {
        0:'bias',1:'bias',2:'bias',3:'bias',4:'bias',5:'bias',6:'bias',7:'bias',
        8:'signal',9:'signal',10:'signal',11:'signal',
        12:'bias',13:'bias',14:'bias',15:'signal',16:'signal',17:'bias',18:'bias',
        19:'signal',20:'signal',21:'signal',22:'signal',23:'signal',
    }
    w, b = group_similarity(cos_sim, sig_map)
    print(f"{'Signal vs Bias':<30} {w:>8.3f} {b:>8.3f} {w/b:>7.2f}")

    # Circuit function
    w, b = group_similarity(cos_sim, FUNCTION_MAP)
    print(f"{'Circuit Function':<30} {w:>8.3f} {b:>8.3f} {w/b:>7.2f}")

    # Gate node (shared control signal)
    w, b = group_similarity(cos_sim, GATE_NODE_MAP)
    print(f"{'Shared Gate Node':<30} {w:>8.3f} {b:>8.3f} {w/b:>7.2f}")

    # =============================================
    # Per-function group details
    # =============================================
    print("\n" + "="*80)
    print("PER CIRCUIT FUNCTION GROUP — INTERNAL SIMILARITY")
    print("="*80)

    func_groups = {}
    for d, f in FUNCTION_MAP.items():
        if f not in func_groups:
            func_groups[f] = []
        func_groups[f].append(d)

    print(f"\n{'Function':<15} {'Devices':<35} {'Avg internal sim':>16} {'Avg to others':>14} {'Ratio':>7}")
    print("-"*90)
    for func in sorted(func_groups.keys()):
        devs = func_groups[func]
        # Internal sim
        internal = []
        for i in devs:
            for j in devs:
                if i < j:
                    internal.append(cos_sim[i, j])
        avg_int = np.mean(internal) if internal else 1.0

        # External sim
        external = []
        for i in devs:
            for j in range(24):
                if j not in devs:
                    external.append(cos_sim[i, j])
        avg_ext = np.mean(external)

        dev_str = ', '.join(f'M{d}' for d in devs)
        dtype_str = '/'.join(sorted(set(DEVICE_INFO[d][0] for d in devs)))
        ratio = avg_int / avg_ext if avg_ext > 0 else float('inf')
        print(f"{func:<15} {dev_str + ' (' + dtype_str + ')':<35} {avg_int:>16.3f} {avg_ext:>14.3f} {ratio:>7.2f}")

    # =============================================
    # Per gate-node group details
    # =============================================
    print("\n" + "="*80)
    print("PER GATE NODE GROUP — INTERNAL SIMILARITY")
    print("(Devices whose gates connect to the same net)")
    print("="*80)

    gate_groups = {}
    for d, g in GATE_NODE_MAP.items():
        if g not in gate_groups:
            gate_groups[g] = []
        gate_groups[g].append(d)

    print(f"\n{'Gate node':<15} {'Devices':<30} {'Avg internal sim':>16} {'Avg to others':>14} {'Ratio':>7}")
    print("-"*85)
    for node in sorted(gate_groups.keys()):
        devs = gate_groups[node]
        if len(devs) < 2:
            continue
        internal = []
        for i in devs:
            for j in devs:
                if i < j:
                    internal.append(cos_sim[i, j])
        avg_int = np.mean(internal) if internal else 1.0

        external = []
        for i in devs:
            for j in range(24):
                if j not in devs:
                    external.append(cos_sim[i, j])
        avg_ext = np.mean(external)

        dev_str = ', '.join(f'M{d}' for d in devs)
        ratio = avg_int / avg_ext if avg_ext > 0 else float('inf')
        print(f"{node:<15} {dev_str:<30} {avg_int:>16.3f} {avg_ext:>14.3f} {ratio:>7.2f}")

    # =============================================
    # Cross-type function similarity
    # =============================================
    print("\n" + "="*80)
    print("CROSS-TYPE SIMILARITY: Do PMOS mirrors look like NMOS current sources?")
    print("="*80)

    pmos_mirrors = [0,1,2,3,4,5,6,7]
    nmos_curr_src = [10,11,17,18]
    nmos_cascode = [12,13,15,16]
    diff_pair = [8,9]
    output = [22,23]

    def avg_cross_sim(group_a, group_b):
        sims = [cos_sim[i,j] for i in group_a for j in group_b]
        return np.mean(sims)

    print(f"\n{'Group A':<25} {'Group B':<25} {'Avg sim':>8}")
    print("-"*62)
    print(f"{'PMOS mirrors (M0-M7)':<25} {'NMOS curr src (M10,11,17,18)':<25} {avg_cross_sim(pmos_mirrors, nmos_curr_src):>8.3f}")
    print(f"{'PMOS mirrors (M0-M7)':<25} {'NMOS cascode (M12,13,15,16)':<25} {avg_cross_sim(pmos_mirrors, nmos_cascode):>8.3f}")
    print(f"{'PMOS mirrors (M0-M7)':<25} {'Diff pair (M8,M9)':<25} {avg_cross_sim(pmos_mirrors, diff_pair):>8.3f}")
    print(f"{'PMOS mirrors (M0-M7)':<25} {'Output (M22,M23)':<25} {avg_cross_sim(pmos_mirrors, output):>8.3f}")
    print(f"{'NMOS curr src':<25} {'NMOS cascode':<25} {avg_cross_sim(nmos_curr_src, nmos_cascode):>8.3f}")
    print(f"{'NMOS curr src':<25} {'Output (M22,M23)':<25} {avg_cross_sim(nmos_curr_src, output):>8.3f}")
    print(f"{'Diff pair (M8,M9)':<25} {'Output (M22,M23)':<25} {avg_cross_sim(diff_pair, output):>8.3f}")
    print(f"{'Diff pair (M8,M9)':<25} {'CS gain (M19)':<25} {avg_cross_sim(diff_pair, [19]):>8.3f}")
    print(f"{'CS gain (M19)':<25} {'Output PMOS (M22)':<25} {avg_cross_sim([19], [22]):>8.3f}")
    print(f"{'Stage2 load (M20,M21)':<25} {'NMOS cascode':<25} {avg_cross_sim([20,21], nmos_cascode):>8.3f}")

    # Internal similarities for reference
    print(f"\n--- Internal group similarities (for reference) ---")
    print(f"PMOS mirrors internal:    {avg_cross_sim(pmos_mirrors, pmos_mirrors):>8.3f}")
    print(f"NMOS curr src internal:   {avg_cross_sim(nmos_curr_src, nmos_curr_src):>8.3f}")
    print(f"NMOS cascode internal:    {avg_cross_sim(nmos_cascode, nmos_cascode):>8.3f}")

    # =============================================
    # PLOTS
    # =============================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    out_dir = Path(ckpt).parent / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for t-SNE
    np.random.seed(42)
    idx = np.random.choice(len(embs), min(5000, len(embs)), replace=False)
    X_sub = embs[idx]
    dev_sub = dev_ids[idx]

    print("\nRunning t-SNE for plots...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_sub)

    # --- Plot 1: t-SNE colored by circuit function ---
    func_colors = {
        'mirror': '#3498db', 'diff_pair': '#e74c3c', 'cascode': '#2ecc71',
        'current_src': '#9b59b6', 'diode': '#f39c12', 'cs_gain': '#1abc9c',
        'active_load': '#e67e22', 'output': '#c0392b',
    }

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Panel 1: by circuit function
    for func, color in func_colors.items():
        mask = np.array([FUNCTION_MAP[d] == func for d in dev_sub])
        if mask.any():
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=8, alpha=0.5, label=func)
    axes[0].legend(markerscale=3, fontsize=9, loc='best')
    axes[0].set_title('Colored by Circuit Function', fontsize=13, fontweight='bold')

    # Panel 2: by shared gate node
    gate_node_colors = {
        'nbias': '#e74c3c', 'nbias_diode': '#c0392b',
        'voutn': '#3498db', 'voutn_diode': '#2980b9',
        'vinn': '#e67e22', 'vinp': '#f39c12',
        'vb3': '#2ecc71', 'vb3_diode': '#27ae60',
        'vb4': '#9b59b6',
        'net050': '#1abc9c', 'net043': '#16a085', 'net043_diode': '#148f77',
        'net049': '#c0392b',
    }
    for node, color in gate_node_colors.items():
        mask = np.array([GATE_NODE_MAP[d] == node for d in dev_sub])
        if mask.any():
            n_devs = len([d for d in range(24) if GATE_NODE_MAP[d] == node])
            label = f"{node} ({n_devs})" if n_devs > 1 else node
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=8, alpha=0.5, label=label)
    axes[1].legend(markerscale=3, fontsize=7, loc='best', ncol=2)
    axes[1].set_title('Colored by Shared Gate Node', fontsize=13, fontweight='bold')

    # Panel 3: by function with device labels at centroids
    for func, color in func_colors.items():
        mask = np.array([FUNCTION_MAP[d] == func for d in dev_sub])
        if mask.any():
            axes[2].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=6, alpha=0.3)
    # Add labels
    for d in range(24):
        mask_d = dev_sub == d
        if mask_d.sum() > 0:
            cx, cy = X_2d[mask_d].mean(axis=0)
            func = FUNCTION_MAP[d]
            axes[2].annotate(f"M{d}\n{func[:6]}", (cx, cy), fontsize=6, fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='gray'))
    axes[2].set_title('Function Labels at Centroids', fontsize=13, fontweight='bold')

    for ax in axes:
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.2)

    plt.suptitle('G+D+S Embeddings — Circuit Function & Gate Node Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'gds_tsne_circuit_function.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'gds_tsne_circuit_function.png'}")

    # --- Plot 2: Grouping comparison bar chart ---
    groupings = ['NMOS/PMOS', 'Circuit\nFunction', 'W/L Sizing\nGroup', 'Shared\nGate Node', 'Circuit\nStage', 'Signal\nvs Bias']
    ratios = []
    # Recompute for the bar chart
    for gmap in [type_map, FUNCTION_MAP, wl_map, GATE_NODE_MAP, stage_map, sig_map]:
        w, b = group_similarity(cos_sim, gmap)
        ratios.append(w / b)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(groupings))
    colors = ['#2ecc71' if r > 1.2 else '#f39c12' if r > 1.1 else '#e74c3c' for r in ratios]
    bars = ax.bar(x_pos, ratios, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axhline(1.0, color='black', linewidth=1.5, linestyle='--', label='No clustering (ratio=1.0)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groupings, fontsize=10)
    ax.set_ylabel('Within-group / Between-group Similarity', fontsize=11)
    ax.set_title('What Drives G+D+S Embedding Clustering?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{ratio:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0.9, max(ratios) + 0.1)
    plt.tight_layout()
    plt.savefig(out_dir / 'gds_clustering_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'gds_clustering_comparison.png'}")

    # --- Plot 3: Cosine similarity heatmap ordered by function ---
    func_order = []
    func_labels = []
    func_boundaries = []
    for func in ['mirror', 'diff_pair', 'cascode', 'current_src', 'diode', 'cs_gain', 'active_load', 'output']:
        devs = [d for d in range(24) if FUNCTION_MAP[d] == func]
        func_boundaries.append(len(func_order))
        for d in devs:
            func_order.append(d)
            func_labels.append(f"M{d}")

    reordered_sim = cos_sim[np.ix_(func_order, func_order)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(reordered_sim, cmap='RdYlBu_r', vmin=0.4, vmax=1.0)
    ax.set_xticks(range(24))
    ax.set_yticks(range(24))
    ax.set_xticklabels(func_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(func_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)

    # Draw function boundaries
    for b in func_boundaries[1:]:
        ax.axhline(b - 0.5, color='black', linewidth=2)
        ax.axvline(b - 0.5, color='black', linewidth=2)

    # Label function groups on the side
    for i, func in enumerate(['mirror', 'diff_pair', 'cascode', 'current_src', 'diode', 'cs_gain', 'active_load', 'output']):
        start = func_boundaries[i]
        end = func_boundaries[i+1] if i+1 < len(func_boundaries) else 24
        mid = (start + end - 1) / 2
        ax.text(-1.5, mid, func, ha='right', va='center', fontsize=8, fontweight='bold',
               color=func_colors.get(func, 'black'))

    ax.set_title('Cosine Similarity — Ordered by Circuit Function', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'gds_cosine_by_function.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'gds_cosine_by_function.png'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
