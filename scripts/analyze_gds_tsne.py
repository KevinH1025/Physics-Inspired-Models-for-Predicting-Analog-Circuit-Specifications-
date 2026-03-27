#!/usr/bin/env python3
"""Quantitative analysis of G+D+S MOSFET embeddings for tower w512 model."""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import load_prebatched_variant, add_ss_node_targets
from scripts.probe_embeddings import extract_embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Circuit role mapping for 3-stage opamp
DEVICE_INFO = {
    0:  ('PMOS', 'Bias',    'Diode-connected bias ref'),
    1:  ('PMOS', 'Bias',    'Bias mirror → VB4'),
    2:  ('PMOS', 'Bias',    'Bias mirror → DM_1'),
    3:  ('PMOS', 'Bias',    'Bias mirror → VB3'),
    4:  ('PMOS', 'Bias',    'Tail current source (4x)'),
    5:  ('PMOS', 'Bias',    'CMFB diode-connected'),
    6:  ('PMOS', 'Bias',    'CMFB mirror → Stage1'),
    7:  ('PMOS', 'Bias',    'Bias mirror → Stage2'),
    8:  ('PMOS', 'Stage1',  'Diff pair INV'),
    9:  ('PMOS', 'Stage1',  'Diff pair NINV'),
    10: ('NMOS', 'Stage1',  'Cascode load left (bottom)'),
    11: ('NMOS', 'Stage1',  'Cascode load right (bottom)'),
    12: ('NMOS', 'Bias',    'Cascode VB4 gen (top)'),
    13: ('NMOS', 'Bias',    'Cascode DM_1 (top)'),
    14: ('NMOS', 'Bias',    'Diode-connected → VB3'),
    15: ('NMOS', 'Stage1',  'Cascode left (top, 4x)'),
    16: ('NMOS', 'Stage1',  'Cascode right (top, 4x)'),
    17: ('NMOS', 'Bias',    'Current src VB4 (bottom)'),
    18: ('NMOS', 'Bias',    'Current src DM_1 (bottom)'),
    19: ('PMOS', 'Stage2',  'CS gain stage'),
    20: ('NMOS', 'Stage2',  'Diode load'),
    21: ('NMOS', 'Stage2',  'Mirror load → Stage2 out'),
    22: ('PMOS', 'Stage3',  'PMOS output driver'),
    23: ('NMOS', 'Stage3',  'NMOS output driver'),
}

# Known matched pairs
MATCHED_PAIRS = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3),  # bias mirrors
                 (5,6), (8,9), (10,11), (12,13), (15,16), (17,18), (20,21)]

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

    # Collect G+D+S embeddings
    print("Extracting G+D+S embeddings...")
    mosfet_embs = []
    mosfet_device_id = []
    mosfet_region = []

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

        gate_emb = final[gate_idx]
        drain_emb = final[drain_idx]
        source_emb = final[source_idx]
        gds_emb = torch.cat([gate_emb, drain_emb, source_emb], dim=1)
        mosfet_embs.append(gds_emb.cpu())

        region_labels = batch.mosfet_region_labels if hasattr(batch, 'mosfet_region_labels') else None
        if region_labels is not None:
            mosfet_region.append(region_labels.cpu())
        else:
            mosfet_region.append(torch.full((num_mosfets,), -1, dtype=torch.long))

        mosfet_device_id.append((torch.arange(num_mosfets, device=device) % mosfets_per_graph).cpu())

    embs = torch.cat(mosfet_embs).numpy()
    dev_ids = torch.cat(mosfet_device_id).numpy()
    regions = torch.cat(mosfet_region).numpy()

    print(f"Total MOSFETs: {len(embs)}, dim: {embs.shape[1]}")
    print(f"Devices per graph: {mosfets_per_graph}")

    # =============================================
    # 1. Per-device centroid analysis
    # =============================================
    print("\n" + "="*80)
    print("1. PER-DEVICE CENTROID ANALYSIS")
    print("="*80)

    centroids = {}
    spreads = {}
    for d in range(24):
        mask = dev_ids == d
        X = embs[mask]
        centroids[d] = X.mean(axis=0)
        # Spread = mean distance from centroid
        spreads[d] = np.mean(np.linalg.norm(X - centroids[d], axis=1))

    print(f"\n{'Dev':>4} {'Type':>5} {'Stage':>7} {'Function':<35} {'Spread':>7} {'Region dist'}")
    print("-"*100)
    for d in range(24):
        dtype, stage, func = DEVICE_INFO[d]
        mask = dev_ids == d
        r = regions[mask]
        sat = (r == 2).mean() * 100
        tri = (r == 1).mean() * 100
        cut = (r == 0).mean() * 100
        print(f"M{d:02d}  {dtype:>5} {stage:>7} {func:<35} {spreads[d]:7.2f}   sat={sat:.0f}% tri={tri:.0f}% cut={cut:.0f}%")

    # =============================================
    # 2. Inter-device centroid distances
    # =============================================
    print("\n" + "="*80)
    print("2. INTER-DEVICE CENTROID DISTANCE MATRIX (cosine similarity)")
    print("="*80)

    centroid_mat = np.stack([centroids[d] for d in range(24)])
    cos_sim = 1 - cdist(centroid_mat, centroid_mat, metric='cosine')

    # Find top-5 most similar pairs
    pairs = []
    for i in range(24):
        for j in range(i+1, 24):
            pairs.append((i, j, cos_sim[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 15 most similar device pairs (cosine similarity of centroids):")
    print(f"{'Pair':>10} {'Sim':>6} {'Same stage?':>12} {'Same type?':>11} {'Matched?':>9}")
    print("-"*55)
    for i, j, sim in pairs[:15]:
        same_stage = DEVICE_INFO[i][1] == DEVICE_INFO[j][1]
        same_type = DEVICE_INFO[i][0] == DEVICE_INFO[j][0]
        matched = (i, j) in MATCHED_PAIRS or (j, i) in MATCHED_PAIRS
        ss = 'YES' if same_stage else 'NO'
        st = 'YES' if same_type else 'NO'
        mp = 'YES' if matched else ''
        print(f"M{i:02d}-M{j:02d}  {sim:6.3f}  {ss:>12}  {st:>11}  {mp:>9}")

    print("\nTop 10 most DISSIMILAR device pairs:")
    print(f"{'Pair':>10} {'Sim':>6} {'Types':>15}")
    print("-"*40)
    for i, j, sim in pairs[-10:]:
        print(f"M{i:02d}-M{j:02d}  {sim:6.3f}  {DEVICE_INFO[i][0]}/{DEVICE_INFO[j][0]}")

    # =============================================
    # 3. Group-level analysis: do devices cluster by stage or function?
    # =============================================
    print("\n" + "="*80)
    print("3. CLUSTER ANALYSIS: STAGE vs FUNCTION vs TYPE")
    print("="*80)

    # Average within-group similarity vs between-group similarity
    def group_similarity(grouping):
        """Compute avg within-group and between-group cosine similarity."""
        groups = {}
        for d in range(24):
            g = grouping[d]
            if g not in groups:
                groups[g] = []
            groups[g].append(d)

        within_sims = []
        between_sims = []
        for g1, devs1 in groups.items():
            for i in devs1:
                for j in devs1:
                    if i < j:
                        within_sims.append(cos_sim[i, j])
                for g2, devs2 in groups.items():
                    if g1 < g2:
                        for j in devs2:
                            between_sims.append(cos_sim[i, j])

        return np.mean(within_sims) if within_sims else 0, np.mean(between_sims) if between_sims else 0

    # Grouping by stage
    stage_grouping = {d: DEVICE_INFO[d][1] for d in range(24)}
    w_stage, b_stage = group_similarity(stage_grouping)

    # Grouping by type (NMOS/PMOS)
    type_grouping = {d: DEVICE_INFO[d][0] for d in range(24)}
    w_type, b_type = group_similarity(type_grouping)

    # Grouping by W/L group (from netlist)
    wl_grouping = {
        0: 'BIASCM_P', 1: 'BIASCM_P', 2: 'BIASCM_P', 3: 'BIASCM_P',
        4: 'BIASCM_P', 5: 'BIASCM_P', 6: 'BIASCM_P', 7: 'BIASCM_P',
        8: 'GM1', 9: 'GM1',
        10: 'BIASCM_N', 11: 'BIASCM_N',
        12: 'BIASCM_N', 13: 'BIASCM_N', 14: 'BIASCM_N',
        15: 'BIASCM_N', 16: 'BIASCM_N',
        17: 'BIASCM_N', 18: 'BIASCM_N',
        19: 'GM2', 20: 'LOAD2', 21: 'LOAD2',
        22: 'GMF2', 23: 'GM3',
    }
    w_wl, b_wl = group_similarity(wl_grouping)

    # Grouping by signal role: bias vs signal-path
    signal_grouping = {
        0: 'bias', 1: 'bias', 2: 'bias', 3: 'bias', 4: 'bias',
        5: 'bias', 6: 'bias', 7: 'bias',
        8: 'signal', 9: 'signal',
        10: 'signal', 11: 'signal',
        12: 'bias', 13: 'bias', 14: 'bias',
        15: 'signal', 16: 'signal',
        17: 'bias', 18: 'bias',
        19: 'signal', 20: 'signal', 21: 'signal',
        22: 'signal', 23: 'signal',
    }
    w_sig, b_sig = group_similarity(signal_grouping)

    print(f"\n{'Grouping':<20} {'Within-group sim':>18} {'Between-group sim':>18} {'Ratio':>8}")
    print("-"*70)
    print(f"{'By NMOS/PMOS':<20} {w_type:>18.3f} {b_type:>18.3f} {w_type/b_type:>8.2f}")
    print(f"{'By Stage':<20} {w_stage:>18.3f} {b_stage:>18.3f} {w_stage/b_stage:>8.2f}")
    print(f"{'By W/L group':<20} {w_wl:>18.3f} {b_wl:>18.3f} {w_wl/b_wl:>8.2f}")
    print(f"{'By Signal/Bias':<20} {w_sig:>18.3f} {b_sig:>18.3f} {w_sig/b_sig:>8.2f}")

    # =============================================
    # 4. Matched pair analysis
    # =============================================
    print("\n" + "="*80)
    print("4. MATCHED PAIR SIMILARITY")
    print("="*80)
    print(f"\n{'Pair':>10} {'Cosine sim':>11} {'Function'}")
    print("-"*65)
    for i, j in MATCHED_PAIRS:
        sim = cos_sim[i, j]
        print(f"M{i:02d}-M{j:02d}  {sim:11.4f}   {DEVICE_INFO[i][2]} / {DEVICE_INFO[j][2]}")

    avg_matched = np.mean([cos_sim[i, j] for i, j in MATCHED_PAIRS])
    # Random non-matched pairs for comparison
    non_matched_sims = []
    for i in range(24):
        for j in range(i+1, 24):
            if (i, j) not in MATCHED_PAIRS and (j, i) not in MATCHED_PAIRS:
                non_matched_sims.append(cos_sim[i, j])
    avg_non_matched = np.mean(non_matched_sims)
    print(f"\nAvg matched pair sim:     {avg_matched:.4f}")
    print(f"Avg non-matched pair sim: {avg_non_matched:.4f}")
    print(f"Ratio: {avg_matched/avg_non_matched:.2f}x")

    # =============================================
    # 5. Intra-device region separation
    # =============================================
    print("\n" + "="*80)
    print("5. INTRA-DEVICE OPERATING REGION SEPARATION")
    print("   (Do sat/triode/cutoff samples form sub-clusters within each device?)")
    print("="*80)

    print(f"\n{'Dev':>4} {'Stage':>7} {'Sat-Tri dist':>13} {'Sat-Cut dist':>13} {'Intra-Sat':>10} {'Separation':>11}")
    print("-"*65)
    for d in range(24):
        mask = dev_ids == d
        X = embs[mask]
        r = regions[mask]

        sat_mask = r == 2
        tri_mask = r == 1
        cut_mask = r == 0

        if sat_mask.sum() < 10:
            continue

        sat_centroid = X[sat_mask].mean(axis=0)
        intra_sat = np.mean(np.linalg.norm(X[sat_mask] - sat_centroid, axis=1))

        results = []
        if tri_mask.sum() >= 10:
            tri_centroid = X[tri_mask].mean(axis=0)
            sat_tri_dist = np.linalg.norm(sat_centroid - tri_centroid)
            results.append(f"{sat_tri_dist:13.2f}")
        else:
            sat_tri_dist = None
            results.append(f"{'N/A':>13}")

        if cut_mask.sum() >= 10:
            cut_centroid = X[cut_mask].mean(axis=0)
            sat_cut_dist = np.linalg.norm(sat_centroid - cut_centroid)
            results.append(f"{sat_cut_dist:13.2f}")
        else:
            sat_cut_dist = None
            results.append(f"{'N/A':>13}")

        # Separation ratio: inter-region distance / intra-region spread
        # > 1 means regions are separated, < 1 means overlapping
        best_dist = sat_tri_dist or sat_cut_dist
        sep = best_dist / (intra_sat + 1e-8) if best_dist else None

        stage = DEVICE_INFO[d][1]
        print(f"M{d:02d}  {stage:>7} {results[0]} {results[1]} {intra_sat:10.2f} {sep:11.2f}" if sep else
              f"M{d:02d}  {stage:>7} {results[0]} {results[1]} {intra_sat:10.2f} {'N/A':>11}")

    # =============================================
    # 6. t-SNE with device labels for visual reference
    # =============================================
    print("\n" + "="*80)
    print("6. RUNNING t-SNE FOR NUMERICAL COORDINATE ANALYSIS")
    print("="*80)

    # Subsample for t-SNE
    np.random.seed(42)
    idx = np.random.choice(len(embs), min(5000, len(embs)), replace=False)
    X_sub = embs[idx]
    dev_sub = dev_ids[idx]
    reg_sub = regions[idx]

    print(f"Running t-SNE on {len(X_sub)} points...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_sub)

    # Per-device t-SNE centroids
    print(f"\n{'Dev':>4} {'Type':>5} {'Stage':>7} {'t-SNE x':>8} {'t-SNE y':>8} {'Spread':>7} {'Nearest neighbor'}")
    print("-"*75)
    tsne_centroids = {}
    tsne_spreads = {}
    for d in range(24):
        mask = dev_sub == d
        if mask.sum() < 5:
            continue
        pts = X_2d[mask]
        c = pts.mean(axis=0)
        tsne_centroids[d] = c
        tsne_spreads[d] = np.mean(np.linalg.norm(pts - c, axis=1))

    # Find nearest neighbor for each device
    for d in sorted(tsne_centroids.keys()):
        best_dist = float('inf')
        best_nn = -1
        for d2 in tsne_centroids:
            if d2 != d:
                dist = np.linalg.norm(tsne_centroids[d] - tsne_centroids[d2])
                if dist < best_dist:
                    best_dist = dist
                    best_nn = d2
        dtype, stage, func = DEVICE_INFO[d]
        nn_type, nn_stage, nn_func = DEVICE_INFO[best_nn]
        print(f"M{d:02d}  {dtype:>5} {stage:>7} {tsne_centroids[d][0]:8.1f} {tsne_centroids[d][1]:8.1f} {tsne_spreads[d]:7.1f}   M{best_nn:02d} ({nn_stage}, {nn_func[:25]})")

    # =============================================
    # 7. PLOTS
    # =============================================
    out_dir = Path(ckpt).parent / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot A: t-SNE colored by Stage, Signal/Bias, W/L group, Operating Region ---
    stage_map = {d: DEVICE_INFO[d][1] for d in range(24)}
    stage_colors = {'Bias': '#95a5a6', 'Stage1': '#3498db', 'Stage2': '#e74c3c', 'Stage3': '#2ecc71'}

    signal_map = {
        0:'bias',1:'bias',2:'bias',3:'bias',4:'bias',5:'bias',6:'bias',7:'bias',
        8:'signal',9:'signal',10:'signal',11:'signal',
        12:'bias',13:'bias',14:'bias',15:'signal',16:'signal',17:'bias',18:'bias',
        19:'signal',20:'signal',21:'signal',22:'signal',23:'signal',
    }
    sig_colors = {'signal': '#e74c3c', 'bias': '#3498db'}

    wl_map = {
        0:'BIASCM_P',1:'BIASCM_P',2:'BIASCM_P',3:'BIASCM_P',
        4:'BIASCM_P',5:'BIASCM_P',6:'BIASCM_P',7:'BIASCM_P',
        8:'GM1',9:'GM1',
        10:'BIASCM_N',11:'BIASCM_N',12:'BIASCM_N',13:'BIASCM_N',14:'BIASCM_N',
        15:'BIASCM_N',16:'BIASCM_N',17:'BIASCM_N',18:'BIASCM_N',
        19:'GM2',20:'LOAD2',21:'LOAD2',22:'GMF2',23:'GM3',
    }
    wl_colors = {'BIASCM_P':'#3498db','GM1':'#e74c3c','BIASCM_N':'#2ecc71',
                 'GM2':'#9b59b6','LOAD2':'#f39c12','GMF2':'#1abc9c','GM3':'#e67e22'}

    region_names = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation', -1: 'Unknown'}
    region_colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', -1: '#95a5a6'}

    # NMOS/PMOS type map
    type_map = {d: DEVICE_INFO[d][0] for d in range(24)}
    type_colors = {'NMOS': '#3498db', 'PMOS': '#e74c3c'}

    fig, axes = plt.subplots(2, 3, figsize=(28, 18))

    # Panel 1: by stage
    for stage, color in stage_colors.items():
        mask_s = np.array([stage_map[d] == stage for d in dev_sub])
        if mask_s.any():
            axes[0,0].scatter(X_2d[mask_s,0], X_2d[mask_s,1], c=color, s=6, alpha=0.5, label=stage)
    axes[0,0].legend(markerscale=4, fontsize=11)
    axes[0,0].set_title('Colored by Circuit Stage', fontsize=13, fontweight='bold')

    # Panel 2: by signal/bias
    for role, color in sig_colors.items():
        mask_s = np.array([signal_map[d] == role for d in dev_sub])
        if mask_s.any():
            axes[0,1].scatter(X_2d[mask_s,0], X_2d[mask_s,1], c=color, s=6, alpha=0.5, label=role)
    axes[0,1].legend(markerscale=4, fontsize=11)
    axes[0,1].set_title('Colored by Signal Path vs Bias', fontsize=13, fontweight='bold')

    # Panel 3: by NMOS/PMOS
    for ttype, color in type_colors.items():
        mask_s = np.array([type_map[d] == ttype for d in dev_sub])
        if mask_s.any():
            axes[0,2].scatter(X_2d[mask_s,0], X_2d[mask_s,1], c=color, s=6, alpha=0.5, label=ttype)
    axes[0,2].legend(markerscale=4, fontsize=11)
    axes[0,2].set_title('Colored by NMOS / PMOS', fontsize=13, fontweight='bold')

    # Panel 4: by W/L group
    for wl, color in wl_colors.items():
        mask_s = np.array([wl_map[d] == wl for d in dev_sub])
        if mask_s.any():
            axes[1,0].scatter(X_2d[mask_s,0], X_2d[mask_s,1], c=color, s=6, alpha=0.5, label=wl)
    axes[1,0].legend(markerscale=4, fontsize=11)
    axes[1,0].set_title('Colored by W/L Group (Device Sizing)', fontsize=13, fontweight='bold')

    # Panel 5: by operating region
    for r_val in [2, 1, 0]:
        mask_s = reg_sub == r_val
        if mask_s.any():
            axes[1,1].scatter(X_2d[mask_s,0], X_2d[mask_s,1], c=region_colors[r_val],
                             s=6, alpha=0.5, label=region_names[r_val])
    axes[1,1].legend(markerscale=4, fontsize=11)
    axes[1,1].set_title('Colored by Operating Region', fontsize=13, fontweight='bold')

    # Panel 6: hide unused
    axes[1,2].axis('off')

    for ax in axes.flat:
        if ax.get_visible() and not ax._frameon == False:
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.grid(True, alpha=0.2)

    plt.suptitle('G+D+S MOSFET Embeddings — Tower w512', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'gds_tsne_by_role.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir / 'gds_tsne_by_role.png'}")

    # --- Plot B: t-SNE with device labels at centroids ---
    fig, ax = plt.subplots(figsize=(14, 12))
    device_cmap = plt.colormaps.get_cmap('tab20').resampled(24)
    for d in range(24):
        mask_d = dev_sub == d
        if mask_d.any():
            ax.scatter(X_2d[mask_d, 0], X_2d[mask_d, 1], c=[device_cmap(d)],
                      s=8, alpha=0.3)
    # Label centroids
    for d in sorted(tsne_centroids.keys()):
        dtype, stage, func = DEVICE_INFO[d]
        label = f"M{d}\n{stage}"
        ax.annotate(label, tsne_centroids[d], fontsize=7, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.set_title('G+D+S t-SNE with Device Labels', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / 'gds_tsne_labeled.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'gds_tsne_labeled.png'}")

    # --- Plot C: 24x24 cosine similarity heatmap ---
    fig, ax = plt.subplots(figsize=(14, 12))
    labels = [f"M{d} ({DEVICE_INFO[d][1][:2]})" for d in range(24)]
    im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(24))
    ax.set_yticks(range(24))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
    # Draw stage boundaries
    for boundary in [8, 12, 19, 22]:  # approximate stage boundaries in device ordering
        ax.axhline(boundary - 0.5, color='black', linewidth=1.5, linestyle='--')
        ax.axvline(boundary - 0.5, color='black', linewidth=1.5, linestyle='--')
    ax.set_title('Device Centroid Cosine Similarity (G+D+S Embeddings)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'gds_cosine_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'gds_cosine_heatmap.png'}")

    # --- Plot D: Intra-device region separation bar chart ---
    devices_with_multi_region = []
    sep_ratios = []
    sat_tri_dists = []
    sat_cut_dists = []
    intra_sats = []

    for d in range(24):
        mask = dev_ids == d
        X = embs[mask]
        r = regions[mask]
        sat_mask = r == 2
        tri_mask = r == 1
        cut_mask = r == 0

        if sat_mask.sum() < 10:
            continue

        sat_c = X[sat_mask].mean(axis=0)
        intra = np.mean(np.linalg.norm(X[sat_mask] - sat_c, axis=1))

        has_data = False
        st_d = 0
        sc_d = 0
        if tri_mask.sum() >= 10:
            tri_c = X[tri_mask].mean(axis=0)
            st_d = np.linalg.norm(sat_c - tri_c)
            has_data = True
        if cut_mask.sum() >= 10:
            cut_c = X[cut_mask].mean(axis=0)
            sc_d = np.linalg.norm(sat_c - cut_c)
            has_data = True

        if has_data:
            best = max(st_d, sc_d)
            devices_with_multi_region.append(d)
            sep_ratios.append(best / (intra + 1e-8))
            sat_tri_dists.append(st_d)
            sat_cut_dists.append(sc_d)
            intra_sats.append(intra)

    if devices_with_multi_region:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        x_pos = np.arange(len(devices_with_multi_region))
        labels = [f"M{d}\n{DEVICE_INFO[d][1]}" for d in devices_with_multi_region]

        # Top: distances
        w = 0.25
        axes[0].bar(x_pos - w, sat_tri_dists, w, label='Sat↔Triode dist', color='#3498db', alpha=0.8)
        axes[0].bar(x_pos, sat_cut_dists, w, label='Sat↔Cutoff dist', color='#e74c3c', alpha=0.8)
        axes[0].bar(x_pos + w, intra_sats, w, label='Intra-Sat spread', color='#2ecc71', alpha=0.8)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].legend(fontsize=10)
        axes[0].set_ylabel('Euclidean Distance')
        axes[0].set_title('Inter-Region Distance vs Intra-Region Spread', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Bottom: separation ratio
        colors = ['#2ecc71' if s > 1 else '#e74c3c' for s in sep_ratios]
        axes[1].bar(x_pos, sep_ratios, color=colors, alpha=0.8)
        axes[1].axhline(1.0, color='black', linewidth=1.5, linestyle='--', label='Separation threshold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, fontsize=8)
        axes[1].legend(fontsize=10)
        axes[1].set_ylabel('Separation Ratio (inter/intra)')
        axes[1].set_title('Operating Region Separation Ratio (>1 = separated, <1 = overlapping)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(out_dir / 'gds_region_separation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_dir / 'gds_region_separation.png'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
