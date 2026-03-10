#!/usr/bin/env python3
"""
Test accuracy of AM bridge equation against SPICE ground truth.

AM_physics = 20 * log10(gm_in * rout1 * gm_out * rout2)
where:
  gm_in = avg(gm_M1, gm_M2), rout1 = 1/(gds_M2 + gds_M4)
  gm_out = gm_M6, rout2 = 1/(gds_M6 + gds_M7)

Uses SPICE ground truth gm/gds values only (no model predictions).

Usage:
    python scripts/test_am_bridge_accuracy.py --dataset datasets/opamp_5k_ss_v1
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data_loading import load_prebatched_variant


def test_am_bridge(batches, device):
    """Test AM bridge equation using SPICE ground truth gm/gds."""

    am_spice_all = []
    am_physics_all = []
    gm_in_all = []
    rout1_all = []
    gm_out_all = []
    rout2_all = []

    for batch in batches:
        batch = batch.to(device)

        if not hasattr(batch, 'ac_am') or not hasattr(batch, 'ac_valid'):
            continue
        if not hasattr(batch, 'mosfet_topology_indices'):
            continue

        mosfet_info = batch.mosfet_info
        mosfet_gm = batch.mosfet_gm
        mosfet_gds = batch.mosfet_gds
        ptr = batch.ptr
        ac_am = batch.ac_am
        ac_valid = batch.ac_valid
        topo_indices = batch.mosfet_topology_indices

        num_mosfets = len(mosfet_info)
        num_graphs = len(ptr) - 1
        mosfets_per_graph = num_mosfets // num_graphs

        # Valid graphs: AC converged + all needed MOSFETs present
        valid_graph_indices = ac_valid.nonzero(as_tuple=True)[0]
        if len(valid_graph_indices) == 0:
            continue

        topo = topo_indices[valid_graph_indices]  # [N, 8]
        # M1=0, M2=1, M4=3, M6=5, M7=6
        needed = topo[:, [0, 1, 3, 5, 6]]
        topo_valid = (needed != -1).all(dim=1)
        if not topo_valid.any():
            continue

        valid_g = valid_graph_indices[topo_valid]
        topo_v = topo[topo_valid]
        mosfet_offsets = valid_g * mosfets_per_graph

        # Get SPICE gm/gds for each MOSFET
        m1_idx = mosfet_offsets + topo_v[:, 0]
        m2_idx = mosfet_offsets + topo_v[:, 1]
        m4_idx = mosfet_offsets + topo_v[:, 3]
        m6_idx = mosfet_offsets + topo_v[:, 5]
        m7_idx = mosfet_offsets + topo_v[:, 6]

        gm_m1 = mosfet_gm[m1_idx]
        gm_m2 = mosfet_gm[m2_idx]
        gds_m2 = mosfet_gds[m2_idx]
        gds_m4 = mosfet_gds[m4_idx]
        gm_m6 = mosfet_gm[m6_idx]
        gds_m6 = mosfet_gds[m6_idx]
        gds_m7 = mosfet_gds[m7_idx]

        # AM bridge equation
        gm_in = (gm_m1 + gm_m2) / 2.0
        rout1 = 1.0 / (gds_m2 + gds_m4).clamp(min=1e-15)
        gm_out = gm_m6
        rout2 = 1.0 / (gds_m6 + gds_m7).clamp(min=1e-15)

        gain = gm_in * rout1 * gm_out * rout2
        am_physics_db = 20.0 * torch.log10(gain.clamp(min=1e-10))

        # SPICE AM (ground truth)
        am_spice_db = ac_am[valid_g]

        am_spice_all.extend(am_spice_db.cpu().tolist())
        am_physics_all.extend(am_physics_db.cpu().tolist())
        gm_in_all.extend(gm_in.cpu().tolist())
        rout1_all.extend(rout1.cpu().tolist())
        gm_out_all.extend(gm_out.cpu().tolist())
        rout2_all.extend(rout2.cpu().tolist())

    return (np.array(am_spice_all), np.array(am_physics_all),
            np.array(gm_in_all), np.array(rout1_all),
            np.array(gm_out_all), np.array(rout2_all))


def main():
    parser = argparse.ArgumentParser(description='Test AM bridge equation accuracy')
    parser.add_argument('--dataset', type=str, default='datasets/opamp_5k_ss_v1')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    device = args.device

    split_dir = dataset_dir / args.split
    print(f"Loading {args.split} data from: {split_dir}")
    batches = load_prebatched_variant(split_dir, variant_id=0, device=device)
    print(f"  Loaded {len(batches)} batches")

    print(f"\nTesting AM bridge equation on SPICE ground truth...")
    am_spice, am_physics, gm_in, rout1, gm_out, rout2 = test_am_bridge(batches, device)
    n = len(am_spice)
    print(f"  Circuits with valid AC + topology: {n}")

    if n == 0:
        print("  No valid circuits found!")
        return

    # Error analysis
    err_db = am_physics - am_spice  # signed error in dB
    abs_err_db = np.abs(err_db)

    print(f"\n{'='*60}")
    print(f"  AM Bridge Equation Accuracy (n={n})")
    print(f"{'='*60}")
    print(f"\n  SPICE AM:    median={np.median(am_spice):.1f}dB  mean={np.mean(am_spice):.1f}dB  [{np.min(am_spice):.1f}, {np.max(am_spice):.1f}]")
    print(f"  Physics AM:  median={np.median(am_physics):.1f}dB  mean={np.mean(am_physics):.1f}dB  [{np.min(am_physics):.1f}, {np.max(am_physics):.1f}]")
    print(f"\n  Signed error (physics - spice):")
    print(f"    median={np.median(err_db):.2f}dB  mean={np.mean(err_db):.2f}dB  std={np.std(err_db):.2f}dB")
    print(f"    [{np.min(err_db):.2f}, {np.max(err_db):.2f}]")
    print(f"\n  Absolute error:")
    print(f"    median={np.median(abs_err_db):.2f}dB  mean={np.mean(abs_err_db):.2f}dB")
    print(f"    P75={np.percentile(abs_err_db, 75):.2f}dB  P90={np.percentile(abs_err_db, 90):.2f}dB  P95={np.percentile(abs_err_db, 95):.2f}dB")
    print(f"\n  Accuracy:")
    for thresh in [1, 2, 3, 5, 10]:
        pct = 100 * np.mean(abs_err_db < thresh)
        print(f"    <{thresh}dB: {pct:.1f}%")

    # Check for systematic bias
    print(f"\n  Bias check:")
    print(f"    Always overestimates: {100*np.mean(err_db > 0):.1f}%")
    print(f"    Always underestimates: {100*np.mean(err_db < 0):.1f}%")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
