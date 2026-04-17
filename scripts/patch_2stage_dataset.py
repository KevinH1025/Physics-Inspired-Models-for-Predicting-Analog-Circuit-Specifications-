#!/usr/bin/env python3
"""Patch 2-stage dataset to add missing physics loss attributes.

Adds: mosfet_vth, node_mosfet_vth, mosfet_drain_mask, node_log_gm, node_log_gds
"""

import pickle
import math
import torch
from pathlib import Path


def patch_sample(sample):
    """Add missing attributes to a single sample's graph."""
    g = sample['graph']
    mi = g.mosfet_info  # [M, 7]
    gm = g.mosfet_gm    # [M]
    gds = g.mosfet_gds   # [M]
    mosfet_regions = sample['specs']['_mosfet_regions']
    node_names = g.node_names
    num_nodes = g.x.shape[0]
    num_mosfets = mi.shape[0]

    # --- mosfet_vth from SPICE ---
    mosfet_vth = torch.zeros(num_mosfets, dtype=torch.float)
    for i in range(num_mosfets):
        drain_t = mi[i, 1].item()
        drain_name = node_names[drain_t]
        dev_name = drain_name.split('_')[0].lower()
        if dev_name in mosfet_regions and mosfet_regions[dev_name].get('vth') is not None:
            mosfet_vth[i] = mosfet_regions[dev_name]['vth']
    g.mosfet_vth = mosfet_vth

    # --- node_mosfet_vth: scatter to drain terminals ---
    node_mosfet_vth = torch.zeros(num_nodes, dtype=torch.float)
    for i in range(num_mosfets):
        drain_t = mi[i, 1].item()
        node_mosfet_vth[drain_t] = mosfet_vth[i].item()
    g.node_mosfet_vth = node_mosfet_vth

    # --- mosfet_gt_vov from SPICE (|Vgs| - |Vth|) ---
    mosfet_gt_vov = torch.zeros(num_mosfets, dtype=torch.float)
    for i in range(num_mosfets):
        drain_t = mi[i, 1].item()
        drain_name = node_names[drain_t]
        dev_name = drain_name.split('_')[0].lower()
        if dev_name in mosfet_regions:
            vgs = mosfet_regions[dev_name].get('vgs', 0.0)
            vth = mosfet_regions[dev_name].get('vth', 0.0)
            # GT Vov = |Vgs| - |Vth| (works for both NMOS and PMOS)
            mosfet_gt_vov[i] = abs(vgs) - abs(vth)
    g.mosfet_gt_vov = mosfet_gt_vov

    # --- mosfet_drain_mask, node_log_gm, node_log_gds ---
    mosfet_drain_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_log_gm = torch.zeros(num_nodes, dtype=torch.float)
    node_log_gds = torch.zeros(num_nodes, dtype=torch.float)
    for i in range(num_mosfets):
        drain_t = mi[i, 1].item()
        gm_val = gm[i].item()
        gds_val = gds[i].item()
        if gm_val > 1e-12:
            mosfet_drain_mask[drain_t] = True
            node_log_gm[drain_t] = math.log10(gm_val)
            node_log_gds[drain_t] = math.log10(max(gds_val, 1e-20))
    g.mosfet_drain_mask = mosfet_drain_mask
    g.node_log_gm = node_log_gm
    g.node_log_gds = node_log_gds


def patch_file(path):
    print(f"Patching {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for sample in data:
        patch_sample(sample)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Done. {len(data)} samples patched.")


if __name__ == '__main__':
    ds_dir = Path('datasets/opamp_2stage_5k_ss')
    for name in ['dataset.pkl', 'dataset_train.pkl', 'dataset_val.pkl', 'dataset_test.pkl']:
        path = ds_dir / name
        if path.exists():
            patch_file(path)
        else:
            print(f"  Skipping {path} (not found)")

    # Verify
    print("\nVerification:")
    with open(ds_dir / 'dataset_train.pkl', 'rb') as f:
        data = pickle.load(f)
    g = data[0]['graph']
    print(f"  node_mosfet_vth: {g.node_mosfet_vth.shape}, nonzero={g.node_mosfet_vth.nonzero().shape[0]}")
    print(f"  mosfet_vth: {g.mosfet_vth}")
    print(f"  mosfet_drain_mask: sum={g.mosfet_drain_mask.sum().item()}")
    print(f"  node_log_gm sample: {g.node_log_gm[g.mosfet_drain_mask][:3]}")
    print(f"  mosfet_gt_vov: {g.mosfet_gt_vov}")
    print(f"  mosfet_region_labels: {g.mosfet_region_labels}")
    sat = g.mosfet_region_labels == 2
    print(f"  sat devices: {sat.sum().item()}, gt_vov >= 0.1: {(g.mosfet_gt_vov[sat] >= 0.1).sum().item()}")
