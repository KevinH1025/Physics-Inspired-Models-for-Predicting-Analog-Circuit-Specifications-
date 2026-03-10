#!/usr/bin/env python3
"""
Validate CLM saturation equation: gm*Vov + 2*gds*Vds = 2*Id
on the 3-stage dataset using ground truth SPICE values.

Also validates 0th-order: gm = 2*Id/Vov
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data_loading import load_prebatched_variant

dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/opamp_3stage_fan_smc_v7_2k_nofil")
min_vov = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15

batches = load_prebatched_variant(dataset_dir / "val", variant_id=0)

all_0th = []  # log10(2*Id/Vov) - log10(gm_spice)
all_clm = []  # log10(gm*Vov + 2*gds*Vds) - log10(2*Id)
all_vov = []

for batch in batches:
    mosfet_info = batch.mosfet_info
    region_labels = batch.mosfet_region_labels
    vth = batch.mosfet_vth
    gm_spice = batch.mosfet_gm
    gds_spice = batch.mosfet_gds if hasattr(batch, 'mosfet_gds') else None

    voltages = batch.node_voltage_targets
    currents = batch.node_current_targets

    ptr = batch.ptr
    num_mosfets = mosfet_info.shape[0]
    num_graphs = ptr.shape[0] - 1
    mosfets_per_graph = num_mosfets // num_graphs

    mosfet_graph_idx = torch.arange(num_mosfets) // mosfets_per_graph
    node_offsets = ptr[mosfet_graph_idx]

    gate_net_idx = (mosfet_info[:, 3] + node_offsets).long()
    drain_net_idx = (mosfet_info[:, 4] + node_offsets).long()
    source_net_idx = (mosfet_info[:, 5] + node_offsets).long()
    drain_term_idx = (mosfet_info[:, 1] + node_offsets).long()

    V_gate = voltages[gate_net_idx]
    V_source = voltages[source_net_idx]
    V_drain = voltages[drain_net_idx]
    Vgs = V_gate - V_source
    Id = currents[drain_term_idx].abs()
    is_nmos = mosfet_info[:, 6].bool()

    for i in range(num_mosfets):
        if region_labels[i].item() != 2:  # saturation only
            continue
        gm_s = gm_spice[i].item()
        vth_i = vth[i].item()
        id_i = Id[i].item()
        vgs_i = Vgs[i].item()

        if gm_s <= 0 or abs(vth_i) < 1e-6 or id_i <= 0:
            continue

        if is_nmos[i]:
            vov = vgs_i - vth_i
        else:
            vov = -vgs_i - vth_i

        if vov < min_vov:
            continue

        # 0th order: gm = 2*Id/Vov
        gm_eq = 2.0 * id_i / vov
        err_0th = np.log10(gm_eq) - np.log10(gm_s)
        all_0th.append(err_0th)
        all_vov.append(vov)

        # CLM: gm*Vov + 2*gds*Vds = 2*Id
        if gds_spice is not None:
            gds_s = gds_spice[i].item()
            if gds_s > 0:
                vds_raw = V_drain[i].item() - V_source[i].item()
                vds = abs(vds_raw) if is_nmos[i] else abs(-vds_raw)
                lhs = gm_s * vov + 2.0 * gds_s * vds
                rhs = 2.0 * id_i
                if lhs > 0 and rhs > 0:
                    err_clm = np.log10(lhs) - np.log10(rhs)
                    all_clm.append(err_clm)

all_0th = np.array(all_0th)
all_vov = np.array(all_vov)
all_clm = np.array(all_clm) if all_clm else np.array([])

print(f"Dataset: {dataset_dir}")
print(f"Min Vov: {min_vov}")
print(f"Saturation devices (Vov >= {min_vov}): {len(all_0th)}")
print()

print("=== 0th Order: gm = 2*Id/Vov ===")
print(f"  MSE(log10):    {np.mean(all_0th**2):.4f}")
print(f"  MAE(log10):    {np.mean(np.abs(all_0th)):.4f}")
print(f"  Mean bias:     {np.mean(all_0th):.4f}")
print(f"  Within 1.5x:   {100*np.mean(np.abs(all_0th) < np.log10(1.5)):.1f}%")
print(f"  Within 2x:     {100*np.mean(np.abs(all_0th) < np.log10(2.0)):.1f}%")
print()

if len(all_clm) > 0:
    print(f"=== CLM: gm*Vov + 2*gds*Vds = 2*Id === (n={len(all_clm)})")
    print(f"  MSE(log10):    {np.mean(all_clm**2):.4f}")
    print(f"  MAE(log10):    {np.mean(np.abs(all_clm)):.4f}")
    print(f"  Mean bias:     {np.mean(all_clm):.4f}")
    print(f"  Within 1.5x:   {100*np.mean(np.abs(all_clm) < np.log10(1.5)):.1f}%")
    print(f"  Within 2x:     {100*np.mean(np.abs(all_clm) < np.log10(2.0)):.1f}%")
else:
    print("=== CLM: No mosfet_gds data available ===")
