#!/usr/bin/env python3
"""
Validate gm = 2*I_D / |Vgs - Vth| against SPICE gm, broken down by operating region
and Vov range. Also compares two Vov definitions:
  - Vov_volt = |Vgs - Vth| (from node voltages and SPICE Vth)
  - Vov_eq   = 2*I_D / gm  (from SPICE operating point)

Data is RAW (not normalized) from prebatched files.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data_loading import load_prebatched_variant

dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/opamp_3stage_fan_smc_v7_2k_nofil")
split = sys.argv[2] if len(sys.argv) > 2 else "val"

batches = load_prebatched_variant(dataset_dir / split, variant_id=0)

region_names = {0: "Cutoff", 1: "Triode", 2: "Saturation", -1: "Unknown"}

# Collect per-MOSFET data
all_gm_spice = []
all_gm_eq = []
all_regions = []
all_vov_volt = []   # |Vgs - Vth|
all_vov_eq = []     # 2*I_D / gm
all_id = []

for batch in batches:
    mosfet_info = batch.mosfet_info
    region_labels = batch.mosfet_region_labels
    vth = batch.mosfet_vth
    gm_spice = batch.mosfet_gm
    voltages = batch.node_voltage_targets
    currents = batch.node_current_targets

    ptr = batch.ptr
    num_mosfets = mosfet_info.shape[0]
    num_graphs = ptr.shape[0] - 1

    if hasattr(batch, 'mosfet_ptr'):
        mosfet_ptr = batch.mosfet_ptr
        mosfet_graph_idx = torch.zeros(num_mosfets, dtype=torch.long)
        for g in range(num_graphs):
            mosfet_graph_idx[mosfet_ptr[g]:mosfet_ptr[g+1]] = g
    else:
        mosfets_per_graph = num_mosfets // num_graphs
        mosfet_graph_idx = torch.arange(num_mosfets) // mosfets_per_graph

    node_offsets = ptr[mosfet_graph_idx]

    gate_net_idx = (mosfet_info[:, 3] + node_offsets).long()
    source_net_idx = (mosfet_info[:, 5] + node_offsets).long()
    drain_term_idx = (mosfet_info[:, 1] + node_offsets).long()

    V_gate = voltages[gate_net_idx]
    V_source = voltages[source_net_idx]
    Vgs = V_gate - V_source

    Id = currents[drain_term_idx].abs()
    is_nmos = mosfet_info[:, 6].bool()

    for i in range(num_mosfets):
        region = region_labels[i].item()
        gm_s = gm_spice[i].item()
        vth_i = vth[i].item()
        id_i = Id[i].item()
        vgs_i = Vgs[i].item()

        if gm_s <= 0 or abs(vth_i) < 1e-6 or id_i <= 0:
            continue

        # Vov from voltages
        if is_nmos[i]:
            vov_v = vgs_i - vth_i
        else:
            vov_v = -vgs_i - vth_i

        # Vov from equation (always positive for valid devices)
        vov_e = 2.0 * id_i / gm_s

        vov_abs = abs(vov_v)
        if vov_abs < 1e-6:
            continue

        gm_eq = 2.0 * id_i / vov_abs

        all_gm_spice.append(gm_s)
        all_gm_eq.append(gm_eq)
        all_regions.append(region)
        all_vov_volt.append(vov_v)
        all_vov_eq.append(vov_e)
        all_id.append(id_i)

all_gm_spice = np.array(all_gm_spice)
all_gm_eq = np.array(all_gm_eq)
all_regions = np.array(all_regions)
all_vov_volt = np.array(all_vov_volt)
all_vov_eq = np.array(all_vov_eq)
all_id = np.array(all_id)

print("=" * 75)
print(f"  gm EQUATION VALIDATION on {dataset_dir.name} ({split})")
print(f"  gm = 2*I_D / |Vgs - Vth|  vs  SPICE gm")
print("=" * 75)
print(f"  Total MOSFETs analyzed: {len(all_gm_spice)}")
print(f"  gm_spice range: [{all_gm_spice.min():.2e}, {all_gm_spice.max():.2e}] S")
print(f"  Id range:        [{all_id.min():.2e}, {all_id.max():.2e}] A")
print()

# --- SECTION 1: gm equation accuracy by region ---
log_ratio = np.log10(all_gm_eq) - np.log10(all_gm_spice)

for region_val in [2, 1, 0]:
    mask = all_regions == region_val
    if not mask.any():
        continue
    n = mask.sum()
    lr = log_ratio[mask]
    r = all_gm_eq[mask] / all_gm_spice[mask]

    pct_err = np.abs(all_gm_eq[mask] - all_gm_spice[mask]) / all_gm_spice[mask] * 100

    print(f"--- {region_names[region_val]} (n={n}) ---")
    print(f"  Ratio (eq/spice):  median={np.median(r):.3f}  p25={np.percentile(r,25):.3f}  p75={np.percentile(r,75):.3f}")
    print(f"  Median % error:    {np.median(pct_err):.1f}%")
    print(f"  <5% err: {100*np.mean(pct_err < 5):.1f}%  |  <10%: {100*np.mean(pct_err < 10):.1f}%  |  <20%: {100*np.mean(pct_err < 20):.1f}%  |  <50%: {100*np.mean(pct_err < 50):.1f}%")
    print(f"  Vov_volt range:    [{np.min(all_vov_volt[mask]):.3f}, {np.max(all_vov_volt[mask]):.3f}] V")
    print(f"  Id range:          [{all_id[mask].min():.2e}, {all_id[mask].max():.2e}] A")
    print()

# --- SECTION 2: gm equation accuracy by Vov range (saturation only) ---
sat_mask = all_regions == 2
if sat_mask.any():
    print("=" * 75)
    print("  gm EQUATION BY Vov RANGE (saturation only)")
    print("=" * 75)

    sat_vov = np.abs(all_vov_volt[sat_mask])
    sat_lr = log_ratio[sat_mask]
    sat_gm_ratio = all_gm_eq[sat_mask] / all_gm_spice[sat_mask]

    vov_ranges = [
        (0, 0.05, "<50 mV"),
        (0.05, 0.1, "50-100 mV"),
        (0.1, 0.2, "100-200 mV"),
        (0.2, 0.5, "200-500 mV"),
        (0.5, float('inf'), ">500 mV"),
    ]

    sat_gm_spice = all_gm_spice[sat_mask]
    sat_gm_eq = all_gm_eq[sat_mask]

    for lo, hi, label in vov_ranges:
        rmask = (sat_vov >= lo) & (sat_vov < hi)
        n = rmask.sum()
        if n == 0:
            print(f"  {label:>12s}: n=0")
            continue
        r = sat_gm_ratio[rmask]
        pct_err = np.abs(sat_gm_eq[rmask] - sat_gm_spice[rmask]) / sat_gm_spice[rmask] * 100
        print(f"  {label:>12s}: n={n:5d}  median_ratio={np.median(r):.3f}  median_err={np.median(pct_err):.1f}%  "
              f"<5%: {100*np.mean(pct_err < 5):.0f}%  <10%: {100*np.mean(pct_err < 10):.0f}%  "
              f"<20%: {100*np.mean(pct_err < 20):.0f}%")

    print()

# --- SECTION 3: Vov comparison (saturation only) ---
if sat_mask.any():
    print("=" * 75)
    print("  Vov COMPARISON: Vov_volt (|Vgs-Vth|) vs Vov_eq (2*I_D/gm)")
    print("  (saturation MOSFETs only)")
    print("=" * 75)

    vov_v = all_vov_volt[sat_mask]
    vov_e = all_vov_eq[sat_mask]

    # Use absolute values for comparison
    vov_v_abs = np.abs(vov_v)

    # Filter out near-zero
    valid = (vov_v_abs > 1e-6) & (vov_e > 1e-6)
    vv = vov_v_abs[valid]
    ve = vov_e[valid]

    ratio = vv / ve
    diff_mv = (vv - ve) * 1000  # mV
    rel_err = np.abs(vv - ve) / ve

    print(f"  Total pairs: {valid.sum()}")
    print(f"  Vov_volt range:  [{vv.min():.4f}, {vv.max():.4f}] V")
    print(f"  Vov_eq range:    [{ve.min():.4f}, {ve.max():.4f}] V")
    print()
    print(f"  Ratio (volt/eq): median={np.median(ratio):.4f}  mean={np.mean(ratio):.4f}")
    print(f"                   p5={np.percentile(ratio,5):.4f}  p95={np.percentile(ratio,95):.4f}")
    print(f"  Diff (mV):       median={np.median(diff_mv):.2f}  mean={np.mean(diff_mv):.2f}")
    print(f"                   p5={np.percentile(diff_mv,5):.2f}  p95={np.percentile(diff_mv,95):.2f}")
    print(f"  Rel error:       median={np.median(rel_err):.4f}  mean={np.mean(rel_err):.4f}")
    print(f"                   p5={np.percentile(rel_err,5):.4f}  p95={np.percentile(rel_err,95):.4f}")
    print()

    # Breakdown by Vov range
    print("  By Vov_eq range:")
    for lo, hi, label in vov_ranges:
        rmask = (ve >= lo) & (ve < hi)
        n = rmask.sum()
        if n == 0:
            print(f"    {label:>12s}: n=0")
            continue
        r = ratio[rmask]
        d = diff_mv[rmask]
        re = rel_err[rmask]
        print(f"    {label:>12s}: n={n:5d}  ratio_median={np.median(r):.4f}  "
              f"diff_median={np.median(d):+.2f}mV  rel_err={np.median(re):.4f}")

    print()

    # How many have Vov_volt negative (device not really in saturation?)
    neg_vov = np.sum(vov_v < 0)
    print(f"  Negative Vov_volt (region=sat but Vgs-Vth<0): {neg_vov} / {len(vov_v)} "
          f"({100*neg_vov/len(vov_v):.1f}%)")

print()
print("=" * 75)
print("  SUMMARY")
print("=" * 75)
if sat_mask.any():
    pct_err_sat = np.abs(all_gm_eq[sat_mask] - all_gm_spice[sat_mask]) / all_gm_spice[sat_mask] * 100
    mr = np.median(all_gm_eq[sat_mask] / all_gm_spice[sat_mask])
    print(f"  gm equation (sat): median ratio={mr:.3f}, median err={np.median(pct_err_sat):.1f}%")
    print(f"    <5%: {100*np.mean(pct_err_sat < 5):.0f}%  <10%: {100*np.mean(pct_err_sat < 10):.0f}%  <20%: {100*np.mean(pct_err_sat < 20):.0f}%  <50%: {100*np.mean(pct_err_sat < 50):.0f}%")

    vv = np.abs(all_vov_volt[sat_mask])
    ve = all_vov_eq[sat_mask]
    valid = (vv > 1e-6) & (ve > 1e-6)
    if valid.any():
        vov_pct_err = np.abs(vv[valid] - ve[valid]) / ve[valid] * 100
        print(f"  Vov consistency:   median ratio={np.median(vv[valid]/ve[valid]):.4f}, median err={np.median(vov_pct_err):.1f}%")
        print(f"    <5%: {100*np.mean(vov_pct_err < 5):.0f}%  <10%: {100*np.mean(vov_pct_err < 10):.0f}%  <20%: {100*np.mean(vov_pct_err < 20):.0f}%  <50%: {100*np.mean(vov_pct_err < 50):.0f}%")
print()
