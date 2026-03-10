#!/usr/bin/env python3
"""
Test accuracy of physics equations against SPICE ground truth.

Equations tested:
  Triode (region=1):
    Eq1: gm = I_DS / (Vov - VDS/2)
    Eq2: gds = (I_DS / VDS) * (Vov - VDS) / (Vov - VDS/2)
    Eq3: gm * (Vov - VDS) = gds * VDS   (self-consistency, no current needed)

  Saturation (region=2):
    Sat_gm: gm = 2 * I_DS / Vov

Reports accuracy tables by Vov filter threshold:
  | Vov range | n | <5% err | <10% err | <20% err | <50% err |

Usage:
    python scripts/test_equation_accuracy.py --dataset datasets/opamp_5k_ss_v1
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data_loading import load_prebatched_variant

REGION_NAMES = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation'}


def collect_data(batches, device):
    """Collect per-MOSFET SPICE ground truth values across all batches."""
    all_data = {
        'region': [], 'Vov': [], 'Vds': [], 'Vgs': [],
        'gm': [], 'gds': [], 'Id': [], 'Id_valid': [],
        'is_nmos': [], 'Vth': [],
    }

    for batch in batches:
        batch = batch.to(device)

        mosfet_info = batch.mosfet_info
        region_labels = batch.mosfet_region_labels
        mosfet_gm = batch.mosfet_gm
        mosfet_gds = batch.mosfet_gds
        mosfet_vth = batch.mosfet_vth
        v_targets = batch.node_voltage_targets
        ptr = batch.ptr
        num_mosfets = mosfet_info.shape[0]
        num_graphs = ptr.shape[0] - 1
        mosfets_per_graph = num_mosfets // num_graphs

        # Global node indices
        mosfet_graph_idx = torch.arange(num_mosfets, device=device) // mosfets_per_graph
        node_offsets = ptr[mosfet_graph_idx]

        gate_net_idx = mosfet_info[:, 3].long() + node_offsets
        drain_net_idx = mosfet_info[:, 4].long() + node_offsets
        source_net_idx = mosfet_info[:, 5].long() + node_offsets
        drain_term_idx = mosfet_info[:, 1].long() + node_offsets
        is_nmos = mosfet_info[:, 6].bool()

        # Raw voltages from SPICE
        Vg = v_targets[gate_net_idx]
        Vd = v_targets[drain_net_idx]
        Vs = v_targets[source_net_idx]

        # NMOS: Vgs = Vg - Vs, Vds = Vd - Vs
        # PMOS: flip -> Vsg = Vs - Vg, Vsd = Vs - Vd (both positive when on)
        Vgs = torch.where(is_nmos, Vg - Vs, Vs - Vg)
        Vds = torch.where(is_nmos, Vd - Vs, Vs - Vd)
        Vth = mosfet_vth.abs()
        Vov = Vgs - Vth

        # Current
        Id = batch.node_current_targets[drain_term_idx].abs()
        Id_valid = batch.has_current_mask[drain_term_idx]

        all_data['region'].append(region_labels.cpu())
        all_data['Vov'].append(Vov.cpu())
        all_data['Vds'].append(Vds.cpu())
        all_data['Vgs'].append(Vgs.cpu())
        all_data['gm'].append(mosfet_gm.cpu())
        all_data['gds'].append(mosfet_gds.cpu())
        all_data['Id'].append(Id.cpu())
        all_data['Id_valid'].append(Id_valid.cpu())
        all_data['is_nmos'].append(is_nmos.cpu())
        all_data['Vth'].append(Vth.cpu())

    # Concatenate all
    return {k: torch.cat(v) for k, v in all_data.items()}


def compute_pct_error(predicted, target):
    """Relative percentage error: |pred - target| / |target| * 100."""
    return (predicted - target).abs() / target.abs().clamp(min=1e-20) * 100


def print_accuracy_table(name, formula, pct_errors, Vov, region_mask, thresholds):
    """Print accuracy table for one equation across Vov filter thresholds."""
    print(f"\n  {name}: {formula}")
    print(f"  {'Vov range':<14s} {'n':>6s} {'<5%':>7s} {'<10%':>7s} {'<20%':>7s} {'<50%':>7s} {'median':>8s}")
    print(f"  {'-'*58}")

    for label, vov_lo, vov_hi in thresholds:
        if vov_lo is None and vov_hi is None:
            vov_mask = region_mask
        elif vov_hi is None:
            vov_mask = region_mask & (Vov >= vov_lo)
        else:
            vov_mask = region_mask & (Vov >= vov_lo) & (Vov < vov_hi)

        # Also need valid pct_errors (not NaN)
        valid = vov_mask & ~torch.isnan(pct_errors)
        n = valid.sum().item()
        if n == 0:
            print(f"  {label:<14s} {0:>6d} {'--':>7s} {'--':>7s} {'--':>7s} {'--':>7s} {'--':>8s}")
            continue

        errs = pct_errors[valid].numpy()
        lt5 = (errs < 5).sum() / n * 100
        lt10 = (errs < 10).sum() / n * 100
        lt20 = (errs < 20).sum() / n * 100
        lt50 = (errs < 50).sum() / n * 100
        med = np.median(errs)

        print(f"  {label:<14s} {n:>6d} {lt5:>6.1f}% {lt10:>6.1f}% {lt20:>6.1f}% {lt50:>6.1f}% {med:>7.1f}%")


def analyse(data):
    """Compute equation errors and print accuracy tables."""
    region = data['region']
    Vov = data['Vov']
    Vds = data['Vds']
    gm = data['gm']
    gds = data['gds']
    Id = data['Id']
    Id_valid = data['Id_valid']

    total = region.shape[0]
    for r in [2, 1, 0]:
        n = (region == r).sum().item()
        print(f"  {REGION_NAMES[r]:>12s}: {n:6d} ({n/total*100:.1f}%)")
    print(f"  {'Total':>12s}: {total:6d}")

    # Base validity: positive gm/gds, valid current
    base_valid = (gm > 1e-15) & (gds > 1e-15) & Id_valid & (Id > 1e-15)

    # ================================================================
    # TRIODE EQUATIONS
    # ================================================================
    triode = (region == 1) & base_valid
    n_triode = triode.sum().item()

    print(f"\n{'='*65}")
    print(f"  TRIODE REGION (n={n_triode} with valid gm/gds/Id)")
    print(f"{'='*65}")

    triode_thresholds = [
        ("All triode",  None, None),
        ("> 50mV",      0.05, None),
        ("> 100mV",     0.10, None),
        ("> 150mV",     0.15, None),
        ("> 200mV",     0.20, None),
        ("200-400mV",   0.20, 0.40),
        ("300-600mV",   0.30, 0.60),
    ]

    # Eq1: gm = I_DS / (Vov - VDS/2)
    denom1 = Vov - Vds / 2
    safe1 = triode & (denom1.abs() > 1e-6)
    eq1_pct = torch.full_like(Vov, float('nan'))
    gm_eq1 = Id[safe1] / denom1[safe1]
    eq1_pct[safe1] = compute_pct_error(gm_eq1, gm[safe1])

    print_accuracy_table(
        "Eq1 (triode)", "gm = I_DS / (Vov - Vds/2)",
        eq1_pct, Vov, triode & safe1, triode_thresholds,
    )

    # Eq2: gds = (I_DS/VDS) * (Vov-VDS)/(Vov-VDS/2)
    denom2a = Vds.abs().clamp(min=1e-9)
    denom2b = (Vov - Vds / 2).abs().clamp(min=1e-9)
    num2 = (Vov - Vds).abs()
    safe2 = triode & (Vds.abs() > 1e-6) & ((Vov - Vds / 2).abs() > 1e-6) & (num2 > 1e-6)
    eq2_pct = torch.full_like(Vov, float('nan'))
    gds_eq2 = (Id[safe2] / denom2a[safe2]) * (num2[safe2] / denom2b[safe2])
    eq2_pct[safe2] = compute_pct_error(gds_eq2, gds[safe2])

    print_accuracy_table(
        "Eq2 (triode)", "gds = (I_DS/Vds)*(Vov-Vds)/(Vov-Vds/2)",
        eq2_pct, Vov, triode & safe2, triode_thresholds,
    )

    # Eq3: gm * (Vov - VDS) = gds * VDS  (self-consistency)
    safe3 = triode & ((Vov - Vds).abs() > 1e-6) & (Vds.abs() > 1e-6)
    eq3_pct = torch.full_like(Vov, float('nan'))
    lhs3 = gm[safe3] * (Vov[safe3] - Vds[safe3])
    rhs3 = gds[safe3] * Vds[safe3]
    # Relative error using max(|lhs|, |rhs|) as denominator
    denom3 = torch.max(lhs3.abs(), rhs3.abs()).clamp(min=1e-20)
    eq3_pct[safe3] = ((lhs3 - rhs3).abs() / denom3) * 100

    print_accuracy_table(
        "Eq3 (triode)", "gm*(Vov-Vds) = gds*Vds  [self-consistency]",
        eq3_pct, Vov, triode & safe3, triode_thresholds,
    )

    # ================================================================
    # SATURATION EQUATIONS
    # ================================================================
    sat = (region == 2) & base_valid
    n_sat = sat.sum().item()

    print(f"\n{'='*65}")
    print(f"  SATURATION REGION (n={n_sat} with valid gm/gds/Id)")
    print(f"{'='*65}")

    sat_thresholds = [
        ("All sat",     None, None),
        ("> 50mV",      0.05, None),
        ("> 100mV",     0.10, None),
        ("> 150mV",     0.15, None),
        ("> 200mV",     0.20, None),
        ("200-400mV",   0.20, 0.40),
        ("300-600mV",   0.30, 0.60),
    ]

    # Sat_gm: gm = 2 * I_DS / Vov
    safe_sat = sat & (Vov.abs() > 1e-6)
    sat_gm_pct = torch.full_like(Vov, float('nan'))
    gm_sat = 2 * Id[safe_sat] / Vov[safe_sat]
    sat_gm_pct[safe_sat] = compute_pct_error(gm_sat, gm[safe_sat])

    print_accuracy_table(
        "Sat_gm", "gm = 2 * I_DS / Vov",
        sat_gm_pct, Vov, sat & safe_sat, sat_thresholds,
    )

    # ================================================================
    # CROSS-CHECKS (wrong equation on wrong region)
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  CROSS-CHECKS (wrong equation on wrong region)")
    print(f"{'='*65}")

    # Triode Eq1 applied to saturation (should fail)
    safe_eq1_sat = sat & (denom1.abs() > 1e-6)
    eq1_sat_pct = torch.full_like(Vov, float('nan'))
    if safe_eq1_sat.any():
        gm_eq1_sat = Id[safe_eq1_sat] / denom1[safe_eq1_sat]
        eq1_sat_pct[safe_eq1_sat] = compute_pct_error(gm_eq1_sat, gm[safe_eq1_sat])

    print_accuracy_table(
        "Triode Eq1 on sat", "gm = I_DS / (Vov - Vds/2)  [should fail]",
        eq1_sat_pct, Vov, sat & safe_eq1_sat, sat_thresholds,
    )

    # Triode Eq2 applied to saturation (should fail)
    safe_eq2_sat = sat & (Vds.abs() > 1e-6) & ((Vov - Vds / 2).abs() > 1e-6) & ((Vov - Vds).abs() > 1e-6)
    eq2_sat_pct = torch.full_like(Vov, float('nan'))
    if safe_eq2_sat.any():
        gds_eq2_sat = (Id[safe_eq2_sat] / Vds[safe_eq2_sat].abs().clamp(min=1e-9)) * \
                      ((Vov[safe_eq2_sat] - Vds[safe_eq2_sat]).abs() / (Vov[safe_eq2_sat] - Vds[safe_eq2_sat] / 2).abs().clamp(min=1e-9))
        eq2_sat_pct[safe_eq2_sat] = compute_pct_error(gds_eq2_sat, gds[safe_eq2_sat])

    print_accuracy_table(
        "Triode Eq2 on sat", "gds = (I/Vds)*(Vov-Vds)/(Vov-Vds/2)  [should fail]",
        eq2_sat_pct, Vov, sat & safe_eq2_sat, sat_thresholds,
    )

    # Triode Eq3 applied to saturation (should fail)
    safe_eq3_sat = sat & ((Vov - Vds).abs() > 1e-6) & (Vds.abs() > 1e-6)
    eq3_sat_pct = torch.full_like(Vov, float('nan'))
    if safe_eq3_sat.any():
        lhs3_sat = gm[safe_eq3_sat] * (Vov[safe_eq3_sat] - Vds[safe_eq3_sat])
        rhs3_sat = gds[safe_eq3_sat] * Vds[safe_eq3_sat]
        d3_sat = torch.max(lhs3_sat.abs(), rhs3_sat.abs()).clamp(min=1e-20)
        eq3_sat_pct[safe_eq3_sat] = ((lhs3_sat - rhs3_sat).abs() / d3_sat) * 100

    print_accuracy_table(
        "Triode Eq3 on sat", "gm*(Vov-Vds) = gds*Vds  [should fail]",
        eq3_sat_pct, Vov, sat & safe_eq3_sat, sat_thresholds,
    )

    # Sat_gm applied to triode (should fail)
    safe_sat_tri = triode & (Vov.abs() > 1e-6)
    sat_on_tri_pct = torch.full_like(Vov, float('nan'))
    if safe_sat_tri.any():
        gm_sat_tri = 2 * Id[safe_sat_tri] / Vov[safe_sat_tri]
        sat_on_tri_pct[safe_sat_tri] = compute_pct_error(gm_sat_tri, gm[safe_sat_tri])

    print_accuracy_table(
        "Sat_gm on triode", "gm = 2*I_DS/Vov  [should fail]",
        sat_on_tri_pct, Vov, triode & safe_sat_tri, triode_thresholds,
    )


def main():
    parser = argparse.ArgumentParser(description='Test physics equation accuracy')
    parser.add_argument('--dataset', type=str, default='datasets/opamp_5k_ss_v1')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    device = args.device

    all_batches = []
    for split in args.splits:
        split_dir = dataset_dir / split
        print(f"Loading {split} from: {split_dir}")
        batches = load_prebatched_variant(split_dir, variant_id=0, device=device)
        print(f"  {len(batches)} batches")
        all_batches.extend(batches)

    print(f"\nCollecting SPICE ground truth data...")
    data = collect_data(all_batches, device)
    print(f"  Total MOSFETs: {data['region'].shape[0]}")

    print(f"\n  Region distribution:")
    analyse(data)

    print()


if __name__ == '__main__':
    main()
