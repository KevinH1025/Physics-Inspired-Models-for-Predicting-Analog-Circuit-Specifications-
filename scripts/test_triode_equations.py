#!/usr/bin/env python3
"""
Test accuracy of triode physics equations against SPICE ground truth.

For MOSFETs in triode region (region_label == 1), compare:
  Eq1: gm = I_DS / (Vov - VDS/2)
  Eq2: gds = (I_DS / VDS) * (Vov - VDS) / (Vov - VDS/2)
  Eq3: gm * (Vov - VDS) = gds * VDS   (self-consistency)

Also tests the existing saturation equation for comparison:
  Sat: gm = 2 * I_DS / Vov

Uses SPICE ground truth values only (no model predictions).

Usage:
    python scripts/test_triode_equations.py --dataset datasets/opamp_5k_ss_v1
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data_loading import load_prebatched_variant

REGION_NAMES = {0: 'Cutoff', 1: 'Triode', 2: 'Saturation'}


def test_equations(batches, device):
    """Test triode and saturation equations using SPICE ground truth."""

    results = {r: defaultdict(list) for r in [0, 1, 2]}

    for batch in batches:
        batch = batch.to(device)

        mosfet_info = batch.mosfet_info          # [num_mosfets, 7]
        region_labels = batch.mosfet_region_labels # [num_mosfets]
        mosfet_gm = batch.mosfet_gm               # [num_mosfets] linear Siemens
        mosfet_gds = batch.mosfet_gds              # [num_mosfets] linear Siemens
        mosfet_vth = batch.mosfet_vth              # [num_mosfets] Volts
        v_targets = batch.node_voltage_targets     # [num_nodes] raw Volts
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
        Vth = mosfet_vth.to(device).abs()
        Vov = Vgs - Vth

        # Ground truth gm/gds from SPICE
        gm_true = mosfet_gm.to(device)
        gds_true = mosfet_gds.to(device)

        # Ground truth I_DS from node_current_targets
        has_current = hasattr(batch, 'node_current_targets') and hasattr(batch, 'has_current_mask')
        if has_current:
            # node_current_targets stores raw |I_D| in Amps (pre-normalization)
            Id_true = batch.node_current_targets[drain_term_idx].abs()
            Id_valid = batch.has_current_mask[drain_term_idx]
        else:
            Id_true = None
            Id_valid = None

        for r in [0, 1, 2]:
            mask = (region_labels.to(device) == r)
            if not mask.any():
                continue

            gm_r = gm_true[mask]
            gds_r = gds_true[mask]
            Vov_r = Vov[mask]
            Vds_r = Vds[mask]

            # Filter: valid voltages, positive gm/gds, avoid div-by-zero
            valid = (gm_r > 1e-15) & (gds_r > 1e-15) & (Vds_r.abs() > 1e-6)
            if not valid.any():
                continue

            gm_v = gm_r[valid]
            gds_v = gds_r[valid]
            Vov_v = Vov_r[valid]
            Vds_v = Vds_r[valid]
            if Id_true is not None:
                Id_v = Id_true[mask][valid]
                Id_ok = Id_valid[mask][valid]  # has_current_mask filter
            else:
                Id_v = None
                Id_ok = None

            results[r]['count'].append(valid.sum().item())
            results[r]['Vov'].extend(Vov_v.cpu().tolist())
            results[r]['Vds'].extend(Vds_v.cpu().tolist())
            results[r]['gm'].extend(gm_v.cpu().tolist())
            results[r]['gds'].extend(gds_v.cpu().tolist())

            # ================================================================
            # Eq3: gm * (Vov - VDS) = gds * VDS  (self-consistency)
            # ================================================================
            lhs3 = gm_v * (Vov_v - Vds_v)
            rhs3 = gds_v * Vds_v
            denom3 = torch.max(lhs3.abs(), rhs3.abs()).clamp(min=1e-20)
            eq3_rel_err = ((lhs3 - rhs3).abs() / denom3) * 100
            eq3_ratio = lhs3 / rhs3.clamp(min=1e-20)
            results[r]['eq3_rel_err'].extend(eq3_rel_err.cpu().tolist())
            results[r]['eq3_ratio'].extend(eq3_ratio.cpu().tolist())

            # Log-space error
            safe3 = ((Vov_v - Vds_v).abs() > 1e-6) & (Vds_v.abs() > 1e-6)
            if safe3.any():
                log_lhs = torch.log10(gm_v[safe3].abs()) + torch.log10((Vov_v[safe3] - Vds_v[safe3]).abs())
                log_rhs = torch.log10(gds_v[safe3].abs()) + torch.log10(Vds_v[safe3].abs())
                results[r]['eq3_log_err'].extend((log_lhs - log_rhs).abs().cpu().tolist())

            # ================================================================
            # Eq1: gm = I_DS / (Vov - VDS/2)  (gm-current)
            # ================================================================
            if Id_v is not None and Id_ok is not None:
                denom1 = Vov_v - Vds_v / 2
                safe1 = Id_ok & (denom1.abs() > 1e-6)
                if safe1.any():
                    gm_eq1 = Id_v[safe1] / denom1[safe1]
                    gm_true_s1 = gm_v[safe1]
                    eq1_ratio = gm_eq1 / gm_true_s1.clamp(min=1e-20)
                    eq1_pct = (gm_eq1 - gm_true_s1).abs() / gm_true_s1.clamp(min=1e-20) * 100
                    results[r]['eq1_ratio'].extend(eq1_ratio.cpu().tolist())
                    results[r]['eq1_pct'].extend(eq1_pct.cpu().tolist())
                    results[r]['eq1_count'].append(safe1.sum().item())

            # ================================================================
            # Eq2: gds = (I_DS/VDS) * (Vov-VDS)/(Vov-VDS/2) (gds-current)
            # ================================================================
            if Id_v is not None and Id_ok is not None:
                denom2a = Vds_v.abs().clamp(min=1e-6)
                denom2b = (Vov_v - Vds_v / 2).abs().clamp(min=1e-6)
                num2 = (Vov_v - Vds_v).abs()
                safe2 = Id_ok & (denom2b > 1e-6) & (num2 > 1e-6)
                if safe2.any():
                    gds_eq2 = (Id_v[safe2] / denom2a[safe2]) * (num2[safe2] / denom2b[safe2])
                    gds_true_s2 = gds_v[safe2]
                    eq2_ratio = gds_eq2 / gds_true_s2.clamp(min=1e-20)
                    eq2_pct = (gds_eq2 - gds_true_s2).abs() / gds_true_s2.clamp(min=1e-20) * 100
                    results[r]['eq2_ratio'].extend(eq2_ratio.cpu().tolist())
                    results[r]['eq2_pct'].extend(eq2_pct.cpu().tolist())
                    results[r]['eq2_count'].append(safe2.sum().item())

            # ================================================================
            # Sat eq: gm = 2 * I_DS / Vov  (saturation gm-current)
            # ================================================================
            if Id_v is not None and Id_ok is not None:
                safe_sat = Id_ok & (Vov_v.abs() > 1e-6)
                if safe_sat.any():
                    gm_sat = 2 * Id_v[safe_sat] / Vov_v[safe_sat]
                    gm_true_sat = gm_v[safe_sat]
                    sat_ratio = gm_sat / gm_true_sat.clamp(min=1e-20)
                    sat_pct = (gm_sat - gm_true_sat).abs() / gm_true_sat.clamp(min=1e-20) * 100
                    results[r]['sat_ratio'].extend(sat_ratio.cpu().tolist())
                    results[r]['sat_pct'].extend(sat_pct.cpu().tolist())
                    results[r]['sat_count'].append(safe_sat.sum().item())

    return results


def print_results(results):
    """Print equation accuracy results."""
    print(f"\n{'='*70}")
    print(f"  EQUATION ACCURACY vs SPICE GROUND TRUTH")
    print(f"{'='*70}")

    for r in [1, 2, 0]:
        data = results[r]
        count = sum(data.get('count', [])) if data.get('count') else 0
        if count == 0:
            continue

        print(f"\n  {'='*60}")
        print(f"  {REGION_NAMES[r]} region  (n={count})")
        print(f"  {'='*60}")

        # Voltage stats
        Vov = np.array(data['Vov'])
        Vds = np.array(data['Vds'])
        print(f"  Vov:  median={np.median(Vov)*1000:.1f}mV  mean={Vov.mean()*1000:.1f}mV  [min={Vov.min()*1000:.1f}, max={Vov.max()*1000:.1f}]")
        print(f"  Vds:  median={np.median(Vds)*1000:.1f}mV  mean={Vds.mean()*1000:.1f}mV  [min={Vds.min()*1000:.1f}, max={Vds.max()*1000:.1f}]")

        # Eq3
        if data.get('eq3_rel_err'):
            err = np.array(data['eq3_rel_err'])
            ratio = np.array(data['eq3_ratio'])
            print(f"\n  Eq3: gm*(Vov-Vds) = gds*Vds  [self-consistency]")
            print(f"    Rel error:  median={np.median(err):.1f}%  P75={np.percentile(err,75):.1f}%  P90={np.percentile(err,90):.1f}%")
            print(f"    Ratio:      median={np.median(ratio):.4f}  mean={np.mean(ratio):.4f}")
            if data.get('eq3_log_err'):
                le = np.array(data['eq3_log_err'])
                print(f"    Log10 err:  median={np.median(le):.3f}  P90={np.percentile(le,90):.3f} decades")

        # Eq1
        if data.get('eq1_pct'):
            pct = np.array(data['eq1_pct'])
            ratio = np.array(data['eq1_ratio'])
            n_eq1 = sum(data.get('eq1_count', []))
            print(f"\n  Eq1: gm = Id/(Vov - Vds/2)  [gm from current, n={n_eq1}]")
            print(f"    Rel error:  median={np.median(pct):.1f}%  P75={np.percentile(pct,75):.1f}%  P90={np.percentile(pct,90):.1f}%")
            print(f"    Ratio:      median={np.median(ratio):.4f}  mean={np.mean(ratio):.4f}")

        # Eq2
        if data.get('eq2_pct'):
            pct = np.array(data['eq2_pct'])
            ratio = np.array(data['eq2_ratio'])
            n_eq2 = sum(data.get('eq2_count', []))
            print(f"\n  Eq2: gds = (Id/Vds)*(Vov-Vds)/(Vov-Vds/2)  [gds from current, n={n_eq2}]")
            print(f"    Rel error:  median={np.median(pct):.1f}%  P75={np.percentile(pct,75):.1f}%  P90={np.percentile(pct,90):.1f}%")
            print(f"    Ratio:      median={np.median(ratio):.4f}  mean={np.mean(ratio):.4f}")

        # Saturation eq
        if data.get('sat_pct'):
            pct = np.array(data['sat_pct'])
            ratio = np.array(data['sat_ratio'])
            n_sat = sum(data.get('sat_count', []))
            print(f"\n  Sat: gm = 2*Id/Vov  [saturation gm-current, n={n_sat}]")
            print(f"    Rel error:  median={np.median(pct):.1f}%  P75={np.percentile(pct,75):.1f}%  P90={np.percentile(pct,90):.1f}%")
            print(f"    Ratio:      median={np.median(ratio):.4f}  mean={np.mean(ratio):.4f}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Test triode equation accuracy')
    parser.add_argument('--dataset', type=str, default='datasets/opamp_5k_ss_v1')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    device = args.device

    # Load data (do NOT normalize — we want raw SPICE values)
    split_dir = dataset_dir / args.split
    print(f"Loading {args.split} data from: {split_dir}")
    batches = load_prebatched_variant(split_dir, variant_id=0, device=device)
    print(f"  Loaded {len(batches)} batches")

    # Check what's available
    b = batches[0]
    print(f"  mosfet_info: {b.mosfet_info.shape}")
    print(f"  mosfet_gm: {b.mosfet_gm.shape}")
    print(f"  mosfet_gds: {b.mosfet_gds.shape}")
    print(f"  mosfet_vth: {b.mosfet_vth.shape}")
    print(f"  node_voltage_targets: {b.node_voltage_targets.shape}")
    print(f"  node_current_targets: {b.node_current_targets.shape}")
    print(f"  region_labels: {b.mosfet_region_labels.shape}")
    rl = b.mosfet_region_labels
    for r in [0, 1, 2]:
        print(f"    region {r} ({REGION_NAMES[r]}): {(rl == r).sum().item()}")

    print(f"\nTesting equations on SPICE ground truth...")
    results = test_equations(batches, device)
    print_results(results)


if __name__ == '__main__':
    main()
