#!/usr/bin/env python3
"""
Validate physics formulas against SPICE ground truth.

Generates 200 samples using the dataset parameter ranges, runs SPICE,
and compares analytical formulas against SPICE results.

Formulas tested:
1. gm = sqrt(2 * mu_Cox * W/L * Id) - device physics (no Vth needed)
2. Current mirror ratios: I_ratio vs WL_ratio (M3/M4, M5/M8, M7/M8)
3. Differential pair KCL: I_M1 + I_M2 = I_M5
4. Output stage balance: I_M6 vs I_M7
"""

import sys
import math
import yaml
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sampling import generate_lhs_samples, generate_netlist
from src.circuits.simulator import CircuitSimulator

# ─── SKY130 PDK constants ───────────────────────────────────────────
# From nfet_01v8.pm3.spice: u0 = 0.030197, toxe = 4.148e-9
# From pfet_01v8.pm3.spice: u0 ~ 0.00978, toxe = 4.148e-9
# Cox = eps0 * eps_SiO2 / tox = 8.854e-12 * 3.9 / 4.148e-9 = 8.329e-3 F/m^2
COX = 8.329e-3  # F/m^2
MU_COX_NMOS = 0.030197 * COX  # ~ 251e-6 A/V^2
MU_COX_PMOS = 0.00978 * COX   # ~ 81e-6 A/V^2

# Device topology: device_name -> (param_W, param_L, is_nmos)
DEVICE_INFO = {
    'xm1': ('W_M1', 'L_M1', True),   # NMOS diff pair
    'xm2': ('W_M2', 'L_M2', True),   # NMOS diff pair
    'xm3': ('W_M3', 'L_M3', False),  # PMOS active load (diode)
    'xm4': ('W_M4', 'L_M4', False),  # PMOS active load (mirror)
    'xm5': ('W_M5', 'L_M5', True),   # NMOS tail current source
    'xm6': ('W_M6', 'L_M6', False),  # PMOS second stage
    'xm7': ('W_M7', 'L_M7', True),   # NMOS output current source
    'xm8': ('W_M8', 'L_M8', True),   # NMOS bias (diode)
}

# Mirror pairs: (reference_device, mirror_device)
MIRROR_PAIRS = [
    ('xm3', 'xm4', 'M3/M4 PMOS load'),
    ('xm8', 'xm5', 'M8/M5 NMOS bias→tail'),
    ('xm8', 'xm7', 'M8/M7 NMOS bias→output'),
]


def simulate_sample(args):
    """Worker: simulate one sample and extract all operating point data."""
    idx, params, ac_template = args
    try:
        if 'VIN_P' not in params and 'VCM' in params:
            params['VIN_P'] = params['VCM'] + params.get('VDIFF', 0) / 2
            params['VIN_N'] = params['VCM'] - params.get('VDIFF', 0) / 2

        netlist_path = f'/tmp/validate_physics_{idx}.sp'
        generate_netlist(ac_template, params, netlist_path)

        with open(netlist_path) as f:
            content = f.read()

        sim = CircuitSimulator(analysis_types=['dc', 'ac'])
        results = sim.simulate(content)

        Path(netlist_path).unlink(missing_ok=True)

        if not results or '_all_net_voltages' not in results:
            return idx, None

        ss_params = results.get('_mosfet_ss_params', {})
        currents = results.get('_all_device_currents', {})

        # Extract Id per device
        device_currents = {}
        for dev_name in DEVICE_INFO:
            for key, val in currents.items():
                if dev_name in key.lower():
                    device_currents[dev_name] = abs(val)
                    break

        return idx, {
            'params': params,
            'ss_params': ss_params,
            'device_currents': device_currents,
        }
    except Exception as e:
        return idx, None


def validate_gm_formula(results):
    """Formula 1: gm = sqrt(2 * mu_Cox * W/L * Id)"""
    print("\n" + "=" * 70)
    print("FORMULA 1: gm = sqrt(2 * mu_Cox * W/L * Id)")
    print("=" * 70)

    per_device_errors = {dev: [] for dev in DEVICE_INFO}
    all_gm_spice = []
    all_gm_formula = []

    for r in results:
        params = r['params']
        ss = r['ss_params']
        ids = r['device_currents']

        for dev, (w_key, l_key, is_nmos) in DEVICE_INFO.items():
            if dev not in ss or dev not in ids:
                continue

            gm_spice = ss[dev]['gm']
            id_val = ids[dev]
            w = params[w_key]
            l = params[l_key]
            mu_cox = MU_COX_NMOS if is_nmos else MU_COX_PMOS

            if id_val < 1e-12 or gm_spice < 1e-12:
                continue

            gm_formula = math.sqrt(2 * mu_cox * (w / l) * id_val)
            error_pct = (gm_formula - gm_spice) / gm_spice * 100

            per_device_errors[dev].append(error_pct)
            all_gm_spice.append(gm_spice)
            all_gm_formula.append(gm_formula)

    # Per-device stats
    print(f"\n{'Device':<6} {'Type':<5} {'N':>4} {'Mean%':>8} {'Med%':>8} {'Std%':>8} {'Min%':>8} {'Max%':>8}")
    print("-" * 60)
    for dev, (_, _, is_nmos) in DEVICE_INFO.items():
        errs = per_device_errors[dev]
        if not errs:
            continue
        dtype = "NMOS" if is_nmos else "PMOS"
        print(f"{dev:<6} {dtype:<5} {len(errs):4d} {np.mean(errs):+8.1f} {np.median(errs):+8.1f} "
              f"{np.std(errs):8.1f} {np.min(errs):+8.1f} {np.max(errs):+8.1f}")

    # Overall stats
    all_errs = []
    for errs in per_device_errors.values():
        all_errs.extend(errs)
    if all_errs:
        abs_errs = [abs(e) for e in all_errs]
        print(f"\n  Overall: N={len(all_errs)}")
        print(f"  Mean absolute error: {np.mean(abs_errs):.1f}%")
        print(f"  Median absolute error: {np.median(abs_errs):.1f}%")

    # R-squared (log-scale)
    if all_gm_spice and all_gm_formula:
        log_spice = np.log10(all_gm_spice)
        log_formula = np.log10(all_gm_formula)
        corr = np.corrcoef(log_spice, log_formula)[0, 1]
        ss_res = np.sum((log_formula - log_spice) ** 2)
        ss_tot = np.sum((log_spice - np.mean(log_spice)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  Correlation (log-scale): {corr:.4f}")
        print(f"  R-squared (log-scale): {r2:.4f}")

        # Systematic bias: what mu_Cox would make it exact?
        # gm = sqrt(2*mu*W/L*Id) => mu = gm^2 / (2*W/L*Id)
        # Ratio: (gm_formula/gm_spice)^2 = mu_cox_used / mu_cox_effective
        ratios = [f / s for f, s in zip(all_gm_formula, all_gm_spice)]
        print(f"  Mean gm_formula/gm_spice ratio: {np.mean(ratios):.3f}")
        print(f"  Median ratio: {np.median(ratios):.3f}")


def validate_mirror_ratios(results):
    """Formula 2-3: Current mirror I_ratio vs WL_ratio"""
    print("\n" + "=" * 70)
    print("FORMULA 2: Current Mirror Ratios")
    print("=" * 70)

    for ref_dev, mir_dev, label in MIRROR_PAIRS:
        errors = []
        for r in results:
            params = r['params']
            ids = r['device_currents']

            if ref_dev not in ids or mir_dev not in ids:
                continue

            i_ref = ids[ref_dev]
            i_mir = ids[mir_dev]

            if i_ref < 1e-12:
                continue

            w_ref, l_ref = params[DEVICE_INFO[ref_dev][0]], params[DEVICE_INFO[ref_dev][1]]
            w_mir, l_mir = params[DEVICE_INFO[mir_dev][0]], params[DEVICE_INFO[mir_dev][1]]

            wl_ratio = (w_mir / l_mir) / (w_ref / l_ref)
            i_ratio = i_mir / i_ref
            error_pct = (i_ratio - wl_ratio) / wl_ratio * 100
            errors.append(error_pct)

        if errors:
            abs_errs = [abs(e) for e in errors]
            print(f"\n  {label}: N={len(errors)}")
            print(f"    Mean error: {np.mean(errors):+.2f}%")
            print(f"    Median error: {np.median(errors):+.2f}%")
            print(f"    Mean |error|: {np.mean(abs_errs):.2f}%")
            print(f"    Std: {np.std(errors):.2f}%")
            print(f"    Range: [{np.min(errors):+.1f}%, {np.max(errors):+.1f}%]")


def validate_diff_pair(results):
    """Formula 3: I_M1 + I_M2 = I_M5 (KCL at tail node)"""
    print("\n" + "=" * 70)
    print("FORMULA 3: Differential Pair KCL (I_M1 + I_M2 = I_M5)")
    print("=" * 70)

    errors = []
    for r in results:
        ids = r['device_currents']
        if not all(d in ids for d in ['xm1', 'xm2', 'xm5']):
            continue

        i_m1 = ids['xm1']
        i_m2 = ids['xm2']
        i_m5 = ids['xm5']

        if i_m5 < 1e-12:
            continue

        error_pct = abs(i_m1 + i_m2 - i_m5) / i_m5 * 100
        errors.append(error_pct)

    if errors:
        print(f"\n  N={len(errors)}")
        print(f"  Mean |error|: {np.mean(errors):.4f}%")
        print(f"  Median |error|: {np.median(errors):.4f}%")
        print(f"  Max |error|: {np.max(errors):.4f}%")
        print(f"  < 1%: {sum(1 for e in errors if e < 1)/len(errors)*100:.1f}%")
        print(f"  < 0.1%: {sum(1 for e in errors if e < 0.1)/len(errors)*100:.1f}%")


def validate_output_stage(results):
    """Formula 4: I_M6 vs I_M7 (output stage balance)"""
    print("\n" + "=" * 70)
    print("FORMULA 4: Output Stage Balance (I_M6 vs I_M7)")
    print("=" * 70)

    errors = []
    for r in results:
        ids = r['device_currents']
        if 'xm6' not in ids or 'xm7' not in ids:
            continue

        i_m6 = ids['xm6']
        i_m7 = ids['xm7']
        total = i_m6 + i_m7

        if total < 1e-12:
            continue

        error_pct = abs(i_m6 - i_m7) / total * 200  # symmetric %
        errors.append(error_pct)

    if errors:
        print(f"\n  N={len(errors)}")
        print(f"  Mean |error|: {np.mean(errors):.2f}%")
        print(f"  Median |error|: {np.median(errors):.2f}%")
        print(f"  Max |error|: {np.max(errors):.2f}%")
        print(f"  < 10%: {sum(1 for e in errors if e < 10)/len(errors)*100:.1f}%")
        print(f"  < 5%: {sum(1 for e in errors if e < 5)/len(errors)*100:.1f}%")


def main():
    # Load parameter specs from config
    config_path = 'configs/opamp_dataset/opamp_dataset_v2.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    param_specs = {}
    for name, spec in config.get('parameters', {}).items():
        if isinstance(spec, dict):
            param_specs[name] = spec
        else:
            param_specs[name] = {'value': spec}

    ac_template = config['dataset']['ac_template']

    # Generate 200 LHS samples
    num_samples = 200
    print(f"Generating {num_samples} LHS parameter samples...")
    all_params = generate_lhs_samples(param_specs, num_samples, seed=42)

    # Compute VIN_P/VIN_N from VCM/VDIFF
    for p in all_params:
        if 'VIN_P' not in p and 'VCM' in p:
            p['VIN_P'] = p['VCM'] + p.get('VDIFF', 0) / 2
            p['VIN_N'] = p['VCM'] - p.get('VDIFF', 0) / 2

    # Simulate in parallel
    n_workers = min(20, len(all_params))
    print(f"Simulating {len(all_params)} samples with {n_workers} workers...")

    work_items = [(i, p, ac_template) for i, p in enumerate(all_params)]
    results = []
    failed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(simulate_sample, item): item[0] for item in work_items}
        for future in as_completed(futures):
            try:
                idx, data = future.result(timeout=30)
                if data is not None:
                    results.append(data)
                else:
                    failed += 1
            except Exception:
                failed += 1

    print(f"\nSuccessful: {len(results)}/{num_samples} (failed: {failed})")

    if len(results) < 10:
        print("Too few successful samples. Aborting.")
        return

    # Validate each formula
    validate_gm_formula(results)
    validate_mirror_ratios(results)
    validate_diff_pair(results)
    validate_output_stage(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print("""
  Formula 1 (gm): Use as SOFT loss if correlation > 0.9 and median error < 100%.
                   Weight: 0.1 (low, approximate).
  Formula 2 (mirror): Use as constraint if median |error| < 10%.
                       Weight: 5.0 (should be near-exact).
  Formula 3 (diff pair KCL): Use as constraint if median |error| < 1%.
                              Weight: 5.0 (exact conservation law).
  Formula 4 (output stage): Use as constraint if median |error| < 20%.
                             Weight: 5.0 (should hold at DC equilibrium).
    """)


if __name__ == '__main__':
    main()
