#!/usr/bin/env python3
"""Patch terminal current targets to use actual SPICE per-device currents.

The original dataset stores the same current magnitude for all terminals,
which breaks KCL validation. This patches each terminal to use the actual
|Id| from SPICE _all_device_currents.
"""

import pickle
import math
import torch
from pathlib import Path


# Normalization constants (from training logs)
CURRENT_MEAN = -5.06
CURRENT_STD = 1.08
MIN_CURRENT = 1e-15  # floor for log10


def normalize_current(amps):
    """Convert |I| in amps to normalized target value."""
    log_i = math.log10(max(abs(amps), MIN_CURRENT))
    return (log_i - CURRENT_MEAN) / CURRENT_STD


def patch_sample(sample):
    """Fix terminal current targets using SPICE device currents."""
    g = sample['graph']
    specs = sample['specs']
    dc = specs.get('_all_device_currents', None)
    if dc is None:
        return False

    names = g.node_names
    n_terms = g.num_terminals
    targets = g.node_current_targets.clone()

    for i in range(n_terms):
        name = names[i]
        parts = name.rsplit('_', 1)
        dev = parts[0].lower()

        # Look up SPICE current for this device
        spice_current = dc.get(dev, None)

        if spice_current is not None:
            targets[i] = normalize_current(spice_current)
        elif dev == 'iref':
            # Iref = Xm8 current (KCL at nbias)
            xm8_current = dc.get('xm8', None)
            if xm8_current is not None:
                targets[i] = normalize_current(xm8_current)
        elif dev in ('cc', 'cload'):
            # Capacitors carry 0 current at DC
            targets[i] = normalize_current(MIN_CURRENT)
        elif dev == 'rbias_g':
            # Rbias_g carries ~0 current at DC (xm4 - xm2 ≈ 0)
            xm4 = dc.get('xm4', 0)
            xm2 = dc.get('xm2', 0)
            rbias_current = abs(xm4 - xm2)
            targets[i] = normalize_current(max(rbias_current, MIN_CURRENT))
        # else: keep original (voltage sources etc.)

    g.node_current_targets = targets
    return True


def patch_file(path):
    print(f"Patching {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    patched = 0
    for sample in data:
        if patch_sample(sample):
            patched += 1

    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Done. {patched}/{len(data)} samples patched.")


def verify(path):
    """Verify KCL at key nets with patched data."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f"\nVerification on {path}:")
    for si in range(min(3, len(data))):
        g = data[si]['graph']
        names = g.node_names
        n_terms = g.num_terminals
        src, dst = g.edge_index

        print(f"  Sample {si}:")
        for net_name in ['tail', 'vout', 'vout_stage1', 'nbias', 'vd1']:
            net_idx = None
            for j in range(n_terms, len(names)):
                if names[j] == net_name:
                    net_idx = j
                    break
            if net_idx is None:
                continue

            mask = (dst == net_idx) & (src < n_terms)
            term_indices = src[mask]

            total_signed = 0.0
            total_abs = 0.0
            for t in term_indices:
                ti = t.item()
                if not g.kcl_include_mask[ti]:
                    continue
                sign = g.terminal_current_sign[ti].item()
                gt_norm = g.node_current_targets[ti].item()
                gt_amps = 10 ** (gt_norm * CURRENT_STD + CURRENT_MEAN)
                total_signed += gt_amps * sign
                total_abs += gt_amps

            if total_abs > 0:
                rel = abs(total_signed) / total_abs
                status = "OK" if rel < 0.01 else f"VIOLATION {rel*100:.1f}%"
                print(f"    {net_name}: rel_viol={rel*100:.4f}%  {status}")


if __name__ == '__main__':
    ds_dir = Path('datasets/opamp_2stage_5k_ss')
    for name in ['dataset.pkl', 'dataset_train.pkl', 'dataset_val.pkl', 'dataset_test.pkl']:
        path = ds_dir / name
        if path.exists():
            patch_file(path)

    verify(ds_dir / 'dataset_train.pkl')
