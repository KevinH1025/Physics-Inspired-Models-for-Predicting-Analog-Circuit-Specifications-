#!/usr/bin/env python3
"""Patch existing dataset to add M (multiplier) as a MOSFET input feature.

Since the raw netlists aren't stored in the dataset, we compute per-device
effective M from the group-level params and hardcoded multipliers in the
netlist template.

Usage:
    python scripts/patch_dataset_add_m.py \
        --input datasets/opamp_3stage_fan_smc_v4/dataset.pkl \
        --output datasets/opamp_3stage_fan_smc_v4_m/dataset.pkl

The output dataset will have x.shape[1] = 9 instead of 8
(4 device props: W, L, W/L, M + 5 global features).
"""

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import torch

# Hardcoded multipliers from opamp_3stage_fan_smc_template.sp
# Maps device ID -> (group_name, hardcoded_multiplier)
DEVICE_MULTIPLIERS = {
    'M0':  ('BIASCM_P', 1),
    'M1':  ('BIASCM_P', 1),
    'M2':  ('BIASCM_P', 1),
    'M3':  ('BIASCM_P', 1),
    'M4':  ('BIASCM_P', 4),   # M='4*{M_BIASCM_P}'
    'M5':  ('BIASCM_P', 1),
    'M6':  ('BIASCM_P', 1),
    'M7':  ('BIASCM_P', 1),
    'M8':  ('GM1', 1),
    'M9':  ('GM1', 1),
    'M10': ('GM2', 1),
    'M11': ('GMF2', 1),
    'M12': ('BIASCM_N', 4),   # M='4*{M_BIASCM_N}'
    'M13': ('BIASCM_N', 4),   # M='4*{M_BIASCM_N}'
    'M14': ('BIASCM_N', 1),
    'M15': ('BIASCM_N', 4),   # M='4*{M_BIASCM_N}'
    'M16': ('BIASCM_N', 4),   # M='4*{M_BIASCM_N}'
    'M17': ('BIASCM_N', 4),   # M='4*{M_BIASCM_N}'
    'M18': ('BIASCM_N', 4),   # M='4*{M_BIASCM_N}'
    'M19': ('BIASCM_N', 8),   # M='8*{M_BIASCM_N}'
    'M20': ('BIASCM_N', 8),   # M='8*{M_BIASCM_N}'
    'M21': ('LOAD2', 1),
    'M22': ('LOAD2', 1),
    'M23': ('GM3', 1),
}

# MOSFET terminal suffixes (4 terminals per MOSFET)
MOSFET_TERMINALS = ['_drain', '_gate', '_source', '_bulk']


def get_effective_m(device_id: str, params: dict) -> float:
    """Compute effective M for a device given group params and hardcoded multipliers."""
    if device_id not in DEVICE_MULTIPLIERS:
        return 1.0
    group, hardcoded_mult = DEVICE_MULTIPLIERS[device_id]
    base_m = params.get(f'M_{group}', 1)
    return float(hardcoded_mult * base_m)


def normalize_m(m: float) -> float:
    """Log-normalize M to [0, 1] range. Same as graph_builder._normalize_mosfet_props()."""
    return min(math.log10(max(m, 1.0)) / 3.0, 1.0)


def patch_graph(graph, params: dict) -> torch.Tensor:
    """Add normalized M feature to graph.x tensor.

    Current x: [num_nodes, 8] = [w, l, wl_ratio, 0_pad, 0_pad, vdd, vcm, vin_p, vin_n, iref]
    Wait — actually x has shape [134, 8]. Let's figure out the layout:
    - MOSFET terminals: 3 device props (w, l, wl_ratio) padded to match max_props
    - Other terminals: 1 device prop padded
    - Net nodes: 0 device props padded
    - Global: 5 features appended

    So for MOSFETs: [w, l, wl_ratio, vdd, vcm, vin_p, vin_n, iref] → 3 + 5 = 8
    For others: [dc/value, 0, 0, vdd, vcm, vin_p, vin_n, iref] → 1 + 2_pad + 5 = 8
    For nets: [0, 0, 0, vdd, vcm, vin_p, vin_n, iref] → 0 + 3_pad + 5 = 8

    After adding M, MOSFETs need 4 device props, so:
    - MOSFET: [w, l, wl_ratio, m_norm, vdd, vcm, vin_p, vin_n, iref] = 9
    - Others: [dc/value, 0, 0, 0, vdd, vcm, vin_p, vin_n, iref] = 9
    - Nets: [0, 0, 0, 0, vdd, vcm, vin_p, vin_n, iref] = 9
    """
    node_names = graph.node_names
    old_x = graph.x
    num_nodes = old_x.shape[0]

    # The device props occupy columns 0:3, global features occupy columns 3:8
    # We need to insert a new column at position 3 (between device props and global features)
    device_props = old_x[:, :3]   # [num_nodes, 3]
    global_feats = old_x[:, 3:]   # [num_nodes, 5]

    # Create M column (zeros for non-MOSFET nodes)
    m_col = torch.zeros(num_nodes, 1, dtype=old_x.dtype)

    for i, name in enumerate(node_names):
        # Check if this is a MOSFET terminal node
        name_upper = name.upper()
        for suffix in MOSFET_TERMINALS:
            if name_upper.endswith(suffix.upper()):
                device_name = name[:len(name) - len(suffix)]
                device_id = device_name.upper().replace('X', '')
                eff_m = get_effective_m(device_id, params)
                m_col[i, 0] = normalize_m(eff_m)
                break

    new_x = torch.cat([device_props, m_col, global_feats], dim=1)
    return new_x


def main():
    parser = argparse.ArgumentParser(description='Patch dataset to add M feature')
    parser.add_argument('--input', required=True, help='Input dataset.pkl path')
    parser.add_argument('--output', required=True, help='Output dataset.pkl path')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    # Verify structure
    sample = data[0]
    graph = sample['graph']
    print(f"Original x shape: {graph.x.shape}")
    assert graph.x.shape[1] == 8, f"Expected 8 features, got {graph.x.shape[1]}"

    # Check that we can identify MOSFET nodes
    node_names = graph.node_names
    mosfet_count = 0
    for name in node_names:
        for suffix in MOSFET_TERMINALS:
            if name.upper().endswith(suffix.upper()):
                mosfet_count += 1
                break
    print(f"Sample 0: {mosfet_count} MOSFET terminal nodes out of {len(node_names)} total nodes")

    # Patch all samples
    for i, sample in enumerate(data):
        graph = sample['graph']
        params = sample['params']
        new_x = patch_graph(graph, params)
        graph.x = new_x

        if i == 0:
            print(f"New x shape: {graph.x.shape}")
            # Print some M values for verification
            for j, name in enumerate(graph.node_names):
                if name.lower().endswith('_drain'):
                    device = name[:name.rfind('_')]
                    device_id = device.upper().replace('X', '')
                    if device_id in DEVICE_MULTIPLIERS:
                        group, hm = DEVICE_MULTIPLIERS[device_id]
                        base_m = params.get(f'M_{group}', 1)
                        eff_m = hm * base_m
                        print(f"  {name}: M_norm={graph.x[j, 3]:.4f} "
                              f"(eff_M={eff_m}, base={base_m}, hardcoded={hm}x)")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print("Done!")


if __name__ == '__main__':
    main()
