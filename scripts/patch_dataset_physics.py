#!/usr/bin/env python3
"""
Patch pre-batched dataset to add mosfet_vth and node_mosfet_vth.

Reads Vth from specs['_mosfet_regions'] in the raw pkl files and matches to
pre-batched graphs using x-feature fingerprints. No SPICE re-run needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import torch


def build_vth_lookup(raw_graphs):
    """Build lookup: x-feature fingerprint -> per-mosfet Vth tensor."""
    lookup = {}
    for g in raw_graphs:
        graph_data = g['graph']
        mosfet_regions = g['specs']['_mosfet_regions']
        node_names = graph_data.node_names
        num_mosfets = graph_data.mosfet_info.shape[0]

        # Build device names from drain terminal names
        device_names = []
        fp_vals = []
        for i in range(num_mosfets):
            drain_idx = graph_data.mosfet_info[i, 1].item()
            terminal_name = node_names[drain_idx]
            device_name = terminal_name.split('_')[0].lower()
            device_names.append(device_name)
            fp_vals.extend(graph_data.x[drain_idx, :4].tolist())

        # Vth tensor [num_mosfets]
        vth = torch.zeros(num_mosfets, dtype=torch.float32)
        for i, dname in enumerate(device_names):
            if dname in mosfet_regions:
                vth[i] = abs(float(mosfet_regions[dname].get('vth', 0.0)))

        fp = tuple(round(v, 6) for v in fp_vals)
        lookup[fp] = vth

    return lookup


def patch_batch(batch, vth_lookup):
    """Patch a single batch with mosfet_vth and node_mosfet_vth.

    Uses mosfet_ptr if available for variable-size MOSFET counts per graph,
    otherwise falls back to uniform division.
    """
    ptr = batch.ptr
    num_graphs = len(ptr) - 1
    num_nodes = batch.x.shape[0]
    num_mosfets = batch.mosfet_info.shape[0]

    mosfet_ptr = getattr(batch, 'mosfet_ptr', None)

    all_vth = torch.zeros(num_mosfets, dtype=torch.float32)
    matched = 0

    for g_idx in range(num_graphs):
        g_start = ptr[g_idx].item()

        if mosfet_ptr is not None:
            m_start = mosfet_ptr[g_idx].item()
            m_end = mosfet_ptr[g_idx + 1].item()
        else:
            mosfets_per_graph = num_mosfets // num_graphs
            m_start = g_idx * mosfets_per_graph
            m_end = m_start + mosfets_per_graph

        # Build fingerprint from x features at drain terminals
        fp_vals = []
        for m_idx in range(m_start, m_end):
            local_drain = batch.mosfet_info[m_idx, 1].item()
            global_drain = g_start + local_drain
            fp_vals.extend(batch.x[global_drain, :4].tolist())
        fp = tuple(round(v, 6) for v in fp_vals)

        if fp in vth_lookup:
            vth = vth_lookup[fp]
            all_vth[m_start:m_end] = vth
            matched += 1

    batch.mosfet_vth = all_vth

    # Build node_mosfet_vth: scatter mosfet_vth to actual drain node positions
    node_mosfet_vth = torch.zeros(num_nodes, dtype=torch.float32)
    for g_idx in range(num_graphs):
        g_start = ptr[g_idx].item()

        if mosfet_ptr is not None:
            m_start = mosfet_ptr[g_idx].item()
            m_end = mosfet_ptr[g_idx + 1].item()
        else:
            mosfets_per_graph = num_mosfets // num_graphs
            m_start = g_idx * mosfets_per_graph
            m_end = m_start + mosfets_per_graph

        for m_idx in range(m_start, m_end):
            local_drain = batch.mosfet_info[m_idx, 1].item()
            global_drain = g_start + local_drain
            node_mosfet_vth[global_drain] = all_vth[m_idx]
    batch.node_mosfet_vth = node_mosfet_vth

    return matched, num_graphs - matched


def main():
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('datasets/opamp_5k_ss_v1')
    print(f"Patching dataset at {dataset_path}")

    for split in ['train', 'val']:
        pkl_path = dataset_path / f'dataset_{split}.pkl'
        split_dir = dataset_path / split

        if not pkl_path.exists():
            print(f"  Skipping {split}: {pkl_path} not found")
            continue

        print(f"\n--- {split} ---")
        print(f"  Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            raw_graphs = pickle.load(f)
        print(f"  {len(raw_graphs)} raw graphs")

        vth_lookup = build_vth_lookup(raw_graphs)
        print(f"  Built Vth lookup: {len(vth_lookup)} unique entries")
        del raw_graphs

        variant_files = sorted(split_dir.glob('variant_*.pkl'))
        total_matched, total_unmatched = 0, 0

        for vf in variant_files:
            print(f"  Patching {vf.name}...", end=' ')
            with open(vf, 'rb') as f:
                batches = pickle.load(f)

            for batch in batches:
                m, u = patch_batch(batch, vth_lookup)
                total_matched += m
                total_unmatched += u

            with open(vf, 'wb') as f:
                pickle.dump(batches, f)
            print(f"{len(batches)} batches OK")

        print(f"  Matched: {total_matched}, Unmatched: {total_unmatched}")

    # Verify
    print(f"\n--- Verification ---")
    with open(dataset_path / 'train' / 'variant_0.pkl', 'rb') as f:
        batches = pickle.load(f)
    b = batches[0]
    print(f"  mosfet_vth shape: {b.mosfet_vth.shape}")
    print(f"  mosfet_vth[:8]: {b.mosfet_vth[:8]}")
    print(f"  mosfet_vth nonzero: {(b.mosfet_vth > 0).sum().item()}/{b.mosfet_vth.shape[0]}")
    print(f"  node_mosfet_vth nonzero: {(b.node_mosfet_vth > 0).sum().item()}/{b.node_mosfet_vth.shape[0]}")
    print("\nDone!")


if __name__ == '__main__':
    main()
