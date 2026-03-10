#!/usr/bin/env python3
"""
Inspect the generated dataset to verify graph structure and features.

This script prints:
1. Number of nodes (terminals + nets)
2. Example feature vectors for each component type
3. Verification that input voltages are modeled correctly
"""

import pickle
import sys
from pathlib import Path
from collections import defaultdict

def main():
    dataset_path = Path('datasets/opamp_2stage_v2/dataset.pkl')
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        sys.exit(1)
    
    with open(dataset_path, 'rb') as f:
        samples = pickle.load(f)
    
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    
    g = samples[0]['graph']
    
    # Basic counts
    print(f"\n{'='*70}")
    print("1. GRAPH STRUCTURE")
    print("=" * 70)
    print(f"Total nodes:      {len(g.node_types)}")
    print(f"  - Terminals:    {g.num_terminals}")
    print(f"  - Nets:         {g.num_nets}")
    print(f"Total edges:      {g.edge_index.shape[1]} (bidirectional)")
    print(f"Feature dim (x):  {g.x.shape}")
    print(f"Type encoding:    {g.type_tens.shape}")
    
    # Count by type
    type_counts = defaultdict(int)
    for ntype in g.node_types:
        type_counts[ntype] += 1
    
    print(f"\nNode type breakdown:")
    for ntype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {str(ntype):25s}: {count}")
    
    # Feature vectors by type
    print(f"\n{'='*70}")
    print("2. EXAMPLE FEATURE VECTORS (before normalization stats applied)")
    print("=" * 70)
    
    # Load stats for denormalization
    stats_path = Path('datasets/opamp_2stage_v2/stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Find examples of each type
    examples = {}
    for i, ntype in enumerate(g.node_types):
        key = ntype[0] if len(ntype) == 2 else ntype[:2]  # Group by first element(s)
        if key not in examples:
            examples[key] = (i, ntype, g.x[i].tolist())
    
    # Print examples with denormalized values
    print("\nNote: x values shown are Z-SCORE NORMALIZED (mean=0, std=1)")
    print("      To get original values: original = x * std + mean\n")
    
    for key in ['V', 'R', 'C', 'I', ('M', 'N'), ('M', 'P'), 'VNode']:
        if key in examples:
            idx, ntype, x_norm = examples[key]
            
            # Denormalize to get original values
            if ntype in stats:
                x_orig = []
                for j, val in enumerate(x_norm):
                    if j in stats[ntype]:
                        mean = stats[ntype][j]['mean']
                        std = stats[ntype][j]['std']
                        x_orig.append(val * std + mean)
                    else:
                        x_orig.append(val)
            else:
                x_orig = x_norm
            
            print(f"{str(ntype):25s} (node {idx}):")
            print(f"  Normalized x:   {x_norm}")
            print(f"  Original value: {x_orig}")
            
            # Explain what each feature means
            if ntype[0] == 'V':
                print(f"  → Features: [dc_voltage, padding]")
            elif ntype[0] == 'R':
                print(f"  → Features: [resistance (Ω), padding]")
            elif ntype[0] == 'C':
                print(f"  → Features: [capacitance (F), padding]")
            elif ntype[0] == 'I':
                print(f"  → Features: [dc_current (A), padding]")
            elif ntype[0] == 'M':
                print(f"  → Features: [width (m), length (m)]")
            elif ntype[0] == 'VNode':
                print(f"  → Features: [none - nets have no properties]")
            print()
    
    # Check input voltage modeling
    print(f"{'='*70}")
    print("3. INPUT VOLTAGE MODELING CHECK")
    print("=" * 70)
    
    print("\nVoltage source terminals (type starts with 'V'):")
    for i, ntype in enumerate(g.node_types):
        if ntype[0] == 'V':
            x_norm = g.x[i].tolist()
            known = g.known_voltage_mask[i].item()
            output = g.output_node_mask[i].item()
            v_target = g.node_voltage_targets[i].item()
            
            # Denormalize dc value
            if ntype in stats and 0 in stats[ntype]:
                dc_orig = x_norm[0] * stats[ntype][0]['std'] + stats[ntype][0]['mean']
            else:
                dc_orig = x_norm[0]
            
            print(f"  Node {i:2d}: {str(ntype):10s} | dc={dc_orig:.4f}V | known={known} | V_target={v_target:.4f}")
    
    print("\nNet nodes (type starts with 'VNode'):")
    for i, ntype in enumerate(g.node_types):
        if ntype[0] == 'VNode':
            known = g.known_voltage_mask[i].item()
            output = g.output_node_mask[i].item()
            v_target = g.node_voltage_targets[i].item()
            
            # Denormalize vdc target
            if 'vdc' in stats:
                v_orig = v_target * stats['vdc']['std'] + stats['vdc']['mean']
            else:
                v_orig = v_target
            
            status = "KNOWN" if known else "UNKNOWN (predict)"
            print(f"  Node {i:2d}: {str(ntype):20s} | V={v_orig:.4f}V | {status}")
    
    # Check which nets are connected to input voltages
    print(f"\n{'='*70}")
    print("4. EDGE CONNECTIVITY (terminal → net)")
    print("=" * 70)
    
    # Build adjacency
    edge_index = g.edge_index
    adj = defaultdict(set)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj[src].add(dst)
    
    print("\nVoltage source terminal connections:")
    for i, ntype in enumerate(g.node_types):
        if ntype[0] == 'V':
            neighbors = adj[i]
            neighbor_types = [g.node_types[n] for n in neighbors]
            print(f"  Node {i:2d} ({ntype}) → {list(neighbors)} {neighbor_types}")
    
    print("\n" + "=" * 70)
    print("5. VERIFICATION SUMMARY")
    print("=" * 70)
    
    # Check if VIN_P and VIN_N are modeled
    vin_terminals = []
    for i, ntype in enumerate(g.node_types):
        if ntype[0] == 'V':
            x_norm = g.x[i].tolist()
            if ntype in stats and 0 in stats[ntype]:
                dc_orig = x_norm[0] * stats[ntype][0]['std'] + stats[ntype][0]['mean']
            else:
                dc_orig = x_norm[0]
            # VIN_P and VIN_N should be around 0.9V (VCM)
            if 0.5 < dc_orig < 1.3 and dc_orig != 1.8:
                vin_terminals.append((i, dc_orig))
    
    print(f"\n✓ Input voltage terminals found: {len(vin_terminals)}")
    for idx, dc in vin_terminals:
        print(f"    Node {idx}: dc = {dc:.4f}V (VIN_P or VIN_N)")
    
    # Check that input nets exist
    known_nets = sum(1 for i, ntype in enumerate(g.node_types) 
                     if ntype[0] == 'VNode' and g.known_voltage_mask[i].item())
    unknown_nets = sum(1 for i, ntype in enumerate(g.node_types) 
                       if ntype[0] == 'VNode' and not g.known_voltage_mask[i].item())
    
    print(f"\n✓ Net nodes: {g.num_nets} total")
    print(f"    - Known (not predicted): {known_nets} (VDD, GND, input nets)")
    print(f"    - Unknown (predicted):   {unknown_nets} (internal nodes)")
    
    print(f"\n✓ Prediction targets (output_node_mask): {g.output_node_mask.sum().item()} nodes")
    print(f"✓ Known voltages (known_voltage_mask):   {g.known_voltage_mask.sum().item()} nodes")
    
    # vdc normalization check
    print(f"\n✓ vdc target normalization: mean={stats['vdc']['mean']:.4f}V, std={stats['vdc']['std']:.4f}V")


if __name__ == '__main__':
    main()
