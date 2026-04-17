#!/usr/bin/env python3
"""
Quick test: verify AC extraction pipeline works on a single sample.

Tests:
1. DC results from AC template match original template
2. AC metrics (UGBW, PM, AM, DC gain) are extracted
3. Small-signal params (gm, gds) are extracted
4. Analytical vs SPICE AC comparison
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits.simulator import CircuitSimulator
from src.data.sampling import generate_netlist

# Test parameters (typical 2-stage op-amp values)
params = {
    'W_M1': 10e-6, 'L_M1': 0.5e-6,
    'W_M2': 10e-6, 'L_M2': 0.5e-6,
    'W_M3': 20e-6, 'L_M3': 0.5e-6,
    'W_M4': 20e-6, 'L_M4': 0.5e-6,
    'W_M5': 10e-6, 'L_M5': 0.5e-6,
    'W_M6': 50e-6, 'L_M6': 0.5e-6,
    'W_M7': 25e-6, 'L_M7': 0.5e-6,
    'W_M8': 10e-6, 'L_M8': 0.5e-6,
    'C_C': 2e-12,
    'R_Z': 1000,
    'R_IN': 50000,
    'R_F': 50000,
    'I_REF': 20e-6,
    'VDD': 1.8,
    'VIN_P': 0.9,
    'VIN_N': 0.9,
}

output_dir = Path('/tmp/test_ac')
output_dir.mkdir(exist_ok=True)

# === Test 1: Run original DC-only simulation ===
print("=" * 60)
print("TEST 1: Original template (DC only)")
print("=" * 60)

orig_template = 'netlists/opamp_2stage_template.sp'
orig_netlist = str(output_dir / 'test_orig.sp')
generate_netlist(orig_template, params, orig_netlist)

with open(orig_netlist) as f:
    orig_content = f.read()

sim_dc = CircuitSimulator(analysis_types=['dc'])
dc_results = sim_dc.simulate(orig_content)

print(f"DC node voltages:")
for net, v in sorted(dc_results.get('_all_net_voltages', {}).items()):
    if not net.startswith('@') and '#' not in net:
        print(f"  {net:20s} = {v:.6f} V")

# === Test 2: Run AC template simulation ===
print("\n" + "=" * 60)
print("TEST 2: AC template (DC + AC)")
print("=" * 60)

ac_template = 'netlists/opamp_2stage_ac_template.sp'
ac_netlist = str(output_dir / 'test_ac.sp')
generate_netlist(ac_template, params, ac_netlist)

with open(ac_netlist) as f:
    ac_content = f.read()

sim_ac = CircuitSimulator(analysis_types=['dc', 'ac'])
ac_results = sim_ac.simulate(ac_content)

# Map node names back
ac_voltages_raw = ac_results.get('_all_net_voltages', {})
ac_voltages = CircuitSimulator.map_ac_node_voltages(ac_voltages_raw)

print(f"AC template DC node voltages (mapped):")
for net, v in sorted(ac_voltages.items()):
    if not net.startswith('@') and '#' not in net:
        print(f"  {net:20s} = {v:.6f} V")

# === Test 3: Compare DC results ===
print("\n" + "=" * 60)
print("TEST 3: DC comparison (original vs AC template)")
print("=" * 60)

dc_orig = dc_results.get('_all_net_voltages', {})
max_diff = 0
for net in dc_orig:
    if net.startswith('@') or '#' in net:
        continue
    v_orig = dc_orig.get(net, 0)
    v_ac = ac_voltages.get(net, 0)
    diff = abs(v_orig - v_ac)
    max_diff = max(max_diff, diff)
    if diff > 1e-6:
        print(f"  WARNING: {net}: orig={v_orig:.6f} ac={v_ac:.6f} diff={diff:.2e}")

if max_diff < 1e-6:
    print(f"  All DC voltages match (max diff: {max_diff:.2e} V)")
else:
    print(f"  Max DC voltage difference: {max_diff:.2e} V")

# === Test 4: AC metrics ===
print("\n" + "=" * 60)
print("TEST 4: AC metrics from SPICE")
print("=" * 60)

ugbw = ac_results.get('_ac_ugbw')
pm = ac_results.get('_ac_pm')
am = ac_results.get('_ac_am')
dc_gain = ac_results.get('_ac_dc_gain')

print(f"  DC Gain:  {dc_gain:.1f} dB" if dc_gain is not None else "  DC Gain:  FAILED")
print(f"  UGBW:     {ugbw:.0f} Hz ({ugbw/1e6:.2f} MHz)" if ugbw is not None else "  UGBW:     FAILED")
print(f"  PM:       {pm:.1f} deg" if pm is not None else "  PM:       FAILED")
print(f"  AM:       {am:.1f} dB" if am is not None else "  AM:       FAILED")

# === Test 5: Small-signal params ===
print("\n" + "=" * 60)
print("TEST 5: Small-signal parameters (gm, gds)")
print("=" * 60)

ss_params = ac_results.get('_mosfet_ss_params', {})
if ss_params:
    for dev in sorted(ss_params.keys()):
        gm = ss_params[dev].get('gm', 0)
        gds = ss_params[dev].get('gds', 0)
        print(f"  {dev}: gm={gm:.3e} A/V, gds={gds:.3e} A/V, rds={1/gds:.0f} Ohm" if gds > 0
              else f"  {dev}: gm={gm:.3e} A/V, gds={gds:.3e} A/V")
else:
    print("  FAILED: No small-signal params extracted")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

# Cleanup
(output_dir / 'test_orig.sp').unlink(missing_ok=True)
(output_dir / 'test_ac.sp').unlink(missing_ok=True)
