#!/usr/bin/env python3
"""Deep analysis and comparison of circuit datasets with plots."""

import pickle
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_dataset(path):
    """Load train+val splits and extract all relevant data."""
    path = Path(path)
    train = pickle.load(open(path / 'dataset_train.pkl', 'rb'))
    val = pickle.load(open(path / 'dataset_val.pkl', 'rb'))
    return train, val

def extract_mosfet_operating_points(samples):
    """Extract per-device operating point data from specs."""
    device_data = defaultdict(lambda: {'vgs': [], 'vds': [], 'vdsat': [], 'vth': [], 'region': [], 'current': [], 'is_pmos': None})

    for s in samples:
        specs = s.get('specs', {})
        regions = specs.get('_mosfet_regions', {})
        currents = specs.get('_all_device_currents', {})

        for dev, info in regions.items():
            device_data[dev]['vgs'].append(info['vgs'])
            device_data[dev]['vds'].append(abs(info['vds']))
            device_data[dev]['vdsat'].append(info['vdsat'])
            device_data[dev]['vth'].append(info['vth'])
            device_data[dev]['region'].append(info['region'])
            device_data[dev]['is_pmos'] = info['is_pmos']
            if dev in currents:
                device_data[dev]['current'].append(abs(currents[dev]))

    return device_data

def extract_design_params(samples):
    """Extract design parameters."""
    param_data = defaultdict(list)
    for s in samples:
        for k, v in s['params'].items():
            param_data[k].append(v)
    return {k: np.array(v) for k, v in param_data.items()}

def extract_ac_data(samples):
    """Extract AC performance data."""
    ugbw, pm, am, dc_gain = [], [], [], []
    valid_count = 0
    total = len(samples)

    for s in samples:
        g = s['graph']
        specs = s.get('specs', {})
        if hasattr(g, 'ac_valid') and g.ac_valid is not None and g.ac_valid.item():
            valid_count += 1
            if hasattr(g, 'ac_ugbw') and g.ac_ugbw is not None:
                ugbw.append(g.ac_ugbw.item())
            if hasattr(g, 'ac_pm') and g.ac_pm is not None:
                pm.append(g.ac_pm.item())
            if hasattr(g, 'ac_am') and g.ac_am is not None:
                am.append(g.ac_am.item())
            if hasattr(g, 'ac_dc_gain') and g.ac_dc_gain is not None:
                dc_gain.append(g.ac_dc_gain.item())

    return {
        'ugbw': np.array(ugbw) if ugbw else None,
        'pm': np.array(pm) if pm else None,
        'am': np.array(am) if am else None,
        'dc_gain': np.array(dc_gain) if dc_gain else None,
        'valid_count': valid_count,
        'total': total,
        'valid_pct': 100 * valid_count / total if total > 0 else 0,
    }

def extract_ss_data(samples):
    """Extract small-signal gm/gds data."""
    all_gm, all_gds = [], []
    per_device_gm = defaultdict(list)
    per_device_gds = defaultdict(list)

    for s in samples:
        specs = s.get('specs', {})
        ss = specs.get('_mosfet_ss_params', {})
        for dev, params in ss.items():
            gm = params['gm']
            gds = params['gds']
            all_gm.append(gm)
            all_gds.append(gds)
            per_device_gm[dev].append(gm)
            per_device_gds[dev].append(gds)

    return {
        'all_gm': np.array(all_gm),
        'all_gds': np.array(all_gds),
        'per_device_gm': {k: np.array(v) for k, v in per_device_gm.items()},
        'per_device_gds': {k: np.array(v) for k, v in per_device_gds.items()},
    }

def extract_node_voltages(samples):
    """Extract per-net DC voltages."""
    net_voltages = defaultdict(list)
    for s in samples:
        specs = s.get('specs', {})
        net_v = specs.get('_all_net_voltages', {})
        # Only get actual net voltages (not device params)
        for k, v in net_v.items():
            if not k.startswith('@') and not k.startswith('m.'):
                net_voltages[k].append(v)
    return {k: np.array(v) for k, v in net_voltages.items()}


def plot_comparison(datasets, output_dir):
    """Generate all comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(datasets.keys())
    colors = {'2k_nofil': '#2196F3', '5k_fixed': '#FF5722'}

    # =========================================================================
    # 1. Design Parameter Distributions
    # =========================================================================
    print("\n[1/8] Design parameter distributions...")
    params = {n: extract_design_params(d['train']) for n, d in datasets.items()}

    # W and L parameters
    w_params = sorted([k for k in params[names[0]].keys() if k.startswith('W_')])
    l_params = sorted([k for k in params[names[0]].keys() if k.startswith('L_')])
    m_params = sorted([k for k in params[names[0]].keys() if k.startswith('M_')])

    fig, axes = plt.subplots(3, max(len(w_params), len(l_params), len(m_params)),
                              figsize=(4*max(len(w_params), len(l_params), len(m_params)), 10))
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    for i, wp in enumerate(w_params):
        ax = axes[0, i]
        for n in names:
            ax.hist(params[n][wp] * 1e6, bins=30, alpha=0.5, label=n, color=colors[n])
        ax.set_title(wp)
        ax.set_xlabel('W (µm)')
        if i == 0:
            ax.legend(fontsize=7)

    for i, lp in enumerate(l_params):
        ax = axes[1, i]
        for n in names:
            ax.hist(params[n][lp] * 1e6, bins=30, alpha=0.5, label=n, color=colors[n])
        ax.set_title(lp)
        ax.set_xlabel('L (µm)')

    for i, mp in enumerate(m_params):
        ax = axes[2, i]
        for n in names:
            ax.hist(params[n][mp], bins=30, alpha=0.5, label=n, color=colors[n])
        ax.set_title(mp)
        ax.set_xlabel('M (multiplier)')

    # Hide unused axes
    for row in range(3):
        param_list = [w_params, l_params, m_params][row]
        for j in range(len(param_list), axes.shape[1]):
            axes[row, j].set_visible(False)

    fig.suptitle('Design Parameter Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'design_params.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Other params (I_BIAS, C_COMP, R_F, R_IN, VCM, etc.)
    other_params = sorted([k for k in params[names[0]].keys()
                           if not k.startswith(('W_', 'L_', 'M_', 'VDD', 'VIN_', 'VDIFF'))])
    fig, axes = plt.subplots(2, (len(other_params)+1)//2, figsize=(4*((len(other_params)+1)//2), 8))
    axes = axes.flatten()
    for i, op in enumerate(other_params):
        ax = axes[i]
        for n in names:
            vals = params[n].get(op)
            if vals is not None:
                # Scale for readability
                if 'I_BIAS' in op:
                    ax.hist(vals * 1e6, bins=30, alpha=0.5, label=n, color=colors[n])
                    ax.set_xlabel(f'{op} (µA)')
                elif 'C_' in op:
                    ax.hist(vals * 1e12, bins=30, alpha=0.5, label=n, color=colors[n])
                    ax.set_xlabel(f'{op} (pF)')
                elif 'R_' in op:
                    ax.hist(vals / 1e3, bins=30, alpha=0.5, label=n, color=colors[n])
                    ax.set_xlabel(f'{op} (kΩ)')
                else:
                    ax.hist(vals, bins=30, alpha=0.5, label=n, color=colors[n])
                    ax.set_xlabel(op)
        ax.set_title(op)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(other_params), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Other Parameter Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'other_params.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 2. Node Voltage Distributions
    # =========================================================================
    print("[2/8] Node voltage distributions...")
    net_voltages = {n: extract_node_voltages(d['train']) for n, d in datasets.items()}

    # Key internal nets (not supply/input)
    key_nets = sorted([k for k in net_voltages[names[0]].keys()
                       if k not in ('gnda', 'vdda', 'vinp', 'vinn', 'vsig', '0')])

    n_nets = len(key_nets)
    ncols = min(6, n_nets)
    nrows = (n_nets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3*nrows))
    axes = np.array(axes).flatten()

    for i, net in enumerate(key_nets):
        ax = axes[i]
        for n in names:
            if net in net_voltages[n]:
                ax.hist(net_voltages[n][net], bins=40, alpha=0.5, label=n, color=colors[n])
        ax.set_title(net, fontsize=9)
        ax.set_xlabel('V')
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n_nets, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Internal Node Voltage Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'node_voltages.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. MOSFET Operating Regions
    # =========================================================================
    print("[3/8] MOSFET operating regions...")
    device_data = {n: extract_mosfet_operating_points(d['train']) for n, d in datasets.items()}

    # Get device list from first dataset
    devices = sorted(device_data[names[0]].keys())

    # Region breakdown per device
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 4*len(names)))
    if len(names) == 1:
        axes = [axes]

    region_colors = {'saturation': '#4CAF50', 'triode': '#FF9800', 'cutoff': '#F44336'}

    for idx, n in enumerate(names):
        ax = axes[idx]
        dd = device_data[n]
        x = np.arange(len(devices))
        width = 0.25

        region_counts = {r: [] for r in ['saturation', 'triode', 'cutoff']}
        for dev in devices:
            total = len(dd[dev]['region'])
            for r in ['saturation', 'triode', 'cutoff']:
                count = sum(1 for reg in dd[dev]['region'] if reg == r)
                region_counts[r].append(100 * count / total if total > 0 else 0)

        bottom = np.zeros(len(devices))
        for r in ['saturation', 'triode', 'cutoff']:
            vals = np.array(region_counts[r])
            ax.bar(x, vals, bottom=bottom, label=r, color=region_colors[r], alpha=0.8)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(devices, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Percentage')
        ax.set_title(f'{n}: Operating Region per Device')
        ax.legend()
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'operating_regions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 4. Vds vs Vdsat scatter (saturation margin)
    # =========================================================================
    print("[4/8] Saturation margin analysis...")
    fig, axes = plt.subplots(1, len(names), figsize=(8*len(names), 7))
    if len(names) == 1:
        axes = [axes]

    for idx, n in enumerate(names):
        ax = axes[idx]
        dd = device_data[n]

        for dev in devices:
            vds = np.array(dd[dev]['vds'])
            vdsat = np.array(dd[dev]['vdsat'])
            regions = dd[dev]['region']

            # Color by region
            for r, color in region_colors.items():
                mask = [reg == r for reg in regions]
                if any(mask):
                    ax.scatter(vdsat[mask], vds[mask], alpha=0.15, s=8, color=color)

        # Add Vds = Vdsat line
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='Vds=Vdsat')
        ax.set_xlabel('Vdsat (V)')
        ax.set_ylabel('|Vds| (V)')
        ax.set_title(f'{n}: Saturation Margin')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'saturation_margin.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 5. Current Distribution per Device
    # =========================================================================
    print("[5/8] Device current distributions...")
    fig, axes = plt.subplots(1, len(names), figsize=(8*len(names), 6))
    if len(names) == 1:
        axes = [axes]

    for idx, n in enumerate(names):
        ax = axes[idx]
        dd = device_data[n]

        currents_per_dev = []
        dev_labels = []
        for dev in devices:
            if dd[dev]['current']:
                currents_per_dev.append(np.array(dd[dev]['current']) * 1e6)
                dev_labels.append(dev)

        bp = ax.boxplot(currents_per_dev, labels=dev_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[n])
            patch.set_alpha(0.5)
        ax.set_xticklabels(dev_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('|Id| (µA)')
        ax.set_title(f'{n}: Device Current Distribution')
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'device_currents.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 6. Small-Signal gm/gds
    # =========================================================================
    print("[6/8] Small-signal gm/gds distributions...")
    ss_data = {n: extract_ss_data(d['train']) for n, d in datasets.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Overall gm distribution
    ax = axes[0, 0]
    for n in names:
        gm = ss_data[n]['all_gm']
        ax.hist(np.log10(gm[gm > 0]), bins=50, alpha=0.5, label=n, color=colors[n])
    ax.set_xlabel('log10(gm)')
    ax.set_ylabel('Count')
    ax.set_title('gm Distribution (all devices)')
    ax.legend()

    # Overall gds distribution
    ax = axes[0, 1]
    for n in names:
        gds = ss_data[n]['all_gds']
        ax.hist(np.log10(gds[gds > 0]), bins=50, alpha=0.5, label=n, color=colors[n])
    ax.set_xlabel('log10(gds)')
    ax.set_ylabel('Count')
    ax.set_title('gds Distribution (all devices)')
    ax.legend()

    # gm/gds ratio (intrinsic gain)
    ax = axes[1, 0]
    for n in names:
        gm = ss_data[n]['all_gm']
        gds = ss_data[n]['all_gds']
        ratio = gm / (gds + 1e-15)
        ax.hist(np.log10(ratio[ratio > 0]), bins=50, alpha=0.5, label=n, color=colors[n])
    ax.set_xlabel('log10(gm/gds) = log10(intrinsic gain)')
    ax.set_ylabel('Count')
    ax.set_title('Intrinsic Gain Distribution')
    ax.legend()

    # Per-device gm/gds ratio boxplot (for first dataset)
    ax = axes[1, 1]
    n0 = names[0]
    ratios_per_dev = []
    dev_labels = []
    for dev in sorted(ss_data[n0]['per_device_gm'].keys()):
        gm = ss_data[n0]['per_device_gm'][dev]
        gds = ss_data[n0]['per_device_gds'][dev]
        r = gm / (gds + 1e-15)
        ratios_per_dev.append(np.log10(r[r > 0]))
        dev_labels.append(dev)
    ax.boxplot(ratios_per_dev, labels=dev_labels)
    ax.set_xticklabels(dev_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('log10(gm/gds)')
    ax.set_title(f'{n0}: Intrinsic Gain per Device')

    plt.tight_layout()
    plt.savefig(output_dir / 'ss_gm_gds.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 7. AC Performance
    # =========================================================================
    print("[7/8] AC performance distributions...")
    ac_data = {n: extract_ac_data(d['train']) for n, d in datasets.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Print AC validity
    for n in names:
        print(f"  {n}: AC valid {ac_data[n]['valid_count']}/{ac_data[n]['total']} ({ac_data[n]['valid_pct']:.1f}%)")

    # UGBW
    ax = axes[0, 0]
    for n in names:
        if ac_data[n]['ugbw'] is not None and len(ac_data[n]['ugbw']) > 0:
            ax.hist(np.log10(np.clip(ac_data[n]['ugbw'], 1, None)), bins=40, alpha=0.5, label=n, color=colors[n])
    ax.set_xlabel('log10(UGBW) [Hz]')
    ax.set_ylabel('Count')
    ax.set_title('Unity Gain Bandwidth')
    ax.legend()

    # Phase Margin
    ax = axes[0, 1]
    for n in names:
        if ac_data[n]['pm'] is not None and len(ac_data[n]['pm']) > 0:
            ax.hist(ac_data[n]['pm'], bins=40, alpha=0.5, label=n, color=colors[n])
    ax.set_xlabel('Phase Margin (°)')
    ax.set_ylabel('Count')
    ax.set_title('Phase Margin')
    ax.axvline(60, color='green', linestyle='--', alpha=0.5, label='60° target')
    ax.legend()

    # Gain Margin
    ax = axes[1, 0]
    for n in names:
        if ac_data[n]['am'] is not None and len(ac_data[n]['am']) > 0:
            ax.hist(ac_data[n]['am'], bins=40, alpha=0.5, label=n, color=colors[n])
    ax.set_xlabel('Amplitude Margin (dB)')
    ax.set_ylabel('Count')
    ax.set_title('Amplitude Margin')
    ax.legend()

    # DC Gain
    ax = axes[1, 1]
    for n in names:
        if ac_data[n]['dc_gain'] is not None and len(ac_data[n]['dc_gain']) > 0:
            ax.hist(ac_data[n]['dc_gain'], bins=40, alpha=0.5, label=n, color=colors[n])
    ax.set_xlabel('DC Gain (dB)')
    ax.set_ylabel('Count')
    ax.set_title('DC Gain')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'ac_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 8. Correlation Analysis
    # =========================================================================
    print("[8/8] Correlation analysis...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, n in enumerate(names):
        dd = device_data[n]
        ac = ac_data[n]
        par = params[n]

        # UGBW vs I_BIAS
        ax = axes[0, idx*1]  # Use first column for each dataset
        if ac['ugbw'] is not None and 'I_BIAS' in par:
            # Need to match AC valid samples to params
            valid_mask = []
            for s in datasets[n]['train']:
                g = s['graph']
                valid_mask.append(hasattr(g, 'ac_valid') and g.ac_valid is not None and g.ac_valid.item())
            valid_mask = np.array(valid_mask)
            ibias = par['I_BIAS'][valid_mask] * 1e6
            if len(ibias) == len(ac['ugbw']):
                ax.scatter(ibias, np.log10(np.clip(ac['ugbw'], 1, None)), alpha=0.2, s=5, color=colors[n])
                ax.set_xlabel('I_BIAS (µA)')
                ax.set_ylabel('log10(UGBW)')
                ax.set_title(f'{n}: UGBW vs I_BIAS')

    # PM vs UGBW (stability tradeoff)
    ax = axes[0, 2]
    for n in names:
        ac = ac_data[n]
        if ac['ugbw'] is not None and ac['pm'] is not None:
            ax.scatter(np.log10(np.clip(ac['ugbw'], 1, None)), ac['pm'],
                      alpha=0.2, s=5, label=n, color=colors[n])
    ax.set_xlabel('log10(UGBW)')
    ax.set_ylabel('Phase Margin (°)')
    ax.set_title('Stability vs Speed Tradeoff')
    ax.axhline(60, color='green', linestyle='--', alpha=0.5)
    ax.legend()

    # Vov distribution (Vgs - Vth) per device type
    for idx, n in enumerate(names):
        ax = axes[1, idx]
        dd = device_data[n]
        nmos_vov, pmos_vov = [], []
        for dev in devices:
            vgs = np.array(dd[dev]['vgs'])
            vth = np.array(dd[dev]['vth'])
            if dd[dev]['is_pmos']:
                vov = vgs - vth  # For PMOS: |Vgs| - |Vth|
                pmos_vov.extend(vov.tolist())
            else:
                vov = vgs - vth
                nmos_vov.extend(vov.tolist())

        ax.hist(nmos_vov, bins=40, alpha=0.5, label='NMOS', color='#2196F3')
        ax.hist(pmos_vov, bins=40, alpha=0.5, label='PMOS', color='#E91E63')
        ax.set_xlabel('Vov = Vgs - Vth (V)')
        ax.set_ylabel('Count')
        ax.set_title(f'{n}: Overdrive Voltage Distribution')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Vov=0 (cutoff)')
        ax.legend()

    # Supply current vs params
    ax = axes[1, 2]
    for n in names:
        par = params[n]
        supply_currents = []
        for s in datasets[n]['train']:
            specs = s.get('specs', {})
            ic = specs.get('_all_device_currents', {})
            vdd_i = abs(ic.get('vdd', 0))
            supply_currents.append(vdd_i * 1e6)
        supply_currents = np.array(supply_currents)
        ibias = par.get('I_BIAS', np.zeros(len(supply_currents))) * 1e6
        ax.scatter(ibias, supply_currents, alpha=0.2, s=5, label=n, color=colors[n])
    ax.set_xlabel('I_BIAS (µA)')
    ax.set_ylabel('Total Supply Current (µA)')
    ax.set_title('Supply Current vs I_BIAS')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "="*70)
    print("  DATASET COMPARISON SUMMARY")
    print("="*70)

    for n in names:
        print(f"\n--- {n} ---")
        d = datasets[n]
        print(f"  Train: {len(d['train'])} samples, Val: {len(d['val'])} samples")
        print(f"  Nodes: {d['train'][0]['graph'].num_nodes}, Edges: {d['train'][0]['graph'].edge_index.shape[1]}")
        print(f"  MOSFETs: {d['train'][0]['graph'].mosfet_info.shape[0]}")

        # Voltage stats
        all_vdc = torch.stack([s['graph'].vdc for s in d['train']])
        print(f"  VDC: mean={all_vdc.mean():.4f}, std={all_vdc.std():.4f}, range=[{all_vdc.min():.4f}, {all_vdc.max():.4f}]")

        # Region breakdown
        dd = device_data[n]
        total_sat = sum(sum(1 for r in dd[dev]['region'] if r == 'saturation') for dev in devices)
        total_tri = sum(sum(1 for r in dd[dev]['region'] if r == 'triode') for dev in devices)
        total_cut = sum(sum(1 for r in dd[dev]['region'] if r == 'cutoff') for dev in devices)
        total = total_sat + total_tri + total_cut
        print(f"  Regions: sat={100*total_sat/total:.1f}%, triode={100*total_tri/total:.1f}%, cutoff={100*total_cut/total:.1f}%")

        # AC stats
        ac = ac_data[n]
        print(f"  AC valid: {ac['valid_count']}/{ac['total']} ({ac['valid_pct']:.1f}%)")
        if ac['ugbw'] is not None and len(ac['ugbw']) > 0:
            print(f"  UGBW: median={np.median(ac['ugbw']):.0f} Hz ({np.median(np.log10(np.clip(ac['ugbw'],1,None))):.2f} log)")
            print(f"  PM:   median={np.median(ac['pm']):.1f}°, >60°: {100*np.mean(ac['pm']>60):.1f}%")
            if ac['am'] is not None and len(ac['am']) > 0:
                print(f"  AM:   median={np.median(ac['am']):.1f} dB")

        # SS stats
        ss = ss_data[n]
        gm_gds = ss['all_gm'] / (ss['all_gds'] + 1e-15)
        print(f"  gm: median={np.median(ss['all_gm']):.2e}, range=[{ss['all_gm'].min():.2e}, {ss['all_gm'].max():.2e}]")
        print(f"  gds: median={np.median(ss['all_gds']):.2e}, range=[{ss['all_gds'].min():.2e}, {ss['all_gds'].max():.2e}]")
        print(f"  Intrinsic gain (gm/gds): median={np.median(gm_gds):.0f}, range=[{gm_gds.min():.0f}, {gm_gds.max():.0f}]")

        # Param ranges
        par = params[n]
        print(f"  I_BIAS: [{par['I_BIAS'].min()*1e6:.1f}, {par['I_BIAS'].max()*1e6:.1f}] µA")

        # Devices in triode that shouldn't be
        # (Typically input pair, cascode, and current mirrors should be in saturation)
        print(f"\n  Devices frequently in triode (>10% of samples):")
        for dev in devices:
            tri_pct = 100 * sum(1 for r in dd[dev]['region'] if r == 'triode') / len(dd[dev]['region'])
            if tri_pct > 10:
                print(f"    {dev}: {tri_pct:.1f}% triode")

        print(f"\n  Devices in cutoff (>1% of samples):")
        for dev in devices:
            cut_pct = 100 * sum(1 for r in dd[dev]['region'] if r == 'cutoff') / len(dd[dev]['region'])
            if cut_pct > 1:
                print(f"    {dev}: {cut_pct:.1f}% cutoff")

    # =========================================================================
    # Key Differences
    # =========================================================================
    print("\n" + "="*70)
    print("  KEY DIFFERENCES")
    print("="*70)

    # Check if 5k_fixed filters out certain samples
    ac0 = ac_data[names[0]]
    ac1 = ac_data[names[1]]
    print(f"\n  AC convergence: {names[0]}={ac0['valid_pct']:.1f}% vs {names[1]}={ac1['valid_pct']:.1f}%")

    # Region distribution difference
    for n in names:
        dd = device_data[n]
        total_tri = sum(sum(1 for r in dd[dev]['region'] if r == 'triode') for dev in devices)
        total = sum(len(dd[dev]['region']) for dev in devices)
        print(f"  {n} overall triode: {100*total_tri/total:.1f}%")

    print(f"\nAll plots saved to {output_dir}/")


def main():
    datasets = {}

    ds_paths = {
        '2k_nofil': 'datasets/opamp_3stage_fan_smc_v7_2k_nofil',
        '5k_fixed': 'datasets/opamp_3stage_fan_smc_v7_5k_fixed',
    }

    for name, path in ds_paths.items():
        print(f"Loading {name}...")
        train, val = load_dataset(path)
        datasets[name] = {'train': train, 'val': val}

    output_dir = Path('datasets/dataset_analysis')
    plot_comparison(datasets, output_dir)


if __name__ == '__main__':
    main()
