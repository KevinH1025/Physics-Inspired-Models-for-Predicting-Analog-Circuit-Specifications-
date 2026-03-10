#!/usr/bin/env python3
"""Deep dive analysis: operating regions, AC convergence, cutoff/triode patterns."""

import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Device-to-group mapping for the 3-stage fan-smc opamp
DEVICE_GROUPS = {
    'xm0': 'BIASCM_P',  'xm1': 'BIASCM_P',  'xm2': 'BIASCM_P',  'xm3': 'BIASCM_P',
    'xm4': 'BIASCM_P',  'xm5': 'BIASCM_P',  'xm6': 'BIASCM_P',  'xm7': 'BIASCM_P',
    'xm8': 'GM1',        'xm9': 'GM1',
    'xm10': 'GM2',
    'xm11': 'GMF2',
    'xm12': 'BIASCM_N',  'xm13': 'BIASCM_N',  'xm14': 'BIASCM_N',
    'xm15': 'BIASCM_N',  'xm16': 'BIASCM_N',
    'xm17': 'BIASCM_N',  'xm18': 'BIASCM_N',
    'xm19': 'BIASCM_N',  'xm20': 'BIASCM_N',
    'xm21': 'LOAD2',     'xm22': 'LOAD2',
    'xm23': 'GM3',
}

DEVICE_ROLES = {
    'xm0': 'PMOS diode bias ref',
    'xm1': 'PMOS mirror→VB4',      'xm2': 'PMOS mirror→DM_1',
    'xm3': 'PMOS mirror→VB3',      'xm4': 'PMOS tail current (diff pair)',
    'xm5': 'PMOS diode VOUTN (CMFB)', 'xm6': 'PMOS mirror→net050',
    'xm7': 'PMOS mirror→net049',
    'xm8': 'Input diff pair (VINN)', 'xm9': 'Input diff pair (VINP)',
    'xm10': 'Stage2 CS amp',
    'xm11': 'Stage3 PMOS output',
    'xm12': 'NMOS cascode VB4 (4x)', 'xm13': 'NMOS cascode DM_1 (4x)',
    'xm14': 'NMOS diode→VB3',
    'xm15': 'NMOS cascode left (4x)→VOUTN', 'xm16': 'NMOS cascode right (4x)→net050',
    'xm17': 'NMOS bottom VB4 cascode (4x)', 'xm18': 'NMOS bottom DM_1 cascode (4x)',
    'xm19': 'NMOS fold-cascode load left (8x)', 'xm20': 'NMOS fold-cascode load right (8x)',
    'xm21': 'NMOS diode load stage2', 'xm22': 'NMOS mirror load→net049',
    'xm23': 'Stage3 NMOS output',
}

def load_datasets():
    datasets = {}
    for name, path in [('2k_nofil', 'datasets/opamp_3stage_fan_smc_v7_2k_nofil'),
                        ('5k_fixed', 'datasets/opamp_3stage_fan_smc_v7_5k_fixed')]:
        train = pickle.load(open(Path(path) / 'dataset_train.pkl', 'rb'))
        val = pickle.load(open(Path(path) / 'dataset_val.pkl', 'rb'))
        datasets[name] = {'train': train, 'val': val}
    return datasets


def extract_sample_data(samples):
    """Extract per-sample: regions, AC, params, operating points."""
    data = []
    for s in samples:
        specs = s.get('specs', {})
        regions = specs.get('_mosfet_regions', {})
        currents = specs.get('_all_device_currents', {})
        ss = specs.get('_mosfet_ss_params', {})
        g = s['graph']
        ac_valid = hasattr(g, 'ac_valid') and g.ac_valid is not None and g.ac_valid.item()

        entry = {
            'params': s['params'],
            'ac_valid': ac_valid,
            'ac_ugbw': g.ac_ugbw.item() if ac_valid and hasattr(g, 'ac_ugbw') and g.ac_ugbw is not None else None,
            'ac_pm': g.ac_pm.item() if ac_valid and hasattr(g, 'ac_pm') and g.ac_pm is not None else None,
            'ac_am': g.ac_am.item() if ac_valid and hasattr(g, 'ac_am') and g.ac_am is not None else None,
            'regions': {dev: info['region'] for dev, info in regions.items()},
            'vgs': {dev: info['vgs'] for dev, info in regions.items()},
            'vds': {dev: abs(info['vds']) for dev, info in regions.items()},
            'vdsat': {dev: info['vdsat'] for dev, info in regions.items()},
            'vth': {dev: info['vth'] for dev, info in regions.items()},
            'is_pmos': {dev: info['is_pmos'] for dev, info in regions.items()},
            'currents': {dev: abs(v) for dev, v in currents.items() if dev.startswith('xm')},
            'supply_current': abs(currents.get('vdd', 0)),
            'n_cutoff': sum(1 for info in regions.values() if info['region'] == 'cutoff'),
            'n_triode': sum(1 for info in regions.values() if info['region'] == 'triode'),
        }
        data.append(entry)
    return data


def plot_deepdive(datasets, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(datasets.keys())
    colors = {'2k_nofil': '#2196F3', '5k_fixed': '#FF5722'}

    sample_data = {n: extract_sample_data(d['train']) for n, d in datasets.items()}
    devices = sorted(DEVICE_GROUPS.keys())

    # =========================================================================
    # PLOT 1: Cutoff count vs AC validity
    # =========================================================================
    print("[1/7] Cutoff count vs AC validity...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, n in enumerate(names):
        ax = axes[idx]
        sd = sample_data[n]
        n_cutoffs = [s['n_cutoff'] for s in sd]
        ac_valids = [s['ac_valid'] for s in sd]

        max_cutoff = max(n_cutoffs)
        cutoff_range = range(max_cutoff + 1)
        total_per_bin = [sum(1 for nc in n_cutoffs if nc == c) for c in cutoff_range]
        valid_per_bin = [sum(1 for nc, av in zip(n_cutoffs, ac_valids) if nc == c and av) for c in cutoff_range]
        pct_valid = [100 * v / t if t > 0 else 0 for v, t in zip(valid_per_bin, total_per_bin)]

        ax2 = ax.twinx()
        bars = ax.bar(list(cutoff_range), total_per_bin, alpha=0.4, color=colors[n], label='Total samples')
        ax.bar(list(cutoff_range), valid_per_bin, alpha=0.7, color=colors[n], label='AC valid')
        ax2.plot(list(cutoff_range), pct_valid, 'ko-', markersize=4, label='% AC valid')
        ax2.set_ylabel('% AC valid')
        ax2.set_ylim(0, 105)

        ax.set_xlabel('Number of devices in cutoff')
        ax.set_ylabel('Number of samples')
        ax.set_title(f'{n}: Cutoff Count vs AC Validity')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'cutoff_vs_ac.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 2: Which devices cause AC failure?
    # =========================================================================
    print("[2/7] Per-device cutoff rate: AC valid vs invalid...")
    n = '2k_nofil'  # Only nofil has AC failures
    sd = sample_data[n]

    fig, ax = plt.subplots(figsize=(16, 6))
    ac_valid_samples = [s for s in sd if s['ac_valid']]
    ac_invalid_samples = [s for s in sd if not s['ac_valid']]

    x = np.arange(len(devices))
    width = 0.35

    cutoff_valid = []
    cutoff_invalid = []
    for dev in devices:
        n_valid = len(ac_valid_samples)
        n_invalid = len(ac_invalid_samples)
        cut_v = sum(1 for s in ac_valid_samples if s['regions'].get(dev) == 'cutoff')
        cut_i = sum(1 for s in ac_invalid_samples if s['regions'].get(dev) == 'cutoff')
        cutoff_valid.append(100 * cut_v / n_valid if n_valid > 0 else 0)
        cutoff_invalid.append(100 * cut_i / n_invalid if n_invalid > 0 else 0)

    bars1 = ax.bar(x - width/2, cutoff_valid, width, label='AC valid samples', color='#4CAF50', alpha=0.7)
    bars2 = ax.bar(x + width/2, cutoff_invalid, width, label='AC invalid samples', color='#F44336', alpha=0.7)

    ax.set_xticks(x)
    labels = [f'{dev}\n({DEVICE_ROLES[dev][:20]})' for dev in devices]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('% of samples where device is in cutoff')
    ax.set_title('2k_nofil: Per-Device Cutoff Rate — AC Valid vs Invalid Samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'cutoff_ac_valid_vs_invalid.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 3: Co-occurrence of cutoff devices
    # =========================================================================
    print("[3/7] Cutoff co-occurrence heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for idx, n in enumerate(names):
        ax = axes[idx]
        sd = sample_data[n]
        N = len(devices)
        cooccur = np.zeros((N, N))

        for s in sd:
            cutoff_devs = [i for i, dev in enumerate(devices) if s['regions'].get(dev) == 'cutoff']
            for i in cutoff_devs:
                for j in cutoff_devs:
                    cooccur[i, j] += 1

        # Normalize to percentage of total samples
        cooccur_pct = 100 * cooccur / len(sd)

        im = ax.imshow(cooccur_pct, cmap='Reds', aspect='auto')
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(devices, rotation=90, fontsize=7)
        ax.set_yticklabels(devices, fontsize=7)
        ax.set_title(f'{n}: Cutoff Co-occurrence (% of samples)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'cutoff_cooccurrence.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 4: What parameters cause cutoff? (2k_nofil)
    # =========================================================================
    print("[4/7] Parameter correlation with cutoff...")
    n = '2k_nofil'
    sd = sample_data[n]

    # Key parameters to check
    param_keys = ['I_BIAS', 'W_GM1', 'L_GM1', 'W_BIASCM_N', 'L_BIASCM_N',
                  'W_BIASCM_P', 'L_BIASCM_P', 'W_GM2', 'W_GM3', 'W_LOAD2',
                  'C_COMP', 'R_F', 'VCM']

    fig, axes = plt.subplots(3, 5, figsize=(25, 14))
    axes = axes.flatten()

    for i, pk in enumerate(param_keys):
        ax = axes[i]
        vals_0cut = [s['params'][pk] for s in sd if s['n_cutoff'] == 0]
        vals_any_cut = [s['params'][pk] for s in sd if s['n_cutoff'] > 0]
        vals_many_cut = [s['params'][pk] for s in sd if s['n_cutoff'] >= 5]

        # Scale
        scale = 1
        unit = ''
        if 'I_BIAS' in pk:
            scale = 1e6; unit = ' (µA)'
        elif pk.startswith('W_') or pk.startswith('L_'):
            scale = 1e6; unit = ' (µm)'
        elif 'C_' in pk:
            scale = 1e12; unit = ' (pF)'
        elif 'R_' in pk:
            scale = 1e-3; unit = ' (kΩ)'

        bins = 30
        ax.hist([v*scale for v in vals_0cut], bins=bins, alpha=0.4, label=f'0 cutoff (n={len(vals_0cut)})', color='#4CAF50')
        ax.hist([v*scale for v in vals_any_cut], bins=bins, alpha=0.4, label=f'≥1 cutoff (n={len(vals_any_cut)})', color='#F44336')
        if vals_many_cut:
            ax.hist([v*scale for v in vals_many_cut], bins=bins, alpha=0.4, label=f'≥5 cutoff (n={len(vals_many_cut)})', color='#9C27B0')
        ax.set_xlabel(f'{pk}{unit}')
        ax.set_title(pk)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(param_keys), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('2k_nofil: Parameter Distributions by Cutoff Count', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'params_vs_cutoff.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 5: Triode analysis — Vds/Vdsat margin per device
    # =========================================================================
    print("[5/7] Triode margin analysis...")
    fig, axes = plt.subplots(2, len(names), figsize=(14, 10))

    triode_devices = ['xm4', 'xm19', 'xm20', 'xm17', 'xm18', 'xm22', 'xm23']

    for idx, n in enumerate(names):
        sd = sample_data[n]

        # Top row: Vds/Vdsat ratio boxplot for triode-prone devices
        ax = axes[0, idx]
        ratios_per_dev = []
        dev_labels = []
        for dev in triode_devices:
            vds = [s['vds'].get(dev, 0) for s in sd]
            vdsat = [s['vdsat'].get(dev, 1) for s in sd]
            ratio = [d / (s + 1e-12) for d, s in zip(vds, vdsat)]
            ratios_per_dev.append(ratio)
            dev_labels.append(f'{dev}\n({DEVICE_ROLES[dev][:18]})')

        bp = ax.boxplot(ratios_per_dev, tick_labels=dev_labels, patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[n])
            patch.set_alpha(0.5)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Vds=Vdsat (sat/triode boundary)')
        ax.set_ylabel('Vds / Vdsat')
        ax.set_title(f'{n}: Saturation Margin (triode-prone devices)')
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=45, labelsize=7)

        # Bottom row: histogram of Vds/Vdsat for xm4 (most triode)
        ax = axes[1, idx]
        vds_4 = [s['vds'].get('xm4', 0) for s in sd]
        vdsat_4 = [s['vdsat'].get('xm4', 1) for s in sd]
        ratio_4 = [d / (s + 1e-12) for d, s in zip(vds_4, vdsat_4)]
        regions_4 = [s['regions'].get('xm4', 'unknown') for s in sd]

        sat_r = [r for r, reg in zip(ratio_4, regions_4) if reg == 'saturation']
        tri_r = [r for r, reg in zip(ratio_4, regions_4) if reg == 'triode']
        cut_r = [r for r, reg in zip(ratio_4, regions_4) if reg == 'cutoff']

        ax.hist(sat_r, bins=40, alpha=0.5, label=f'saturation (n={len(sat_r)})', color='#4CAF50')
        ax.hist(tri_r, bins=40, alpha=0.5, label=f'triode (n={len(tri_r)})', color='#FF9800')
        if cut_r:
            ax.hist(cut_r, bins=40, alpha=0.5, label=f'cutoff (n={len(cut_r)})', color='#F44336')
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Vds / Vdsat for xm4')
        ax.set_ylabel('Count')
        ax.set_title(f'{n}: xm4 (tail current) Vds/Vdsat distribution')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'triode_margin.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 6: AC performance vs number of cutoff/triode devices
    # =========================================================================
    print("[6/7] AC performance degradation with non-saturation devices...")
    n = '2k_nofil'
    sd = sample_data[n]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Group samples by number of cutoff devices
    for metric_idx, (metric, label, log_scale) in enumerate([
        ('ac_ugbw', 'UGBW (Hz)', True),
        ('ac_pm', 'Phase Margin (°)', False),
        ('ac_am', 'Amplitude Margin (dB)', False),
    ]):
        # vs cutoff count
        ax = axes[0, metric_idx]
        cutoff_groups = defaultdict(list)
        for s in sd:
            if s['ac_valid'] and s[metric] is not None:
                nc = min(s['n_cutoff'], 8)  # cap at 8+
                cutoff_groups[nc].append(s[metric])

        positions = sorted(cutoff_groups.keys())
        data_lists = [cutoff_groups[p] for p in positions]
        if data_lists:
            bp = ax.boxplot(data_lists, positions=positions, tick_labels=[str(p) if p < 8 else '8+' for p in positions],
                           patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('#2196F3')
                patch.set_alpha(0.5)
        if log_scale:
            ax.set_yscale('log')
        ax.set_xlabel('# devices in cutoff')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Cutoff Count')

        # vs triode count
        ax = axes[1, metric_idx]
        triode_groups = defaultdict(list)
        for s in sd:
            if s['ac_valid'] and s[metric] is not None:
                nt = min(s['n_triode'], 8)
                triode_groups[nt].append(s[metric])

        positions = sorted(triode_groups.keys())
        data_lists = [triode_groups[p] for p in positions]
        if data_lists:
            bp = ax.boxplot(data_lists, positions=positions, tick_labels=[str(p) if p < 8 else '8+' for p in positions],
                           patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('#FF9800')
                patch.set_alpha(0.5)
        if log_scale:
            ax.set_yscale('log')
        ax.set_xlabel('# devices in triode')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Triode Count')

    fig.suptitle('2k_nofil: AC Performance vs Operating Region Violations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ac_vs_regions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 7: Filtering effect — what exactly does 5k_fixed filter out?
    # =========================================================================
    print("[7/7] Filtering analysis — what 5k_fixed removes...")
    n = '2k_nofil'
    sd = sample_data[n]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Classify samples as "would pass filter" vs "would fail filter"
    # 5k_fixed has 100% AC valid and ~5% cutoff → filter removes high-cutoff samples
    pass_filter = [s for s in sd if s['n_cutoff'] <= 2]  # approximate filter
    fail_filter = [s for s in sd if s['n_cutoff'] > 2]

    print(f"  Approximate filter: {len(pass_filter)} pass ({100*len(pass_filter)/len(sd):.1f}%), "
          f"{len(fail_filter)} fail ({100*len(fail_filter)/len(sd):.1f}%)")

    # 1. Region distribution: pass vs fail
    ax = axes[0, 0]
    for label, group, color in [('Pass filter', pass_filter, '#4CAF50'), ('Fail filter', fail_filter, '#F44336')]:
        total_sat = sum(sum(1 for dev in devices if s['regions'].get(dev) == 'saturation') for s in group)
        total_tri = sum(sum(1 for dev in devices if s['regions'].get(dev) == 'triode') for s in group)
        total_cut = sum(sum(1 for dev in devices if s['regions'].get(dev) == 'cutoff') for s in group)
        total = total_sat + total_tri + total_cut
        bars = ax.bar(label, [100*total_sat/total, 100*total_tri/total, 100*total_cut/total],
                      color=['#4CAF50', '#FF9800', '#F44336'], alpha=0.7)

    ax.set_ylabel('% of device-instances')
    ax.set_title('Region Distribution')

    # Actually make a grouped bar chart
    ax.clear()
    x = np.arange(3)
    width = 0.35
    for g_idx, (label, group, color) in enumerate([('Pass', pass_filter, '#4CAF50'), ('Fail', fail_filter, '#F44336')]):
        total_sat = sum(sum(1 for dev in devices if s['regions'].get(dev) == 'saturation') for s in group)
        total_tri = sum(sum(1 for dev in devices if s['regions'].get(dev) == 'triode') for s in group)
        total_cut = sum(sum(1 for dev in devices if s['regions'].get(dev) == 'cutoff') for s in group)
        total = total_sat + total_tri + total_cut
        vals = [100*total_sat/total, 100*total_tri/total, 100*total_cut/total]
        ax.bar(x + g_idx * width, vals, width, label=label, color=color, alpha=0.7)

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(['Saturation', 'Triode', 'Cutoff'])
    ax.set_ylabel('% of device-instances')
    ax.set_title('Region Distribution: Pass vs Fail Filter')
    ax.legend()

    # 2. Supply current distribution
    ax = axes[0, 1]
    supply_pass = [s['supply_current'] * 1e6 for s in pass_filter]
    supply_fail = [s['supply_current'] * 1e6 for s in fail_filter]
    ax.hist(supply_pass, bins=40, alpha=0.5, label='Pass', color='#4CAF50')
    ax.hist(supply_fail, bins=40, alpha=0.5, label='Fail', color='#F44336')
    ax.set_xlabel('Total Supply Current (µA)')
    ax.set_title('Supply Current Distribution')
    ax.legend()

    # 3. I_BIAS distribution
    ax = axes[0, 2]
    ibias_pass = [s['params']['I_BIAS'] * 1e6 for s in pass_filter]
    ibias_fail = [s['params']['I_BIAS'] * 1e6 for s in fail_filter]
    ax.hist(ibias_pass, bins=40, alpha=0.5, label='Pass', color='#4CAF50')
    ax.hist(ibias_fail, bins=40, alpha=0.5, label='Fail', color='#F44336')
    ax.set_xlabel('I_BIAS (µA)')
    ax.set_title('I_BIAS Distribution')
    ax.legend()

    # 4. AC performance (only valid samples)
    ax = axes[1, 0]
    ugbw_pass = [s['ac_ugbw'] for s in pass_filter if s['ac_valid'] and s['ac_ugbw'] is not None]
    ugbw_fail = [s['ac_ugbw'] for s in fail_filter if s['ac_valid'] and s['ac_ugbw'] is not None]
    if ugbw_pass:
        ax.hist(np.log10(np.clip(ugbw_pass, 1, None)), bins=30, alpha=0.5, label=f'Pass (n={len(ugbw_pass)})', color='#4CAF50')
    if ugbw_fail:
        ax.hist(np.log10(np.clip(ugbw_fail, 1, None)), bins=30, alpha=0.5, label=f'Fail (n={len(ugbw_fail)})', color='#F44336')
    ax.set_xlabel('log10(UGBW)')
    ax.set_title('UGBW: Pass vs Fail')
    ax.legend()

    # 5. W_BIASCM_N vs I_BIAS colored by cutoff count
    ax = axes[1, 1]
    ibias = [s['params']['I_BIAS'] * 1e6 for s in sd]
    w_bn = [s['params']['W_BIASCM_N'] * 1e6 for s in sd]
    n_cut = [s['n_cutoff'] for s in sd]
    sc = ax.scatter(ibias, w_bn, c=n_cut, cmap='RdYlGn_r', s=8, alpha=0.5, vmin=0, vmax=10)
    plt.colorbar(sc, ax=ax, label='# devices in cutoff')
    ax.set_xlabel('I_BIAS (µA)')
    ax.set_ylabel('W_BIASCM_N (µm)')
    ax.set_title('Parameter Space: Cutoff Regions')

    # 6. W/L ratio for bias devices
    ax = axes[1, 2]
    wl_pmos = [s['params']['W_BIASCM_P'] / s['params']['L_BIASCM_P'] for s in sd]
    wl_nmos = [s['params']['W_BIASCM_N'] / s['params']['L_BIASCM_N'] for s in sd]
    ac_valid_mask = [s['ac_valid'] for s in sd]
    sc = ax.scatter(wl_pmos, wl_nmos, c=['#4CAF50' if v else '#F44336' for v in ac_valid_mask],
                    s=8, alpha=0.3)
    ax.set_xlabel('W/L BIASCM_P')
    ax.set_ylabel('W/L BIASCM_N')
    ax.set_title('Bias W/L Ratios (green=AC valid)')
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.suptitle('What Does Filtering Remove?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'filtering_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Print detailed statistics
    # =========================================================================
    print("\n" + "="*70)
    print("  DEEP DIVE FINDINGS")
    print("="*70)

    # Finding 1: Cutoff cascades
    print("\n[Finding 1] CUTOFF CASCADE PATTERNS")
    n = '2k_nofil'
    sd = sample_data[n]

    # Find which groups of devices go to cutoff together
    cutoff_patterns = defaultdict(int)
    for s in sd:
        cutoff_devs = frozenset(dev for dev in devices if s['regions'].get(dev) == 'cutoff')
        if cutoff_devs:
            # Group by circuit block
            blocks = frozenset(DEVICE_GROUPS.get(dev, '?') for dev in cutoff_devs)
            cutoff_patterns[blocks] += 1

    print("  Most common cutoff block patterns:")
    for pattern, count in sorted(cutoff_patterns.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / len(sd)
        print(f"    {set(pattern)}: {count} samples ({pct:.1f}%)")

    # Finding 2: Cutoff → AC failure correlation
    print("\n[Finding 2] CUTOFF → AC FAILURE CORRELATION")
    for nc in range(13):
        samples_with_nc = [s for s in sd if s['n_cutoff'] == nc]
        if not samples_with_nc:
            continue
        ac_valid_pct = 100 * sum(1 for s in samples_with_nc if s['ac_valid']) / len(samples_with_nc)
        print(f"    {nc} cutoff devices: {len(samples_with_nc)} samples, {ac_valid_pct:.0f}% AC valid")

    # Finding 3: Critical devices for AC
    print("\n[Finding 3] CRITICAL DEVICES FOR AC CONVERGENCE")
    print("  Device cutoff rate in AC-invalid vs AC-valid samples:")
    for dev in devices:
        cut_valid = sum(1 for s in sd if s['ac_valid'] and s['regions'].get(dev) == 'cutoff')
        cut_invalid = sum(1 for s in sd if not s['ac_valid'] and s['regions'].get(dev) == 'cutoff')
        n_valid = sum(1 for s in sd if s['ac_valid'])
        n_invalid = sum(1 for s in sd if not s['ac_valid'])
        if n_invalid > 0:
            rate_valid = 100 * cut_valid / n_valid
            rate_invalid = 100 * cut_invalid / n_invalid
            ratio = rate_invalid / (rate_valid + 0.01)
            if ratio > 1.5 or rate_invalid > 20:
                print(f"    {dev} ({DEVICE_ROLES[dev][:30]}): "
                      f"AC-valid={rate_valid:.1f}%, AC-invalid={rate_invalid:.1f}%, ratio={ratio:.1f}x")

    # Finding 4: Triode is structural, not random
    print("\n[Finding 4] TRIODE DEVICES — STRUCTURAL ANALYSIS")
    for dev in ['xm4', 'xm19', 'xm20', 'xm17', 'xm18']:
        for n in names:
            sd_n = sample_data[n]
            tri_pct = 100 * sum(1 for s in sd_n if s['regions'].get(dev) == 'triode') / len(sd_n)
            vds_vals = [s['vds'].get(dev, 0) for s in sd_n]
            vdsat_vals = [s['vdsat'].get(dev, 1) for s in sd_n]
            margin = [d - s for d, s in zip(vds_vals, vdsat_vals)]
            print(f"    {dev} ({DEVICE_ROLES[dev][:25]}) [{n}]: "
                  f"triode={tri_pct:.1f}%, median Vds-Vdsat={np.median(margin)*1e3:.1f}mV, "
                  f"min margin={min(margin)*1e3:.1f}mV")

    # Finding 5: Parameter space that causes problems
    print("\n[Finding 5] PARAMETER SPACE CAUSING PROBLEMS (2k_nofil)")
    sd = sample_data['2k_nofil']
    healthy = [s for s in sd if s['n_cutoff'] == 0 and s['ac_valid']]
    unhealthy = [s for s in sd if s['n_cutoff'] >= 5]

    for pk in ['I_BIAS', 'W_BIASCM_N', 'L_BIASCM_N', 'W_BIASCM_P', 'L_BIASCM_P', 'W_GM1', 'VCM']:
        h_vals = [s['params'][pk] for s in healthy]
        u_vals = [s['params'][pk] for s in unhealthy]
        scale = 1e6 if pk.startswith(('W_', 'L_', 'I_')) else 1
        unit = 'µm' if pk.startswith(('W_', 'L_')) else ('µA' if pk == 'I_BIAS' else 'V')
        if pk.startswith('I_'):
            unit = 'µA'
        print(f"    {pk}: healthy=[{np.median(h_vals)*scale:.2f}] vs unhealthy=[{np.median(u_vals)*scale:.2f}] {unit} "
              f"(median, n={len(healthy)} vs {len(unhealthy)})")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    datasets = load_datasets()
    plot_deepdive(datasets, Path('datasets/dataset_analysis/deepdive'))
