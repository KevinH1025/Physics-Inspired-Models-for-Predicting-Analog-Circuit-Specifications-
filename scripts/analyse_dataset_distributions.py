#!/usr/bin/env python3
"""Distribution plots for all prediction metrics: 2k_nofil vs 5k_fixed."""

import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_samples(path):
    path = Path(path)
    samples = pickle.load(open(path / 'dataset_train.pkl', 'rb'))
    samples += pickle.load(open(path / 'dataset_val.pkl', 'rb'))
    return samples

def main():
    datasets = {
        '2k_nofil': load_all_samples('datasets/opamp_3stage_fan_smc_v7_2k_nofil'),
        '5k_fixed': load_all_samples('datasets/opamp_3stage_fan_smc_v7_5k_fixed'),
    }
    colors = {'2k_nofil': '#2196F3', '5k_fixed': '#FF5722'}
    output_dir = Path('datasets/dataset_analysis/distributions')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get node info from first sample
    node_names = datasets['2k_nofil'][0]['graph'].node_names
    train_mask = datasets['2k_nofil'][0]['graph'].train_mask
    trainable_idx = torch.where(train_mask)[0]
    trainable_names = [node_names[i.item()] for i in trainable_idx]

    # =========================================================================
    # 1. Per-node DC voltage distributions
    # =========================================================================
    print("[1/7] Per-node voltage distributions...")
    n_nodes = len(trainable_names)
    ncols = 4
    nrows = (n_nodes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()

    for j, (ni, nname) in enumerate(zip(trainable_idx, trainable_names)):
        ax = axes[j]
        for name, samples in datasets.items():
            vals = [s['graph'].vdc[j].item() for s in samples]
            ax.hist(vals, bins=50, alpha=0.45, label=name, color=colors[name], density=True)
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Density')
        ax.set_title(nname, fontweight='bold')
        if j == 0:
            ax.legend(fontsize=8)

    for j in range(n_nodes, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('DC Voltage Distributions per Node', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '1_node_voltages.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 1_node_voltages.png")

    # =========================================================================
    # 2. Per-device current distributions
    # =========================================================================
    print("[2/7] Per-device current distributions...")
    mosfet_devs = sorted([k for k in datasets['2k_nofil'][0]['specs']['_all_device_currents'].keys()
                          if k.startswith('xm')])
    n_devs = len(mosfet_devs)
    ncols = 6
    nrows = (n_devs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()

    for j, dev in enumerate(mosfet_devs):
        ax = axes[j]
        for name, samples in datasets.items():
            vals = [abs(s['specs']['_all_device_currents'].get(dev, 0)) * 1e6 for s in samples]
            vals = [v for v in vals if v > 0.001]  # skip near-zero
            if vals:
                ax.hist(np.log10(vals), bins=50, alpha=0.45, label=name, color=colors[name], density=True)
        ax.set_xlabel('log10(|Id|) [µA]')
        ax.set_title(dev, fontweight='bold')
        if j == 0:
            ax.legend(fontsize=7)

    for j in range(n_devs, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Device Current Distributions (log scale)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '2_device_currents.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 2_device_currents.png")

    # Supply current
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, samples in datasets.items():
        vals = [abs(s['specs']['_all_device_currents'].get('vdd', 0)) * 1e6 for s in samples]
        ax.hist(np.log10(np.clip(vals, 1, None)), bins=50, alpha=0.45, label=name, color=colors[name], density=True)
    ax.set_xlabel('log10(Supply Current) [µA]')
    ax.set_ylabel('Density')
    ax.set_title('Total Supply Current Distribution', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '2b_supply_current.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 2b_supply_current.png")

    # =========================================================================
    # 3. Per-device gm distributions
    # =========================================================================
    print("[3/7] Per-device gm distributions...")
    ss_devs = sorted(datasets['2k_nofil'][0]['specs']['_mosfet_ss_params'].keys())
    n_ss = len(ss_devs)
    ncols = 6
    nrows = (n_ss + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()

    for j, dev in enumerate(ss_devs):
        ax = axes[j]
        for name, samples in datasets.items():
            vals = [s['specs']['_mosfet_ss_params'][dev]['gm'] for s in samples]
            vals = [v for v in vals if v > 1e-20]
            if vals:
                ax.hist(np.log10(vals), bins=50, alpha=0.45, label=name, color=colors[name], density=True)
        ax.set_xlabel('log10(gm)')
        ax.set_title(f'{dev} gm', fontweight='bold')
        if j == 0:
            ax.legend(fontsize=7)

    for j in range(n_ss, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Transconductance (gm) Distributions per Device', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '3_device_gm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 3_device_gm.png")

    # =========================================================================
    # 4. Per-device gds distributions
    # =========================================================================
    print("[4/7] Per-device gds distributions...")
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()

    for j, dev in enumerate(ss_devs):
        ax = axes[j]
        for name, samples in datasets.items():
            vals = [s['specs']['_mosfet_ss_params'][dev]['gds'] for s in samples]
            vals = [v for v in vals if v > 1e-20]
            if vals:
                ax.hist(np.log10(vals), bins=50, alpha=0.45, label=name, color=colors[name], density=True)
        ax.set_xlabel('log10(gds)')
        ax.set_title(f'{dev} gds', fontweight='bold')
        if j == 0:
            ax.legend(fontsize=7)

    for j in range(n_ss, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Output Conductance (gds) Distributions per Device', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '4_device_gds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 4_device_gds.png")

    # =========================================================================
    # 5. AC metrics distributions
    # =========================================================================
    print("[5/7] AC metric distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # UGBW
    ax = axes[0, 0]
    for name, samples in datasets.items():
        vals = [s['graph'].ac_ugbw.item() for s in samples
                if hasattr(s['graph'], 'ac_valid') and s['graph'].ac_valid is not None and s['graph'].ac_valid.item()
                and hasattr(s['graph'], 'ac_ugbw') and s['graph'].ac_ugbw is not None]
        if vals:
            ax.hist(np.log10(np.clip(vals, 1, None)), bins=60, alpha=0.45, label=name, color=colors[name], density=True)
    ax.set_xlabel('log10(UGBW) [Hz]')
    ax.set_ylabel('Density')
    ax.set_title('Unity Gain Bandwidth', fontweight='bold')
    ax.legend()

    # Phase Margin
    ax = axes[0, 1]
    for name, samples in datasets.items():
        vals = [s['graph'].ac_pm.item() for s in samples
                if hasattr(s['graph'], 'ac_valid') and s['graph'].ac_valid is not None and s['graph'].ac_valid.item()
                and hasattr(s['graph'], 'ac_pm') and s['graph'].ac_pm is not None]
        if vals:
            ax.hist(vals, bins=60, alpha=0.45, label=name, color=colors[name], density=True)
    ax.set_xlabel('Phase Margin (°)')
    ax.set_ylabel('Density')
    ax.set_title('Phase Margin', fontweight='bold')
    ax.axvline(60, color='green', linestyle='--', alpha=0.5, label='60° target')
    ax.legend()

    # Amplitude Margin
    ax = axes[1, 0]
    for name, samples in datasets.items():
        vals = [s['graph'].ac_am.item() for s in samples
                if hasattr(s['graph'], 'ac_valid') and s['graph'].ac_valid is not None and s['graph'].ac_valid.item()
                and hasattr(s['graph'], 'ac_am') and s['graph'].ac_am is not None]
        if vals:
            ax.hist(vals, bins=60, alpha=0.45, label=name, color=colors[name], density=True)
    ax.set_xlabel('Amplitude Margin (dB)')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude Margin', fontweight='bold')
    ax.legend()

    # DC Gain
    ax = axes[1, 1]
    for name, samples in datasets.items():
        vals = [s['graph'].ac_dc_gain.item() for s in samples
                if hasattr(s['graph'], 'ac_valid') and s['graph'].ac_valid is not None and s['graph'].ac_valid.item()
                and hasattr(s['graph'], 'ac_dc_gain') and s['graph'].ac_dc_gain is not None]
        if vals:
            ax.hist(vals, bins=60, alpha=0.45, label=name, color=colors[name], density=True)
    ax.set_xlabel('DC Gain (dB)')
    ax.set_ylabel('Density')
    ax.set_title('DC Gain', fontweight='bold')
    ax.legend()

    fig.suptitle('AC Performance Metric Distributions', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '5_ac_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 5_ac_metrics.png")

    # =========================================================================
    # 6. Per-device Vov (overdrive voltage) distributions
    # =========================================================================
    print("[6/7] Per-device overdrive voltage (Vov) distributions...")
    region_devs = sorted(datasets['2k_nofil'][0]['specs']['_mosfet_regions'].keys())
    n_rd = len(region_devs)
    ncols = 6
    nrows = (n_rd + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()

    for j, dev in enumerate(region_devs):
        ax = axes[j]
        for name, samples in datasets.items():
            vov = [s['specs']['_mosfet_regions'][dev]['vgs'] - s['specs']['_mosfet_regions'][dev]['vth']
                   for s in samples if dev in s['specs']['_mosfet_regions']]
            if vov:
                ax.hist(vov, bins=50, alpha=0.45, label=name, color=colors[name], density=True)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Vov = Vgs - Vth (V)')
        ax.set_title(dev, fontweight='bold')
        if j == 0:
            ax.legend(fontsize=7)

    for j in range(n_rd, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Overdrive Voltage (Vov) Distributions per Device\n(red line = cutoff boundary)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '6_device_vov.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 6_device_vov.png")

    # =========================================================================
    # 7. Per-device Vds/Vdsat ratio distributions (saturation margin)
    # =========================================================================
    print("[7/7] Per-device saturation margin (Vds/Vdsat) distributions...")
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()

    for j, dev in enumerate(region_devs):
        ax = axes[j]
        for name, samples in datasets.items():
            ratios = []
            for s in samples:
                info = s['specs']['_mosfet_regions'].get(dev)
                if info and info['vdsat'] > 1e-6:
                    ratios.append(abs(info['vds']) / info['vdsat'])
            if ratios:
                # Clip for visualization
                ratios_clipped = np.clip(ratios, 0, 10)
                ax.hist(ratios_clipped, bins=50, alpha=0.45, label=name, color=colors[name], density=True)
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('|Vds| / Vdsat')
        ax.set_title(dev, fontweight='bold')
        if j == 0:
            ax.legend(fontsize=7)

    for j in range(n_rd, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Saturation Margin (|Vds|/Vdsat) Distributions per Device\n(red line = sat/triode boundary, >1 = saturation)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '7_device_sat_margin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 7_device_sat_margin.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
