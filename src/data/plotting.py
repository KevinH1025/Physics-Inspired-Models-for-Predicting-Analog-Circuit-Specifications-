"""
Visualization utilities for dataset analysis.

This module provides plotting functions for parameter distributions,
voltage distributions, current distributions, and normalization analysis.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_parameter_distributions(samples: List[Dict], param_specs: Dict, output_dir: Path) -> None:
    """Plot distribution of all sampling parameters."""
    var_params = [name for name, spec in param_specs.items()
                  if isinstance(spec, dict) and 'min' in spec and 'max' in spec]

    if not var_params:
        print("No variable parameters to plot")
        return

    n_params = len(var_params)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]

    for i, param_name in enumerate(var_params):
        ax = axes[i]
        spec = param_specs[param_name]
        values = [s['params'].get(param_name, 0) for s in samples]

        scale = spec.get('scale', 'linear')

        if scale == 'log':
            log_values = np.log10(values)
            ax.hist(log_values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel(f'{param_name} (log10)')
            ax.axvline(np.log10(spec['min']), color='r', linestyle='--', alpha=0.5, label='min')
            ax.axvline(np.log10(spec['max']), color='r', linestyle='--', alpha=0.5, label='max')
        else:
            ax.hist(values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel(param_name)
            ax.axvline(spec['min'], color='r', linestyle='--', alpha=0.5)
            ax.axvline(spec['max'], color='r', linestyle='--', alpha=0.5)

        ax.set_ylabel('Count')
        ax.set_title(f'{param_name} ({scale})')
        ax.grid(True, alpha=0.3)

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Parameter Distributions (N={len(samples)} samples)', fontweight='bold', fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / 'param_distributions.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved parameter distribution plot to {plot_path}")


def plot_node_voltage_distributions(samples: List[Dict], output_dir: Path) -> None:
    """Plot distribution of predicted unknown net node voltages."""
    node_voltages_per_sample = []
    net_names = None

    for sample in samples:
        graph = sample['graph']
        vdc = graph.vdc.flatten().numpy()
        node_voltages_per_sample.append(vdc)

        if net_names is None:
            net_names = []
            for i, ntype in enumerate(graph.node_types):
                if graph.output_node_mask[i]:
                    net_names.append(f"Net{i}")

    if not node_voltages_per_sample:
        print("No voltage data to plot")
        return

    all_voltages = np.array(node_voltages_per_sample)
    n_nodes = all_voltages.shape[1]

    n_cols = 4
    n_rows = (n_nodes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if n_nodes > 1 else [axes]

    for i in range(n_nodes):
        ax = axes[i]
        voltages = all_voltages[:, i]

        if np.std(voltages) < 0.001:
            ax.bar([0], [len(samples)], color='gray', alpha=0.7)
            ax.set_title(f'Node {i}: CONSTANT ({voltages[0]:.3f}V)')
            ax.set_xticks([0])
            ax.set_xticklabels([f'{voltages[0]:.3f}V'])
        else:
            ax.hist(voltages, bins=20, edgecolor='black', alpha=0.7, color='coral')
            ax.set_xlabel('Voltage (V)')
            ax.set_title(f'Node {i}: mean={np.mean(voltages):.3f}V, std={np.std(voltages):.3f}V')

        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    for i in range(n_nodes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Unknown Node Voltage Distributions (N={len(samples)} samples)', fontweight='bold', fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / 'node_voltage_distributions.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved node voltage distribution plot to {plot_path}")


def plot_device_current_distributions(samples: List[Dict], output_dir: Path) -> None:
    """Plot distribution of device currents across all samples."""
    device_currents = defaultdict(list)

    for sample in samples:
        specs = sample.get('specs', {})

        if '_all_device_currents' in specs:
            for dev_name, current in specs['_all_device_currents'].items():
                dev_lower = dev_name.lower()
                current_abs = abs(current)

                if dev_lower.startswith('xm'):
                    device_currents[dev_name.upper()].append(current_abs)
                elif dev_lower in ['rin', 'rfb', 'rz']:
                    device_currents[dev_name.capitalize()].append(current_abs)
                elif dev_lower in ['vdd', 'vss', 'vsig', 'vcm_ref']:
                    device_currents[dev_name.upper()].append(current_abs)
                elif dev_lower == 'iref':
                    device_currents['IREF'].append(current_abs)

    if not device_currents:
        print("No current data to plot")
        return

    device_types = sorted(device_currents.keys())
    n_devices = len(device_types)

    if n_devices == 0:
        print("No current data to plot")
        return

    n_cols = min(4, n_devices)
    n_rows = (n_devices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    if n_devices == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, dev_type in enumerate(device_types):
        ax = axes[idx]
        currents = np.array(device_currents[dev_type]) * 1e6

        if currents.max() / (currents.min() + 1e-12) > 100:
            currents_log = np.log10(currents + 1e-12)
            ax.hist(currents_log, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('Log10(Current) [log10(µA)]')
            ax.set_title(f'{dev_type}: med={np.median(currents):.2f}µA')
        else:
            ax.hist(currents, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('Current [µA]')
            ax.set_title(f'{dev_type}: mean={np.mean(currents):.2f}µA, std={np.std(currents):.2f}µA')

        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    for i in range(n_devices, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Device Current Distributions (N={len(samples)} samples)', fontweight='bold', fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / 'device_current_distributions.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved device current distribution plot to {plot_path}")


def plot_target_normalization_distributions(samples: List[Dict], output_dir: Path) -> None:
    """Plot target (vdc, current) distributions showing raw and z-score normalized versions."""
    current_values = []
    vdc_values = []

    EXCLUDED_CURRENT_DEVICES = {'rz', 'vcm_ref'}

    for sample in samples:
        graph = sample['graph']
        specs = sample.get('specs', {})

        if hasattr(graph, 'vdc') and graph.vdc is not None:
            vdc_values.extend(graph.vdc.cpu().numpy().flatten())

        device_currents = specs.get('_all_device_currents', {})
        for dev_name, current in device_currents.items():
            if dev_name.lower() not in EXCLUDED_CURRENT_DEVICES:
                current_values.append(abs(current))

    if not current_values or not vdc_values:
        print("No target data to plot for normalization distributions")
        return

    current_arr = np.array(current_values)
    vdc_arr = np.array(vdc_values)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: Currents
    axes[0, 0].hist(current_arr * 1e6, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Current (µA)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Current Distribution (Linear)')
    axes[0, 0].axvline(current_arr.mean()*1e6, color='r', linestyle='--', label=f'Mean: {current_arr.mean()*1e6:.1f}µA')
    axes[0, 0].legend()

    log_epsilon = 1e-12
    log_current = np.log10(current_arr + log_epsilon)
    axes[0, 1].hist(log_current, bins=100, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('log10(Current in A)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Current Distribution (Log Scale)')
    axes[0, 1].axvline(log_current.mean(), color='r', linestyle='--', label=f'Mean: {log_current.mean():.2f}')
    axes[0, 1].legend()

    log_mean = log_current.mean()
    log_std = log_current.std()
    z_current = (log_current - log_mean) / log_std
    axes[0, 2].hist(z_current, bins=100, edgecolor='black', alpha=0.7, color='green')
    axes[0, 2].set_xlabel('Z-score (log)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title(f'Current: Log Z-score\nRange: [{z_current.min():.2f}, {z_current.max():.2f}]')
    axes[0, 2].axvline(0, color='r', linestyle='--', label='Mean: 0')
    axes[0, 2].legend()

    # Row 2: Voltages
    axes[1, 0].hist(vdc_arr, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].set_xlabel('Voltage (V)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Voltage Distribution (Linear)')
    axes[1, 0].axvline(vdc_arr.mean(), color='r', linestyle='--', label=f'Mean: {vdc_arr.mean():.2f}V')
    axes[1, 0].legend()

    axes[1, 1].hist(vdc_arr, bins=100, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Voltage (V)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Voltage: mean={vdc_arr.mean():.3f}V, std={vdc_arr.std():.3f}V')

    vdc_mean = vdc_arr.mean()
    vdc_std = vdc_arr.std()
    z_vdc = (vdc_arr - vdc_mean) / vdc_std
    axes[1, 2].hist(z_vdc, bins=100, edgecolor='black', alpha=0.7, color='teal')
    axes[1, 2].set_xlabel('Z-score')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title(f'Voltage: Z-score\nRange: [{z_vdc.min():.2f}, {z_vdc.max():.2f}]')
    axes[1, 2].axvline(0, color='r', linestyle='--', label='Mean: 0')
    axes[1, 2].legend()

    plt.suptitle('Target Normalization Distributions\n(rz, vcm_ref excluded from currents - always 0 DC by design)',
                 fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / 'target_normalization_distributions.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved target normalization plot to {plot_path}")
    print(f"  Current (log z-score): range=[{z_current.min():.2f}, {z_current.max():.2f}]")
    print(f"  Voltage (z-score): range=[{z_vdc.min():.2f}, {z_vdc.max():.2f}]")
