#!/usr/bin/env python3
"""
Generate dataset for circuit GNN training.

Data format:
- x: [num_nodes, max_props] - continuous properties (padded)
  - MOSFET terminals: [w] (width)
  - Voltage sources: [dc] (DC voltage value)
  - Resistors/Capacitors: [value]
  - VNodes (nets): [] empty
- type_tens: [num_nodes, type_dim] - hierarchical one-hot encoding
- vdc: [num_output_nodes, 1] - DC voltages for output nodes
- output_node_mask: [num_nodes] - True for VNode NGND (prediction targets)

Key: The DC voltage of voltage sources IS an input feature!
"""

import sys
import gc
import pickle
import argparse
import random
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Add project root to path (same as train_v3.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from modular data package
from src.data.sampling import generate_lhs_samples, worker_generate_sample
from src.data.batching import create_prebatched_dataset
from src.data.plotting import (
    plot_parameter_distributions,
    plot_node_voltage_distributions,
    plot_device_current_distributions,
    plot_target_normalization_distributions,
)


def main():
    parser = argparse.ArgumentParser(description='Generate circuit dataset for GNN training')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='datasets/opamp_2stage_v2', help='Output directory')
    parser.add_argument('--template', type=str, default='netlists/opamp_2stage_template.sp', help='Netlist template')
    parser.add_argument('--config', type=str, default='configs/opamp_dataset/opamp_dataset_v2.yaml', help='YAML config file')
    parser.add_argument('--verify-only', action='store_true', help='Generate 5 samples and verify format')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    use_lhs = False
    filter_enabled = False
    rail_margin = 0.1
    exclude_nodes = set()
    saturation_filter = False
    max_cutoff_devices = None
    require_ac_convergence = False
    min_ugbw_hz = None
    save_interval = 1000
    param_specs = {}
    sampling_config = {}

    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        dataset_config = config.get('dataset', {})
        if 'template' in dataset_config:
            args.template = dataset_config['template']
        ac_template = dataset_config.get('ac_template', None)

        graph_encoding_config = config.get('graph_encoding', {})
        normalize_props = graph_encoding_config.get('normalize_props', False)
        device_to_group = config.get('device_to_group', None)
        excluded_current_devices = config.get('excluded_current_devices', None)
        if excluded_current_devices is not None:
            excluded_current_devices = set(s.lower() for s in excluded_current_devices)

        sampling_config = config.get('sampling', {})
        param_constraints = sampling_config.get('constraints', None)
        use_lhs = sampling_config.get('method', 'random').lower() == 'lhs'
        save_interval = sampling_config.get('save_interval', 1000)

        if 'total_samples' in sampling_config:
            args.num_samples = sampling_config['total_samples']

        filter_config = sampling_config.get('filter', {})
        filter_enabled = filter_config.get('enabled', False)
        rail_margin = filter_config.get('rail_margin', 0.1)
        exclude_nodes = set(n.lower() for n in filter_config.get('exclude_nodes', []))
        saturation_filter = filter_config.get('saturation_only', False)
        max_cutoff_devices = filter_config.get('max_cutoff_devices', None)
        require_ac_convergence = filter_config.get('require_ac_convergence', False)
        min_ugbw_hz = filter_config.get('min_ugbw_hz', None)

        prebatch_config = config.get('pre_batching', {})
        prebatch_enabled = prebatch_config.get('enabled', False)
        prebatch_batch_size = prebatch_config.get('batch_size', 1024)
        prebatch_num_variants = prebatch_config.get('num_batching_variants', 10)
        prebatch_seed = prebatch_config.get('seed', 42)

        split_enabled = prebatch_enabled
        split_train = prebatch_config.get('train', 0.8)
        split_val = prebatch_config.get('val', 0.1)
        split_test = prebatch_config.get('test', 0.1)
        split_seed = prebatch_config.get('split_seed', 42)

        for name, spec in config.get('parameters', {}).items():
            if isinstance(spec, dict):
                param_specs[name] = spec
            else:
                param_specs[name] = {'value': spec}

        print(f"Loaded config from {args.config}")
        print(f"  Num samples: {args.num_samples}")
        print(f"  Template: {args.template}")
        print(f"  AC template: {ac_template or 'none (DC only)'}")
        print(f"  Sampling method: {'LHS' if use_lhs else 'Random'}")
        print(f"  Normalize props (W, L): {normalize_props}")
        print(f"  Filter enabled: {filter_enabled} (rail_margin={rail_margin}V, exclude: {exclude_nodes})")
        print(f"  Saturation filter: {saturation_filter}")
        print(f"  Max cutoff devices: {max_cutoff_devices if max_cutoff_devices is not None else 'disabled'}")
        print(f"  Require AC convergence: {require_ac_convergence}")
        print(f"  Min UGBW: {min_ugbw_hz if min_ugbw_hz is not None else 'disabled'}")
        print(f"  Save interval: {save_interval}")
        if prebatch_enabled:
            print(f"  Pre-batching: enabled (batch_size={prebatch_batch_size}, variants={prebatch_num_variants})")
            if split_enabled:
                print(f"  Split: train={split_train:.0%}, val={split_val:.0%}, test={split_test:.0%}")
        else:
            print(f"  Pre-batching: disabled")
        print(f"  Parameters: {len(param_specs)}")
    else:
        ac_template = None
        normalize_props = False
        device_to_group = None
        excluded_current_devices = None
        param_constraints = None
        prebatch_enabled = False
        prebatch_batch_size = 1024
        prebatch_num_variants = 10
        prebatch_seed = 42
        split_enabled = False
        split_train = 0.8
        split_val = 0.1
        split_test = 0.1
        split_seed = 42
        param_specs = {
            'VDD': {'value': 1.8},
            'VCM': {'min': 0.7, 'max': 1.1, 'scale': 'linear'},
            'VDIFF': {'min': -0.1, 'max': 0.1, 'scale': 'linear'},
            'I_REF': {'min': 5e-6, 'max': 50e-6, 'scale': 'log'},
            'W_M1': {'min': 1e-6, 'max': 20e-6, 'scale': 'log'},
            'W_M2': {'min': 1e-6, 'max': 20e-6, 'scale': 'log'},
            'W_M3': {'min': 2e-6, 'max': 40e-6, 'scale': 'log'},
            'W_M4': {'min': 2e-6, 'max': 40e-6, 'scale': 'log'},
            'W_M5': {'min': 1e-6, 'max': 20e-6, 'scale': 'log'},
            'W_M6': {'min': 5e-6, 'max': 100e-6, 'scale': 'log'},
            'W_M7': {'min': 1e-6, 'max': 20e-6, 'scale': 'log'},
            'W_M8': {'min': 1e-6, 'max': 20e-6, 'scale': 'log'},
            'L_M1': {'min': 180e-9, 'max': 1e-6, 'scale': 'log'},
            'L_M2': {'min': 180e-9, 'max': 1e-6, 'scale': 'log'},
            'L_M3': {'min': 180e-9, 'max': 1e-6, 'scale': 'log'},
            'L_M4': {'min': 180e-9, 'max': 1e-6, 'scale': 'log'},
            'L_M5': {'min': 180e-9, 'max': 1e-6, 'scale': 'log'},
            'L_M6': {'min': 180e-9, 'max': 1e-6, 'scale': 'log'},
            'L_M7': {'min': 180e-9, 'max': 1e-6, 'scale': 'log'},
            'L_M8': {'min': 180e-9, 'max': 1e-6, 'scale': 'log'},
            'C_C': {'min': 0.5e-12, 'max': 5e-12, 'scale': 'log'},
            'R_Z': {'min': 100, 'max': 10000, 'scale': 'log'},
            'R_IN': {'min': 1e3, 'max': 100e3, 'scale': 'log'},
            'R_F': {'min': 1e3, 'max': 100e3, 'scale': 'log'},
        }

    if args.verify_only:
        num_samples = 5
        print("=== VERIFICATION MODE: Generating 5 samples ===")
    else:
        num_samples = args.num_samples

    # Generate parameter samples
    multiplier = sampling_config.get('max_attempts_multiplier', 3.0)
    max_attempts = int(num_samples * multiplier)
    n_workers = sampling_config.get('n_workers', None)
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    if use_lhs:
        print(f"Generating {max_attempts} LHS parameter samples (multiplier={multiplier})...")
        all_param_samples = generate_lhs_samples(param_specs, max_attempts)
    else:
        print(f"Will generate {max_attempts} random parameter samples...")
        all_param_samples = [generate_random_params(param_specs) for _ in range(max_attempts)]

    vdd = param_specs.get('VDD', {}).get('value', 1.8)

    # Parallel generation
    if args.verify_only or n_workers <= 1:
        print(f"Generating up to {num_samples} samples (single-threaded)...")
        samples = []
        failed = 0
        filtered = 0

        for param_idx, params in enumerate(tqdm(all_param_samples, total=len(all_param_samples))):
            if len(samples) >= num_samples:
                break

            result = worker_generate_sample((
                param_idx, params, args.template, str(output_dir),
                vdd, rail_margin, filter_enabled, exclude_nodes,
                normalize_props, param_specs, saturation_filter, ac_template,
                device_to_group, excluded_current_devices, param_constraints,
                max_cutoff_devices, require_ac_convergence, min_ugbw_hz
            ))

            sample_idx, sample_data, status = result
            if status == 'ok' and sample_data:
                samples.append(sample_data)
            elif status in ('filtered', 'filtered_saturation', 'filtered_constraint'):
                filtered += 1
            else:
                failed += 1
    else:
        print(f"Generating up to {num_samples} samples using {n_workers} workers...")

        samples = []
        failed = 0
        filtered = 0
        filtered_saturation = 0
        filtered_constraint = 0

        batch_size = n_workers * 6
        temp_sample_files = []
        worker_restart_interval = save_interval
        samples_since_restart = 0
        param_idx = 0

        executor = ProcessPoolExecutor(max_workers=n_workers)

        with tqdm(total=num_samples, desc="Valid samples") as pbar:
            while len(temp_sample_files) * save_interval + len(samples) < num_samples and param_idx < len(all_param_samples):
                batch_end = min(param_idx + batch_size, len(all_param_samples))
                batch_params = all_param_samples[param_idx:batch_end]

                batch_items = [
                    (param_idx + i, params, args.template, str(output_dir), vdd, rail_margin,
                     filter_enabled, exclude_nodes, normalize_props, param_specs, saturation_filter,
                     ac_template, device_to_group, excluded_current_devices, param_constraints,
                     max_cutoff_devices, require_ac_convergence, min_ugbw_hz)
                    for i, params in enumerate(batch_params)
                ]

                futures = {executor.submit(worker_generate_sample, item): item[0] for item in batch_items}
                completed_futures = []

                for future in as_completed(futures):
                    total_collected = len(temp_sample_files) * save_interval + len(samples)
                    if total_collected >= num_samples:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

                    try:
                        sample_idx, sample_data, status = future.result(timeout=60)
                        samples_since_restart += 1

                        if status == 'ok' and sample_data:
                            samples.append(sample_data)
                            pbar.update(1)

                            if len(samples) >= save_interval:
                                temp_file = output_dir / f'temp_samples_{len(temp_sample_files)}.pkl'
                                pbar.write(f"[Saving {len(samples)} samples to disk...]")
                                with open(temp_file, 'wb') as f:
                                    pickle.dump(samples, f)
                                temp_sample_files.append(temp_file)
                                del samples
                                samples = []
                                gc.collect()

                        elif status == 'filtered':
                            filtered += 1
                        elif status == 'filtered_saturation':
                            filtered_saturation += 1
                        elif status == 'filtered_constraint':
                            filtered_constraint += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1

                    completed_futures.append(future)
                    total_filtered = filtered + filtered_saturation + filtered_constraint
                    pbar.set_postfix({
                        'yield': f'{100*(len(temp_sample_files)*save_interval+len(samples))/(len(temp_sample_files)*save_interval+len(samples)+failed+total_filtered):.1f}%' if (len(samples)+failed+total_filtered) > 0 else '0%',
                        'failed': failed,
                        'rail': filtered,
                        'sat': filtered_saturation,
                        'constr': filtered_constraint,
                    })

                for f in completed_futures:
                    del futures[f]
                del completed_futures
                del futures

                # Restart workers periodically to prevent ngspice memory leaks
                if samples_since_restart >= worker_restart_interval:
                    pbar.write(f"[Restarting workers after {samples_since_restart} processed samples...]")
                    executor.shutdown(wait=True)
                    executor = ProcessPoolExecutor(max_workers=n_workers)
                    samples_since_restart = 0
                    gc.collect()

                param_idx = batch_end

        executor.shutdown(wait=True)

        if samples:
            temp_file = output_dir / f'temp_samples_{len(temp_sample_files)}.pkl'
            with open(temp_file, 'wb') as f:
                pickle.dump(samples, f)
            temp_sample_files.append(temp_file)
            del samples
            samples = []
            gc.collect()

        print(f"\n=== LOADING SAMPLES FOR NORMALIZATION ===")
        all_samples = []
        for temp_file in temp_sample_files:
            with open(temp_file, 'rb') as f:
                batch_samples = pickle.load(f)
                all_samples.extend(batch_samples)
            temp_file.unlink()
        samples = all_samples
        gc.collect()

    print(f"\nGenerated {len(samples)} samples (failed={failed}, filtered={filtered})")

    if args.verify_only and samples:
        print("\n=== VERIFYING GRAPH FORMAT ===")
        g = samples[0]['graph']

        print(f"\nGraph structure:")
        print(f"  x shape: {g.x.shape}")
        print(f"  type_tens shape: {g.type_tens.shape}")
        print(f"  edge_index shape: {g.edge_index.shape}")
        print(f"  num_terminals: {g.num_terminals}")
        print(f"  num_nets: {g.num_nets}")
        print(f"  output_node_mask sum: {g.output_node_mask.sum().item()}")
        print(f"  known_voltage_mask sum: {g.known_voltage_mask.sum().item()}")

        print(f"\nNode types (first 10):")
        for i, ntype in enumerate(g.node_types[:10]):
            x_feat = g.x[i].tolist()
            known = g.known_voltage_mask[i].item()
            output = g.output_node_mask[i].item()
            v = g.node_voltage_targets[i].item()
            print(f"  Node {i:2d}: {str(ntype):20s} | x={x_feat} | known={known} | output={output} | V={v:.4f}")

        print(f"\nNet nodes:")
        for i in range(g.num_terminals, len(g.node_types)):
            ntype = g.node_types[i]
            x_feat = g.x[i].tolist()
            known = g.known_voltage_mask[i].item()
            output = g.output_node_mask[i].item()
            v = g.node_voltage_targets[i].item()
            print(f"  Node {i:2d}: {str(ntype):20s} | x={x_feat} | known={known} | output={output} | V={v:.4f}")

        return

    # Save dataset
    if samples:
        if split_enabled:
            print(f"\n=== SPLITTING DATASET ===")
            random.seed(split_seed)
            np.random.seed(split_seed)

            indices = list(range(len(samples)))
            random.shuffle(indices)

            n_total = len(samples)
            n_train = int(n_total * split_train)
            n_val = int(n_total * split_val)

            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train+n_val]
            test_indices = indices[n_train+n_val:]

            train_samples = [samples[i] for i in train_indices]
            val_samples = [samples[i] for i in val_indices]
            test_samples = [samples[i] for i in test_indices]

            print(f"  Train: {len(train_samples)} samples ({len(train_samples)/n_total:.1%})")
            print(f"  Val:   {len(val_samples)} samples ({len(val_samples)/n_total:.1%})")
            print(f"  Test:  {len(test_samples)} samples ({len(test_samples)/n_total:.1%})")

            print("\n=== Feature normalization ===")
            if normalize_props:
                print("  - Base features (W, L): Min-max normalized [0, 1]")
            else:
                print("  - Base features (W, L): RAW")
            print("  - Global features: Domain-normalized")
            print("  - Targets (vdc, currents): RAW (training script handles normalization)")

            for split_name, split_data in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
                split_path = output_dir / f'dataset_{split_name}.pkl'
                with open(split_path, 'wb') as f:
                    pickle.dump(split_data, f)
                print(f"  Saved {split_path.name}")

            full_dataset_path = output_dir / 'dataset.pkl'
            with open(full_dataset_path, 'wb') as f:
                pickle.dump(samples, f)
            print(f"  Saved full dataset to {full_dataset_path.name}")

            if prebatch_enabled:
                print(f"\n=== CREATING PRE-BATCHED DATASET ({prebatch_num_variants} variants) ===")
                for split_name, split_data in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
                    if not split_data:
                        continue
                    print(f"\n  Pre-batching {split_name} split...")
                    split_output_dir = output_dir / split_name
                    split_output_dir.mkdir(exist_ok=True)
                    create_prebatched_dataset(
                        split_data, split_output_dir,
                        batch_size=prebatch_batch_size,
                        num_variants=prebatch_num_variants,
                        seed=prebatch_seed,
                        is_validation=(split_name != 'train')
                    )
        else:
            output_path = output_dir / 'dataset.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f)
            print(f"\nSaved {len(samples)} samples to {output_path}")

            if prebatch_enabled:
                print(f"\n=== CREATING PRE-BATCHED DATASET ({prebatch_num_variants} variants) ===")
                create_prebatched_dataset(
                    samples, output_dir,
                    batch_size=prebatch_batch_size,
                    num_variants=prebatch_num_variants,
                    seed=prebatch_seed,
                    is_validation=False
                )

        # Generate distribution plots
        print("\n=== GENERATING DISTRIBUTION PLOTS ===")
        plot_parameter_distributions(samples, param_specs, output_dir)
        plot_node_voltage_distributions(samples, output_dir)
        plot_device_current_distributions(samples, output_dir)
        plot_target_normalization_distributions(samples, output_dir)


if __name__ == '__main__':
    main()
