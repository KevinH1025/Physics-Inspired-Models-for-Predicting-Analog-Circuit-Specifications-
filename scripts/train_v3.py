#!/usr/bin/env python3
"""
Training script for GNN model.

Features:
- Voltage and current prediction
- Pre-batched dataset support
- GPU pre-loading for fast training
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set CUBLAS workspace config for deterministic algorithms (must be before torch import)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import argparse
from tqdm import tqdm
import numpy as np
import time
import random

from src.training.data_loading import (
    PrebatchedLoader,
    load_prebatched_variant,
    load_prebatched_metadata,
    compute_vdc_normalization,
    compute_current_normalization,
    normalize_batches_vdc,
    normalize_batches_current,
    attach_normalization_stats,
    add_ss_node_targets,
    add_vth_node_targets,
    add_mosfet_gt_vov,
    add_region_node_targets,
    compute_ss_normalization,
    normalize_batches_ss,
)
from src.training.loops import train_epoch, validate
from src.training.losses import compute_kcl_loss, compute_kcl_per_net_debug
from src.training.scheduler import create_scheduler, apply_warmup, step_scheduler
from src.training.plotting import plot_training_curves
from src.training.checkpoint import save_checkpoint, build_full_config, create_model_from_args
from src.training.config import load_config, parse_training_config


def main():
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, default='datasets/opamp_v3', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=700, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=15, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--jk-mode', type=str, default='cat', help='Jumping knowledge mode')
    parser.add_argument('--jk-attention', action='store_true', help='Use attention for JK')
    parser.add_argument('--num-mlp-layers', type=int, default=3, help='MLP layers in head')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'poly', 'none'])
    parser.add_argument('--plateau-patience', type=int, default=10, help='Epochs to wait before reducing LR (plateau scheduler)')
    parser.add_argument('--plateau-factor', type=float, default=0.5, help='Factor to multiply LR on plateau (e.g., 0.5 = halve)')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup epochs')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--virtual-node', action='store_true', help='Use virtual node')
    parser.add_argument('--model-type', type=str, default=None,
                        help='Model type from registry (e.g., deepgen, deepgen_vn). Overrides --virtual-node if set.')
    parser.add_argument('--predict-currents', action='store_true', help='Enable current prediction')
    parser.add_argument('--current-weight', type=float, default=1.0, help='Weight for current loss')
    parser.add_argument('--derive-currents-from-voltage', action='store_true',
                        help='Derive MOSFET currents from predicted voltages instead of independent prediction')
    parser.add_argument('--mosfet-current-mlp-hidden', type=int, default=64,
                        help='Hidden dimension for MOSFET current MLP')
    parser.add_argument('--mosfet-current-mlp-layers', type=int, default=2,
                        help='Number of layers in MOSFET current MLP')
    parser.add_argument('--phase2', action='store_true',
                        help='Phase 2 training: freeze backbone/voltage, train current MLP only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for Phase 2 or fine-tuning')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune from checkpoint on new dataset (cross-topology transfer)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze GNN backbone during fine-tuning (train heads only)')
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1,
                        help='LR scale factor for backbone params when not frozen (default: 0.1 = 10x lower)')
    parser.add_argument('--preload-to-gpu', action='store_true',
                        help='Pre-load all batches to GPU (faster training, uses more VRAM)')
    parser.add_argument('--ss-loss-weight', type=float, default=0.0, dest='ss_loss_weight',
                        help='Weight for supervised gm/gds loss (0 = disabled)')
    parser.add_argument('--ac-loss-weight', type=float, default=0.0, dest='ac_loss_weight',
                        help='Weight for AC loss (UGBW, PM, AM) (0 = disabled)')
    parser.add_argument('--kcl-weight', type=float, default=0.0,
                        help='Weight for KCL physics loss (0 = disabled)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: disable deterministic algorithms, enable cudnn.benchmark')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for model optimization (PyTorch 2.0+)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (saves outputs to experiments/<name>/)')
    args = parser.parse_args()

    # Track which CLI args were explicitly provided (for overriding config)
    cli_explicit = {action.dest for action in parser._actions
                    if action.dest in vars(args) and
                    any(opt in sys.argv for opt in action.option_strings)}

    # Convert model-type to model_type for consistency
    args.model_type = getattr(args, 'model_type', None)

    # Load config from YAML if provided
    if args.config:
        config = load_config(args.config)
        parsed = parse_training_config(config)

        for key, value in parsed.items():
            if value is not None:
                setattr(args, key, value)

        # CLI-explicit args override config values
        cli_args = parser.parse_args()
        for key in cli_explicit:
            setattr(args, key, getattr(cli_args, key))

        print(f"Loaded config from {args.config}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Performance vs reproducibility tradeoff
    if getattr(args, 'fast', False):
        # Fast mode: prioritize speed over exact reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        torch.set_float32_matmul_precision('high')  # Use TensorCores
        print("Fast mode: cudnn.benchmark=True, deterministic=False")
    else:
        # Reproducible mode: exact results but slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    # Load dataset
    dataset_path = Path(args.dataset)
    use_fixed_topology = getattr(args, 'use_fixed_topology', False)
    use_prebatched = getattr(args, 'use_prebatched', False)
    # Auto-detect prebatched if train/ directory has variant files (skip if fixed_topology)
    if not use_fixed_topology and not use_prebatched and (dataset_path / 'train' / 'variant_0.pkl').exists():
        use_prebatched = True
        print("Auto-detected prebatched dataset")

    # Output directory: experiments/<name>/ if --name provided, else dataset root
    if args.name:
        output_path = dataset_path / 'experiments' / args.name
    else:
        output_path = dataset_path

    # Tee stdout to a log file in the output directory (overwritten each run)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = open(output_path / 'training.log', 'w')
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"\n=== LOADING DATASET ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Mode: {'prebatched' if use_prebatched else 'dynamic batching'}")

    target_norm_type = getattr(args, 'target_norm_type', 'zscore')
    vdd = getattr(args, 'vdd', 1.8)
    current_mean, current_std = 0.0, 1.0
    ss_gm_mean, ss_gm_std = 0.0, 1.0
    ss_gds_mean, ss_gds_std = 0.0, 1.0
    has_ss = False
    variant_config = {'enabled': False}

    if use_prebatched:
        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'val'

        if not train_dir.exists():
            raise FileNotFoundError(f"Pre-batched train directory not found: {train_dir}")

        train_metadata = load_prebatched_metadata(train_dir)
        num_train_variants = train_metadata.get('num_variants', 1)
        preload_device = args.device if getattr(args, 'preload_to_gpu', False) and args.device != 'cpu' else None
        num_to_preload = min(getattr(args, 'max_preload_variants', 1), num_train_variants)

        all_train_variants = []
        print(f"Pre-loading {num_to_preload}/{num_train_variants} train variants...")
        for vid in range(num_to_preload):
            variant_batches = load_prebatched_variant(train_dir, variant_id=vid, device=preload_device)
            all_train_variants.append(variant_batches)
            print(f"  Loaded variant {vid}: {len(variant_batches)} batches")

        train_batches = all_train_variants[0]
        val_batches = load_prebatched_variant(val_dir, variant_id=0, device=preload_device) if val_dir.exists() else None
        if val_batches:
            print(f"Loaded {len(val_batches)} val batches")

        vdc_mean, vdc_std = compute_vdc_normalization(dataset_path, train_batches, target_norm_type, vdd)
        print(f"vdc normalization: mean={vdc_mean:.4f}, std={vdc_std:.4f}")

        for variant_batches in all_train_variants:
            normalize_batches_vdc(variant_batches, vdc_mean, vdc_std)
        if val_batches:
            normalize_batches_vdc(val_batches, vdc_mean, vdc_std)

        # Pre-compute GT Vov per MOSFET (must be before current normalization)
        for variant_batches in all_train_variants:
            add_mosfet_gt_vov(variant_batches)
        if val_batches:
            add_mosfet_gt_vov(val_batches)

        has_currents = any(hasattr(b, 'node_current_targets') and b.node_current_targets is not None for b in train_batches[:3])
        if has_currents:
            current_mean, current_std = compute_current_normalization(all_train_variants)
            current_z_clip = getattr(args, 'current_z_clip', 0.0)
            print(f"Current normalization: log10 mean={current_mean:.2f}, std={current_std:.2f}")
            if current_z_clip > 0:
                print(f"  Soft z-clip enabled at ±{current_z_clip}σ")
            for variant_batches in all_train_variants:
                normalize_batches_current(variant_batches, current_mean, current_std, z_clip=current_z_clip)
            if val_batches:
                normalize_batches_current(val_batches, current_mean, current_std, z_clip=current_z_clip)

        # Attach normalization stats to batches for physics-based current prediction
        for variant_batches in all_train_variants:
            attach_normalization_stats(variant_batches, vdc_mean, vdc_std, current_mean, current_std)
        if val_batches:
            attach_normalization_stats(val_batches, vdc_mean, vdc_std, current_mean, current_std)

        # Add per-node SS targets for mask-based gm/gds prediction
        for variant_batches in all_train_variants:
            add_ss_node_targets(variant_batches)
        if val_batches:
            add_ss_node_targets(val_batches)

        # Add per-node Vth targets for gm physics loss
        for variant_batches in all_train_variants:
            add_vth_node_targets(variant_batches)
        if val_batches:
            add_vth_node_targets(val_batches)

        # Add per-node region labels for region classification head
        for variant_batches in all_train_variants:
            add_region_node_targets(variant_batches)
        if val_batches:
            add_region_node_targets(val_batches)

        # Normalize SS targets (log10 gm/gds) with z-score
        has_ss = any(hasattr(b, 'mosfet_drain_mask') and b.mosfet_drain_mask.any() for b in train_batches[:3])
        if has_ss:
            ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std = compute_ss_normalization(all_train_variants)
            print(f"SS normalization: gm mean={ss_gm_mean:.2f}, std={ss_gm_std:.2f} | gds mean={ss_gds_mean:.2f}, std={ss_gds_std:.2f}")
            for variant_batches in all_train_variants:
                normalize_batches_ss(variant_batches, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)
            if val_batches:
                normalize_batches_ss(val_batches, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)

        train_loader = PrebatchedLoader(train_batches, shuffle=True)
        val_loader = PrebatchedLoader(val_batches, shuffle=False) if val_batches else None
        variant_config = {'enabled': num_to_preload > 1, 'num_variants': num_to_preload, 'all_train_variants': all_train_variants}
        sample_batch = train_batches[0]

    elif use_fixed_topology:
        from src.data.fixed_topology_loader import build_fixed_topology_dataset, FixedTopologyLoader

        print(f"\n=== LOADING FIXED-TOPOLOGY DATASET ===")
        gpu_device = args.device if args.device != 'cpu' else None
        load_device = args.device  # load directly onto training device

        train_ds = build_fixed_topology_dataset(dataset_path / 'dataset_train.pkl', device=load_device)
        val_ds = build_fixed_topology_dataset(dataset_path / 'dataset_val.pkl', device=load_device)

        # VDC normalization (compute from raw stacked tensor)
        raw_vdc = train_ds.all_vdc.flatten()
        if target_norm_type == 'minmax':
            vdc_mean, vdc_std = 0.0, vdd
        else:
            vdc_mean = raw_vdc.mean().item()
            vdc_std = raw_vdc.std().item()
            if vdc_std == 0:
                vdc_std = 1.0
        print(f"vdc normalization: mean={vdc_mean:.4f}, std={vdc_std:.4f}")
        train_ds.normalize_vdc(vdc_mean, vdc_std)
        val_ds.normalize_vdc(vdc_mean, vdc_std)

        # Current normalization (use per-sample masks — has_current_mask varies across samples)
        log_eps = 1e-12
        masked_values = train_ds.all_currents[train_ds.all_has_current_mask].abs()
        log_c = torch.log10(masked_values + log_eps)
        current_mean = log_c.mean().item()
        current_std = log_c.std().item()
        if current_std == 0:
            current_std = 1.0
        print(f"Current normalization: log10 mean={current_mean:.2f}, std={current_std:.2f}")
        train_ds.normalize_currents(current_mean, current_std)
        val_ds.normalize_currents(current_mean, current_std)

        # SS normalization (log10 gm/gds z-score)
        has_ss = train_ds.all_node_log_gm is not None
        if has_ss:
            drain_mask = train_ds.mosfet_drain_mask  # [N]
            gm_vals = train_ds.all_node_log_gm[:, drain_mask].flatten()
            gds_vals = train_ds.all_node_log_gds[:, drain_mask].flatten()
            # Filter out -inf/nan from log10(0)
            gm_valid = gm_vals[torch.isfinite(gm_vals)]
            gds_valid = gds_vals[torch.isfinite(gds_vals)]
            ss_gm_mean, ss_gm_std = gm_valid.mean().item(), gm_valid.std().item()
            ss_gds_mean, ss_gds_std = gds_valid.mean().item(), gds_valid.std().item()
            if ss_gm_std == 0: ss_gm_std = 1.0
            if ss_gds_std == 0: ss_gds_std = 1.0
            print(f"SS normalization: gm mean={ss_gm_mean:.2f}, std={ss_gm_std:.2f} | gds mean={ss_gds_mean:.2f}, std={ss_gds_std:.2f}")
            train_ds.normalize_ss(ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)
            val_ds.normalize_ss(ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)

        ft_batch_size = getattr(args, 'batch_size', 1024)
        train_loader = FixedTopologyLoader(train_ds, batch_size=ft_batch_size, shuffle=True)
        val_loader = FixedTopologyLoader(val_ds, batch_size=ft_batch_size, shuffle=False)
        variant_config = {'enabled': False}
        sample_batch = train_ds.get_batch(list(range(min(4, len(train_ds)))))

    else:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from src.training.data_loading import CircuitGraphDataset

        train_file = dataset_path / 'dataset_train.pkl'
        val_file = dataset_path / 'dataset_val.pkl'

        if not train_file.exists():
            raise FileNotFoundError(f"Training dataset not found: {train_file}")

        train_dataset = CircuitGraphDataset(train_file)
        val_dataset = CircuitGraphDataset(val_file) if val_file.exists() else None
        print(f"Loaded {len(train_dataset)} train samples")
        if val_dataset:
            print(f"Loaded {len(val_dataset)} val samples")

        # Compute normalization from training graphs
        all_vdc = torch.cat([g.vdc for g in train_dataset.graphs])
        if target_norm_type == 'minmax':
            vdc_mean, vdc_std = 0.0, vdd
        else:
            vdc_mean, vdc_std = all_vdc.mean().item(), all_vdc.std().item()
            if vdc_std == 0:
                vdc_std = 1.0
        print(f"vdc normalization: mean={vdc_mean:.4f}, std={vdc_std:.4f}")

        # Normalize graphs in-place
        for g in train_dataset.graphs:
            g.vdc = (g.vdc - vdc_mean) / vdc_std
        if val_dataset:
            for g in val_dataset.graphs:
                g.vdc = (g.vdc - vdc_mean) / vdc_std

        # Current normalization
        has_currents = any(hasattr(g, 'node_current_targets') and g.node_current_targets is not None for g in train_dataset.graphs[:3])
        if has_currents:
            log_epsilon = 1e-12
            all_currents = []
            for g in train_dataset.graphs:
                if hasattr(g, 'node_current_targets') and hasattr(g, 'has_current_mask'):
                    if g.node_current_targets is not None and g.has_current_mask is not None:
                        mask = g.has_current_mask
                        if mask.any():
                            all_currents.extend(g.node_current_targets[mask].abs().cpu().tolist())
            if all_currents:
                log_currents = np.log10(np.array(all_currents) + log_epsilon)
                current_mean, current_std = float(log_currents.mean()), float(log_currents.std())
                if current_std == 0:
                    current_std = 1.0
                print(f"Current normalization: log10 mean={current_mean:.2f}, std={current_std:.2f}")

                for g in train_dataset.graphs:
                    if hasattr(g, 'node_current_targets') and g.node_current_targets is not None:
                        log_targets = torch.log10(g.node_current_targets.abs() + log_epsilon)
                        g.node_current_targets = (log_targets - current_mean) / current_std
                if val_dataset:
                    for g in val_dataset.graphs:
                        if hasattr(g, 'node_current_targets') and g.node_current_targets is not None:
                            log_targets = torch.log10(g.node_current_targets.abs() + log_epsilon)
                            g.node_current_targets = (log_targets - current_mean) / current_std

        # Attach normalization stats to graphs for physics-based current prediction
        for g in train_dataset.graphs:
            g.voltage_mean = vdc_mean
            g.voltage_std = vdc_std
            g.current_mean = current_mean
            g.current_std = current_std
        if val_dataset:
            for g in val_dataset.graphs:
                g.voltage_mean = vdc_mean
                g.voltage_std = vdc_std
                g.current_mean = current_mean
                g.current_std = current_std

        batch_size = getattr(args, 'batch_size', 128)
        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        print(f"Using batch_size={batch_size}")
        sample_batch = next(iter(train_loader))

    # Feature dimensions
    x_dim = sample_batch.x.shape[1]
    type_dim = sample_batch.type_tens.shape[1]
    net_type_dim = sample_batch.net_type.shape[1] if hasattr(sample_batch, 'net_type') and sample_batch.net_type is not None else 0
    total_input_dim = x_dim + type_dim + net_type_dim
    print(f"Input features: {total_input_dim} (x={x_dim}, type={type_dim}, net={net_type_dim})")

    # Fine-tune: override architecture args from checkpoint config so model matches exactly
    if getattr(args, 'finetune', False):
        if not args.checkpoint:
            raise ValueError("--finetune requires --checkpoint path to pre-trained model")
        print(f"\n=== FINE-TUNING: Loading architecture from checkpoint ===")
        _ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        _src_cfg = _ckpt.get('config', {})
        # Architecture args to inherit from checkpoint
        _arch_map = {
            'hidden': ('hidden', 'hidden_dim'),
            'layers': ('layers', 'num_layers'),
            'dropout': ('dropout',),
            'genconv_num_layers': ('genconv_num_layers',),
            'num_mlp_layers': ('num_mlp_layers',),
            'jk_mode': ('jk_mode',),
            'jk_attention': ('jk_attention',),
            'jk_learn_temperature': ('jk_learn_temperature',),
            'norm_type': ('norm_type',),
            'skip_connection': ('skip_connection',),
            'virtual_node': ('virtual_node', 'use_virtual_node'),
            'use_attention_pooling': ('use_attention_pooling',),
            'vn_learn_temperature': ('vn_learn_temperature',),
        }
        # Dict configs to inherit as-is
        _dict_configs = [
            'voltage_head_config', 'current_head_config', 'ss_head_config',
            'ac_head_config', 'region_head_config', 'mosfet_current_mlp_config',
            'current_gnn_config', 'frozen_device_mlp_config', 'refinement_config',
        ]
        overridden = []
        for arg_name, cfg_keys in _arch_map.items():
            for ck in cfg_keys:
                if ck in _src_cfg:
                    old_val = getattr(args, arg_name, None)
                    setattr(args, arg_name, _src_cfg[ck])
                    if old_val != _src_cfg[ck]:
                        overridden.append(f"  {arg_name}: {old_val} -> {_src_cfg[ck]}")
                    break
        for dc in _dict_configs:
            if dc in _src_cfg:
                old_val = getattr(args, dc, {})
                setattr(args, dc, _src_cfg[dc])
                if old_val != _src_cfg[dc]:
                    overridden.append(f"  {dc}: {old_val} -> {_src_cfg[dc]}")
        # Bool configs
        for bc in ['predict_currents', 'derive_currents_from_voltage', 'use_gnn_current_prediction',
                    'use_frozen_device_mlp', 'use_refinement_pass', 'gradient_checkpointing']:
            if bc in _src_cfg:
                old_val = getattr(args, bc, False)
                setattr(args, bc, _src_cfg[bc])
                if old_val != _src_cfg[bc]:
                    overridden.append(f"  {bc}: {old_val} -> {_src_cfg[bc]}")
        # Disable gradient checkpointing when backbone is frozen (no backbone grads needed)
        if getattr(args, 'freeze_backbone', False) and getattr(args, 'gradient_checkpointing', False):
            args.gradient_checkpointing = False
            overridden.append(f"  gradient_checkpointing: True -> False (frozen backbone, not needed)")
        if overridden:
            print(f"Overrode {len(overridden)} args from checkpoint:")
            for o in overridden:
                print(o)
        else:
            print("All architecture args already match checkpoint.")
        del _ckpt  # free memory, will reload later for weights

    # Create model
    predict_currents = getattr(args, 'predict_currents', False)
    voltage_head_config = getattr(args, 'voltage_head_config', {})
    current_head_config = getattr(args, 'current_head_config', {})

    # Voltage-derived currents config
    derive_currents_from_voltage = getattr(args, 'derive_currents_from_voltage', False)
    if not hasattr(args, 'mosfet_current_mlp_config'):
        args.mosfet_current_mlp_config = {
            'hidden_dim': getattr(args, 'mosfet_current_mlp_hidden', 64),
            'num_layers': getattr(args, 'mosfet_current_mlp_layers', 2),
            'use_wl_ratio': True,
        }
    if derive_currents_from_voltage:
        print(f"Using voltage-derived currents (MLP: {args.mosfet_current_mlp_config})")

    model, model_class = create_model_from_args(args, total_input_dim, args.device)
    print(f"Using {model_class}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # torch.compile() for PyTorch 2.0+ optimization
    if getattr(args, 'compile', False):
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully")
        else:
            print("Warning: torch.compile() not available (requires PyTorch 2.0+)")

    # Phase 2: Load checkpoint and freeze backbone/voltage head
    if getattr(args, 'phase2', False):
        if not args.checkpoint:
            raise ValueError("--phase2 requires --checkpoint path to Phase 1 model")

        print(f"\n=== PHASE 2: Current MLP Fine-tuning ===")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

        # Load only backbone/voltage head weights (skip MLP to use fresh config)
        state_dict = ckpt['model_state_dict']
        mlp_keys = [k for k in state_dict.keys() if 'mosfet_current_mlp' in k]
        for k in mlp_keys:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded backbone from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
        print(f"Fresh MLP with config: {args.mosfet_current_mlp_config}")

        # Freeze everything except mosfet_current_mlp
        for name, param in model.named_parameters():
            if 'mosfet_current_mlp' not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Frozen: {total - trainable:,} params, Trainable: {trainable:,} params")

        # Set model to Phase 2 mode (don't detach voltages)
        model.phase2_mode = True

        # Ensure current training is enabled
        args.predict_currents = True
        if getattr(args, 'current_weight', 0.0) == 0.0:
            args.current_weight = 1.0
            print(f"Setting current_weight=1.0 for Phase 2")

    # Fine-tune: load checkpoint weights, optionally freeze backbone
    if getattr(args, 'finetune', False):
        print(f"\n=== FINE-TUNING: Loading weights ===")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        source_config = ckpt.get('config', {})

        # Filter out keys with shape mismatches (e.g., different head MLP depths)
        model_sd = model.state_dict()
        skipped = []
        for k in list(state_dict.keys()):
            if k in model_sd and state_dict[k].shape != model_sd[k].shape:
                skipped.append(f"{k}: ckpt {list(state_dict[k].shape)} vs model {list(model_sd[k].shape)}")
                del state_dict[k]
        if skipped:
            print(f"  Skipped {len(skipped)} shape-mismatched keys (will use random init):")
            for s in skipped:
                print(f"    {s}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (randomly initialized): {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        print(f"Loaded pre-trained weights from {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.4f})")

        # Backbone parameter names: GNN layers, JK, virtual node, input projection
        backbone_prefixes = ('layers.', 'jk_linear.', 'jk_attn.', 'input_proj.', 'input_linear.',
                             'vn_', 'virtual_node', 'norms.')

        if getattr(args, 'freeze_backbone', False):
            # Phase 1 of fine-tuning: freeze backbone, train heads only
            frozen_count = 0
            trainable_count = 0
            for name, param in model.named_parameters():
                if any(name.startswith(p) for p in backbone_prefixes):
                    param.requires_grad = False
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"Backbone FROZEN: {frozen_count:,} params")
            print(f"Heads trainable: {trainable_count:,} params")
        else:
            # Phase 2 of fine-tuning: all params trainable with discriminative LR
            total = sum(p.numel() for p in model.parameters())
            print(f"All {total:,} params trainable (backbone LR scale: {args.backbone_lr_scale}x)")

        # Store source checkpoint info for saving
        args._finetune_source = str(args.checkpoint)
        args._finetune_source_epoch = ckpt.get('epoch', -1)

    # Optimizer and scheduler (use only trainable params for Phase 2 / freeze-backbone)
    weight_decay = getattr(args, 'weight_decay', 0.0)
    use_fused = torch.cuda.is_available()

    if getattr(args, 'phase2', False) or (getattr(args, 'finetune', False) and getattr(args, 'freeze_backbone', False)):
        # Only optimize trainable (unfrozen) parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=weight_decay, fused=use_fused)
    elif getattr(args, 'finetune', False) and not getattr(args, 'freeze_backbone', False):
        # Discriminative LR: backbone gets lower LR, heads get full LR
        backbone_prefixes = ('layers.', 'jk_linear.', 'jk_attn.', 'input_proj.', 'input_linear.',
                             'vn_', 'virtual_node', 'norms.')
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in backbone_prefixes):
                backbone_params.append(param)
            else:
                head_params.append(param)
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale},
            {'params': head_params, 'lr': args.lr},
        ], weight_decay=weight_decay, fused=use_fused)
        print(f"Discriminative LR: backbone={args.lr * args.backbone_lr_scale:.2e}, heads={args.lr:.2e}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay, fused=use_fused)

    scheduler = create_scheduler(
        optimizer, args.scheduler, args.epochs, args.warmup,
        min_lr=getattr(args, 'end_lr', 1e-6),
        plateau_factor=getattr(args, 'plateau_factor', 0.5),
        plateau_patience=getattr(args, 'plateau_patience', 10)
    )

    # AMP - use bfloat16 for better precision (less overflow issues than float16)
    use_amp = args.device != 'cpu' and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else (torch.float16 if use_amp else None)
    scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None  # bfloat16 doesn't need scaler
    if use_amp:
        dtype_name = 'bfloat16' if amp_dtype == torch.bfloat16 else 'float16'
        print(f"Using AMP with {dtype_name}")

    # Training state
    best_val_loss, best_epoch = float('inf'), 0
    train_losses, val_losses, val_maes = [], [], []
    train_voltage_losses, val_voltage_losses = [], []
    train_current_losses, val_current_losses = [], []
    train_ss_losses, val_ss_losses = [], []
    train_kcl_losses, val_kcl_losses = [], []
    train_ac_losses, val_ac_losses = [], []
    train_region_losses, val_region_losses = [], []
    train_gm_physics_losses, val_gm_physics_losses = [], []
    train_triode_physics_losses, val_triode_physics_losses = [], []
    train_cutoff_physics_losses, val_cutoff_physics_losses = [], []
    train_maes, train_current_maes, val_current_maes, learning_rates = [], [], [], []
    epochs_without_improvement = 0
    best_model_state = None
    best_metrics = {'val_mae_mv': float('inf'), 'val_current_mae_ua': 0.0, 'acc80': 0.0, 'acc50': 0.0, 'acc20': 0.0, 'acc10': 0.0, 'current_acc50': 0.0, 'current_acc20': 0.0, 'current_acc5': 0.0, 'current_acc2': 0.0}

    early_stopping_patience = getattr(args, 'early_stopping_patience', 80)
    val_freq = getattr(args, 'val_freq', 1)
    loss_type = getattr(args, 'loss_type', 'mse')
    huber_delta = getattr(args, 'huber_delta', 1.0)
    current_weight_target = getattr(args, 'current_weight', 1.0)
    current_warmup_epochs = getattr(args, 'current_warmup_epochs', 0)
    kcl_weight_target = getattr(args, 'kcl_weight', 0.0)
    kcl_warmup_epochs = getattr(args, 'kcl_warmup_epochs', 0)
    kcl_start_epoch_cfg = getattr(args, 'kcl_start_epoch', 0)
    kcl_min_current = getattr(args, 'kcl_min_current', 1e-9)
    kcl_mode = getattr(args, 'kcl_mode', 'logsumexp')
    kcl_exclusive = getattr(args, 'kcl_exclusive', False)
    kcl_detach_backbone = getattr(args, 'kcl_detach_backbone', False)
    kcl_violation_threshold = getattr(args, 'kcl_violation_threshold', 0.0)
    kcl_huber_delta = getattr(args, 'kcl_huber_delta', 0.0)
    kcl_conservation = getattr(args, 'kcl_conservation', False)
    kcl_skip_two_term = getattr(args, 'kcl_skip_two_term', False)
    kcl_only_two_term = getattr(args, 'kcl_only_two_term', False)
    # Physics constraint config (diff pair, current mirrors)
    constraint_weight_target = getattr(args, 'constraint_weight', 0.0)
    constraint_warmup_epochs = getattr(args, 'constraint_warmup_epochs', 0)
    constraint_start_epoch_cfg = getattr(args, 'constraint_start_epoch', 0)

    # gm self-consistency physics loss config
    gm_physics_loss_weight_target = getattr(args, 'gm_physics_loss_weight', 0.0)
    gm_physics_loss_warmup_epochs = getattr(args, 'gm_physics_loss_warmup_epochs', 0)
    gm_physics_loss_start_epoch_cfg = getattr(args, 'gm_physics_loss_start_epoch', 0)
    gm_physics_min_vov = getattr(args, 'gm_physics_min_vov', 0.0)
    gm_physics_use_clm = getattr(args, 'gm_physics_use_clm', False)
    gm_physics_use_gt_voltages = getattr(args, 'gm_physics_use_gt_voltages', True)

    # AC prediction loss config
    ac_loss_weight_target = getattr(args, 'ac_loss_weight', 0.0)
    ac_loss_warmup_epochs = getattr(args, 'ac_loss_warmup_epochs', 0)
    ac_loss_start_epoch_cfg = getattr(args, 'ac_loss_start_epoch', 0)

    # Supervised gm/gds loss config
    ss_loss_weight_target = getattr(args, 'ss_loss_weight', 0.0)
    ss_loss_warmup_epochs = getattr(args, 'ss_loss_warmup_epochs', 0)
    ss_loss_start_epoch_cfg = getattr(args, 'ss_loss_start_epoch', 0)

    # Triode physics regularizer loss config
    triode_physics_loss_weight_target = getattr(args, 'triode_physics_loss_weight', 0.0)
    triode_physics_loss_warmup_epochs = getattr(args, 'triode_physics_loss_warmup_epochs', 0)
    triode_physics_loss_start_epoch_cfg = getattr(args, 'triode_physics_loss_start_epoch', 0)
    triode_physics_config = getattr(args, 'triode_physics_config', None)

    # Cutoff/subthreshold physics loss config
    cutoff_physics_loss_weight_target = getattr(args, 'cutoff_physics_loss_weight', 0.0)
    cutoff_physics_loss_warmup_epochs = getattr(args, 'cutoff_physics_loss_warmup_epochs', 0)
    cutoff_physics_loss_start_epoch_cfg = getattr(args, 'cutoff_physics_loss_start_epoch', 0)
    cutoff_physics_n_nmos = getattr(args, 'cutoff_physics_n_nmos', 1.5)
    cutoff_physics_n_pmos = getattr(args, 'cutoff_physics_n_pmos', 2.0)

    # Region classification loss config
    region_loss_weight_target = getattr(args, 'region_loss_weight', 0.0)
    region_loss_start_epoch_cfg = getattr(args, 'region_loss_start_epoch', 0)

    # Build AC components list from config
    ac_head_cfg = getattr(args, 'ac_head_config', {})
    ac_components = []
    if ac_head_cfg.get('predict_ugbw', True):
        ac_components.append('ugbw')
    if ac_head_cfg.get('predict_pm', True):
        ac_components.append('pm')
    if ac_head_cfg.get('predict_am', False):
        ac_components.append('am')

    # Compute AC normalization stats from training data (if AC loss enabled)
    ac_mean = None
    ac_std = None
    if ac_loss_weight_target > 0 and ac_components:
        # Collect raw values for each enabled component
        ac_raw = {comp: [] for comp in ac_components}
        if use_fixed_topology:
            # For FixedTopologyLoader, scan the dataset tensors directly
            if train_ds.all_ac_valid is not None and train_ds.all_ac_ugbw is not None:
                valid = train_ds.all_ac_valid.bool()
                if valid.any():
                    if 'ugbw' in ac_raw:
                        ac_raw['ugbw'].extend(torch.log10(train_ds.all_ac_ugbw[valid].clamp(min=1.0)).cpu().tolist())
                    if 'pm' in ac_raw:
                        ac_raw['pm'].extend(train_ds.all_ac_pm[valid].cpu().tolist())
                    if 'am' in ac_raw and train_ds.all_ac_am is not None:
                        ac_raw['am'].extend(train_ds.all_ac_am[valid].cpu().tolist())
        else:
            scan_batches = train_batches if use_prebatched else []
            for b in scan_batches:
                if hasattr(b, 'ac_valid') and hasattr(b, 'ac_ugbw'):
                    valid = b.ac_valid
                    if valid.any():
                        if 'ugbw' in ac_raw:
                            ac_raw['ugbw'].extend(torch.log10(b.ac_ugbw[valid].clamp(min=1.0)).tolist())
                        if 'pm' in ac_raw:
                            ac_raw['pm'].extend(b.ac_pm[valid].tolist())
                        if 'am' in ac_raw:
                            ac_raw['am'].extend(b.ac_am[valid].tolist())
        n_valid = len(ac_raw[ac_components[0]])
        if n_valid > 0:
            ac_mean = torch.tensor([np.mean(ac_raw[c]) for c in ac_components], dtype=torch.float32)
            ac_std = torch.tensor([np.std(ac_raw[c]) for c in ac_components], dtype=torch.float32).clamp(min=1e-6)
            print(f"AC components: {ac_components}")
            print(f"AC normalization: mean={ac_mean.tolist()}, std={ac_std.tolist()}")
            print(f"  ({n_valid} valid AC samples in training set)")
        else:
            print("WARNING: No valid AC samples found. AC loss will be disabled.")
            ac_loss_weight_target = 0.0

    # Stage 2 node weighting config
    stage2_weight = getattr(args, 'stage2_weight', 1.0)
    stage2_nodes = getattr(args, 'stage2_nodes', None)

    # Terminal voltage supervision
    use_terminal_voltage_loss = getattr(args, 'use_terminal_voltage_loss', False)
    if use_terminal_voltage_loss:
        print("Terminal voltage supervision enabled (loss on terminal nodes)")

    training_start_time = time.time()
    print(f"\nTraining for {args.epochs} epochs (early stopping: patience={early_stopping_patience})...")

    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        # Variant rotation
        if variant_config['enabled'] and epoch > 0:
            variant_id = epoch % variant_config['num_variants']
            train_batches = variant_config['all_train_variants'][variant_id]
            train_loader = PrebatchedLoader(train_batches, shuffle=True)

        apply_warmup(optimizer, epoch, args.warmup, args.lr)

        # Apply current loss warmup - linearly ramp from 0 to target over warmup epochs
        if current_warmup_epochs > 0 and epoch < current_warmup_epochs:
            current_weight = current_weight_target * (epoch + 1) / current_warmup_epochs
        else:
            current_weight = current_weight_target

        # Apply KCL loss warmup - starts at kcl_start_epoch (or after current warmup if 0)
        # This ensures predictions are reasonable before enforcing physics constraint
        kcl_start_epoch = kcl_start_epoch_cfg if kcl_start_epoch_cfg > 0 else current_warmup_epochs
        if kcl_warmup_epochs > 0 and epoch >= kcl_start_epoch:
            kcl_epoch = epoch - kcl_start_epoch
            if kcl_epoch < kcl_warmup_epochs:
                kcl_weight = kcl_weight_target * (kcl_epoch + 1) / kcl_warmup_epochs
            else:
                kcl_weight = kcl_weight_target
        elif epoch < kcl_start_epoch:
            kcl_weight = 0.0  # No KCL during current warmup
        else:
            kcl_weight = kcl_weight_target

        # Apply physics constraint warmup (diff pair, current mirrors)
        constraint_start_epoch = constraint_start_epoch_cfg if constraint_start_epoch_cfg > 0 else current_warmup_epochs
        if constraint_warmup_epochs > 0 and epoch >= constraint_start_epoch:
            constraint_epoch = epoch - constraint_start_epoch
            if constraint_epoch < constraint_warmup_epochs:
                constraint_weight = constraint_weight_target * (constraint_epoch + 1) / constraint_warmup_epochs
            else:
                constraint_weight = constraint_weight_target
        elif epoch < constraint_start_epoch:
            constraint_weight = 0.0
        else:
            constraint_weight = constraint_weight_target

        # Apply gm physics loss warmup
        gm_phy_start = gm_physics_loss_start_epoch_cfg if gm_physics_loss_start_epoch_cfg > 0 else current_warmup_epochs
        if gm_physics_loss_warmup_epochs > 0 and epoch >= gm_phy_start:
            gm_phy_epoch = epoch - gm_phy_start
            if gm_phy_epoch < gm_physics_loss_warmup_epochs:
                gm_physics_loss_weight = gm_physics_loss_weight_target * (gm_phy_epoch + 1) / gm_physics_loss_warmup_epochs
            else:
                gm_physics_loss_weight = gm_physics_loss_weight_target
        elif epoch < gm_phy_start:
            gm_physics_loss_weight = 0.0
        else:
            gm_physics_loss_weight = gm_physics_loss_weight_target

        # Apply AC loss warmup
        ac_loss_start_epoch = ac_loss_start_epoch_cfg if ac_loss_start_epoch_cfg > 0 else current_warmup_epochs
        if ac_loss_warmup_epochs > 0 and epoch >= ac_loss_start_epoch:
            ac_epoch = epoch - ac_loss_start_epoch
            if ac_epoch < ac_loss_warmup_epochs:
                ac_loss_weight = ac_loss_weight_target * (ac_epoch + 1) / ac_loss_warmup_epochs
            else:
                ac_loss_weight = ac_loss_weight_target
        elif epoch < ac_loss_start_epoch:
            ac_loss_weight = 0.0
        else:
            ac_loss_weight = ac_loss_weight_target

        # Apply supervised gm/gds loss warmup
        ss_loss_start_epoch = ss_loss_start_epoch_cfg if ss_loss_start_epoch_cfg > 0 else current_warmup_epochs
        if ss_loss_warmup_epochs > 0 and epoch >= ss_loss_start_epoch:
            ss_epoch = epoch - ss_loss_start_epoch
            if ss_epoch < ss_loss_warmup_epochs:
                ss_loss_weight = ss_loss_weight_target * (ss_epoch + 1) / ss_loss_warmup_epochs
            else:
                ss_loss_weight = ss_loss_weight_target
        elif epoch < ss_loss_start_epoch:
            ss_loss_weight = 0.0
        else:
            ss_loss_weight = ss_loss_weight_target

        # Apply triode physics loss warmup
        tri_phy_start = triode_physics_loss_start_epoch_cfg if triode_physics_loss_start_epoch_cfg > 0 else current_warmup_epochs
        if triode_physics_loss_warmup_epochs > 0 and epoch >= tri_phy_start:
            tri_phy_epoch = epoch - tri_phy_start
            if tri_phy_epoch < triode_physics_loss_warmup_epochs:
                triode_physics_loss_weight = triode_physics_loss_weight_target * (tri_phy_epoch + 1) / triode_physics_loss_warmup_epochs
            else:
                triode_physics_loss_weight = triode_physics_loss_weight_target
        elif epoch < tri_phy_start:
            triode_physics_loss_weight = 0.0
        else:
            triode_physics_loss_weight = triode_physics_loss_weight_target

        # Apply cutoff physics loss warmup
        cut_phy_start = cutoff_physics_loss_start_epoch_cfg if cutoff_physics_loss_start_epoch_cfg > 0 else current_warmup_epochs
        if cutoff_physics_loss_warmup_epochs > 0 and epoch >= cut_phy_start:
            cut_phy_epoch = epoch - cut_phy_start
            if cut_phy_epoch < cutoff_physics_loss_warmup_epochs:
                cutoff_physics_loss_weight = cutoff_physics_loss_weight_target * (cut_phy_epoch + 1) / cutoff_physics_loss_warmup_epochs
            else:
                cutoff_physics_loss_weight = cutoff_physics_loss_weight_target
        elif epoch < cut_phy_start:
            cutoff_physics_loss_weight = 0.0
        else:
            cutoff_physics_loss_weight = cutoff_physics_loss_weight_target

        # Region loss: apply start_epoch
        region_loss_weight = region_loss_weight_target if epoch >= region_loss_start_epoch_cfg else 0.0

        loss, mae_norm, voltage_loss, current_loss, current_mae_ua, kcl_loss, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss, gm_physics_loss, ac_loss, ss_loss, triode_physics_loss, triode_eq1_loss, triode_eq2_loss, triode_eq3_loss, cutoff_physics_loss, region_loss = train_epoch(
            model, train_loader, optimizer, args.gradient_clip, args.device, scaler,
            predict_currents=predict_currents, current_weight=current_weight, kcl_weight=kcl_weight,
            current_mean=current_mean, current_std=current_std, loss_type=loss_type, huber_delta=huber_delta,
            kcl_min_current=kcl_min_current, constraint_weight=constraint_weight, amp_dtype=amp_dtype,
            vdc_mean=vdc_mean, vdc_std=vdc_std,
            stage2_nodes=stage2_nodes, stage2_weight=stage2_weight,
            use_terminal_voltage_loss=use_terminal_voltage_loss,
            gm_physics_loss_weight=gm_physics_loss_weight, gm_physics_min_vov=gm_physics_min_vov, gm_physics_use_clm=gm_physics_use_clm,
            gm_physics_use_gt_voltages=gm_physics_use_gt_voltages,
            ss_gm_mean=ss_gm_mean if has_ss else 0.0, ss_gm_std=ss_gm_std if has_ss else 1.0,
            ss_gds_mean=ss_gds_mean if has_ss else 0.0, ss_gds_std=ss_gds_std if has_ss else 1.0,
            ac_loss_weight=ac_loss_weight, ac_mean=ac_mean, ac_std=ac_std, ac_components=ac_components,
            ss_loss_weight=ss_loss_weight,
            triode_physics_loss_weight=triode_physics_loss_weight, triode_physics_config=triode_physics_config,
            cutoff_physics_loss_weight=cutoff_physics_loss_weight, cutoff_physics_n_nmos=cutoff_physics_n_nmos, cutoff_physics_n_pmos=cutoff_physics_n_pmos,
            region_loss_weight=region_loss_weight,
            kcl_mode=kcl_mode,
            kcl_exclusive=kcl_exclusive,
            kcl_detach_backbone=kcl_detach_backbone,
            kcl_violation_threshold=kcl_violation_threshold,
            kcl_huber_delta=kcl_huber_delta,
            kcl_conservation=kcl_conservation,
            kcl_skip_two_term=kcl_skip_two_term,
            kcl_only_two_term=kcl_only_two_term,
        )

        mae_mv = mae_norm * vdc_std * 1000
        train_losses.append(loss)
        train_voltage_losses.append(voltage_loss)
        train_current_losses.append(current_loss)
        train_ss_losses.append(ss_loss)
        train_kcl_losses.append(kcl_loss)
        train_ac_losses.append(ac_loss)
        train_region_losses.append(region_loss)
        train_gm_physics_losses.append(gm_physics_loss)
        train_triode_physics_losses.append(triode_physics_loss)
        train_cutoff_physics_losses.append(cutoff_physics_loss)
        train_maes.append(mae_mv)
        train_current_maes.append(current_mae_ua)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Validation
        if val_loader and epoch % val_freq == 0:
            val_loss, val_mae_mv, val_v_loss, val_c_loss, val_c_mae, acc80, acc50, acc20, acc10, current_acc50, current_acc20, current_acc5, current_acc2, val_kcl_loss, val_dp_loss, val_mirror_loss, val_os_loss, val_lm_loss, val_gm_physics_loss, val_ac_loss, val_ss_loss, val_triode_physics_loss, val_triode_eq1_loss, val_triode_eq2_loss, val_triode_eq3_loss, val_cutoff_physics_loss, val_region_loss = validate(
                model, val_loader, args.device, vdc_mean, vdc_std, current_mean, current_std,
                predict_currents=predict_currents, current_weight=current_weight, kcl_weight=kcl_weight,
                loss_type=loss_type, huber_delta=huber_delta, kcl_min_current=kcl_min_current,
                constraint_weight=constraint_weight,
                stage2_nodes=stage2_nodes, stage2_weight=stage2_weight,
                use_terminal_voltage_loss=use_terminal_voltage_loss,
                gm_physics_loss_weight=gm_physics_loss_weight, gm_physics_min_vov=gm_physics_min_vov, gm_physics_use_clm=gm_physics_use_clm,
                gm_physics_use_gt_voltages=gm_physics_use_gt_voltages,
                ss_gm_mean=ss_gm_mean if has_ss else 0.0, ss_gm_std=ss_gm_std if has_ss else 1.0,
                ss_gds_mean=ss_gds_mean if has_ss else 0.0, ss_gds_std=ss_gds_std if has_ss else 1.0,
                ac_loss_weight=ac_loss_weight, ac_mean=ac_mean, ac_std=ac_std, ac_components=ac_components,
                ss_loss_weight=ss_loss_weight,
                triode_physics_loss_weight=triode_physics_loss_weight, triode_physics_config=triode_physics_config,
                cutoff_physics_loss_weight=cutoff_physics_loss_weight, cutoff_physics_n_nmos=cutoff_physics_n_nmos, cutoff_physics_n_pmos=cutoff_physics_n_pmos,
                region_loss_weight=region_loss_weight,
                kcl_mode=kcl_mode,
                kcl_exclusive=kcl_exclusive,
                kcl_detach_backbone=kcl_detach_backbone,
                kcl_violation_threshold=kcl_violation_threshold,
                kcl_huber_delta=kcl_huber_delta,
                kcl_conservation=kcl_conservation,
                kcl_skip_two_term=kcl_skip_two_term,
                kcl_only_two_term=kcl_only_two_term,
                amp_dtype=amp_dtype,
            )

            val_losses.append(val_loss)
            val_maes.append(val_mae_mv)
            val_voltage_losses.append(val_v_loss)
            val_current_losses.append(val_c_loss)
            val_ss_losses.append(val_ss_loss)
            val_kcl_losses.append(val_kcl_loss)
            val_ac_losses.append(val_ac_loss)
            val_region_losses.append(val_region_loss)
            val_gm_physics_losses.append(val_gm_physics_loss)
            val_triode_physics_losses.append(val_triode_physics_loss)
            val_cutoff_physics_losses.append(val_cutoff_physics_loss)
            val_current_maes.append(val_c_mae)

            if epoch >= args.warmup:
                step_scheduler(scheduler, args.scheduler, val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss, best_epoch = val_loss, epoch
                best_model_state = model.state_dict()
                epochs_without_improvement = 0
                best_metrics.update(val_mae_mv=val_mae_mv, val_current_mae_ua=val_c_mae, acc80=acc80, acc50=acc50, acc20=acc20, acc10=acc10,
                                     current_acc50=current_acc50, current_acc20=current_acc20, current_acc5=current_acc5, current_acc2=current_acc2)
            else:
                epochs_without_improvement += 1

            lr = optimizer.param_groups[0]['lr']
            postfix = {'loss': f'{loss:.4f}', 'v_mae': f'{mae_mv:.1f}mV', 'lr': f'{lr:.2e}'}
            if predict_currents:
                postfix['i_mae'] = f'{current_mae_ua:.1f}µA'
            if predict_currents:
                postfix['kcl'] = f'{kcl_loss:.2e}'
            if constraint_weight > 0:
                postfix['dp'] = f'{diff_pair_loss:.2e}'
                postfix['mir'] = f'{mirror_loss:.2e}'
                postfix['os'] = f'{output_stage_loss:.2e}'
                postfix['lm'] = f'{lambda_mirror_loss:.2e}'
            if gm_physics_loss_weight > 0:
                postfix['gm_phy'] = f'{gm_physics_loss:.2e}'
            if ac_loss_weight > 0:
                postfix['ac'] = f'{ac_loss:.2e}'
            if ss_loss_weight > 0:
                postfix['ss'] = f'{ss_loss:.2e}'
            if triode_physics_loss_weight > 0:
                postfix['tri_phy'] = f'{triode_physics_loss:.2e}'
            if cutoff_physics_loss_weight > 0:
                postfix['cut_phy'] = f'{cutoff_physics_loss:.2e}'
            pbar.set_postfix(postfix)

            # Detailed progress every 50 epochs
            if epoch > 0 and epoch % 50 == 0:
                tr_loss, tr_mae_mv, tr_v_loss, tr_c_loss, tr_c_mae, tr_acc80, tr_acc50, tr_acc20, tr_acc10, tr_current_acc50, tr_current_acc20, tr_current_acc5, tr_current_acc2, tr_kcl_loss, tr_dp_loss, tr_mirror_loss, tr_os_loss, tr_lm_loss, tr_gm_physics_loss, tr_ac_loss, tr_ss_loss, tr_triode_physics_loss, tr_triode_eq1, tr_triode_eq2, tr_triode_eq3, tr_cutoff_physics_loss, tr_region_loss = validate(
                    model, train_loader, args.device, vdc_mean, vdc_std, current_mean, current_std,
                    predict_currents=predict_currents, current_weight=current_weight, kcl_weight=kcl_weight,
                    loss_type=loss_type, huber_delta=huber_delta, constraint_weight=constraint_weight,
                    stage2_nodes=stage2_nodes, stage2_weight=stage2_weight,
                    use_terminal_voltage_loss=use_terminal_voltage_loss,
                    gm_physics_loss_weight=gm_physics_loss_weight, gm_physics_min_vov=gm_physics_min_vov, gm_physics_use_clm=gm_physics_use_clm,
                    ss_gm_mean=ss_gm_mean if has_ss else 0.0, ss_gm_std=ss_gm_std if has_ss else 1.0,
                    ss_gds_mean=ss_gds_mean if has_ss else 0.0, ss_gds_std=ss_gds_std if has_ss else 1.0,
                    ac_loss_weight=ac_loss_weight, ac_mean=ac_mean, ac_std=ac_std, ac_components=ac_components,
                    ss_loss_weight=ss_loss_weight,
                    triode_physics_loss_weight=triode_physics_loss_weight, triode_physics_config=triode_physics_config,
                    cutoff_physics_loss_weight=cutoff_physics_loss_weight, cutoff_physics_n_nmos=cutoff_physics_n_nmos, cutoff_physics_n_pmos=cutoff_physics_n_pmos,
                    region_loss_weight=region_loss_weight,
                    amp_dtype=amp_dtype,
                )
                # Build rows dynamically: (label, train_value, val_value)
                rows = []
                rows.append(("Voltage", f"Loss={tr_v_loss:.4f}  MAE={tr_mae_mv:.1f}mV", f"Loss={val_v_loss:.4f}  MAE={val_mae_mv:.1f}mV"))
                rows.append(("  Acc", f"@80={tr_acc80:.0f}% @50={tr_acc50:.0f}% @20={tr_acc20:.0f}% @10={tr_acc10:.0f}%", f"@80={acc80:.0f}% @50={acc50:.0f}% @20={acc20:.0f}% @10={acc10:.0f}%"))
                if predict_currents:
                    rows.append(("Current", f"Loss={tr_c_loss:.5f}  MAE={tr_c_mae:.1f}uA", f"Loss={val_c_loss:.5f}  MAE={val_c_mae:.1f}uA"))
                    rows.append(("  Acc", f"@50={tr_current_acc50:.0f}% @20={tr_current_acc20:.0f}% @5={tr_current_acc5:.0f}% @2={tr_current_acc2:.0f}%", f"@50={current_acc50:.0f}% @20={current_acc20:.0f}% @5={current_acc5:.0f}% @2={current_acc2:.0f}%"))
                if predict_currents:
                    rows.append(("KCL", f"{tr_kcl_loss:.2e}", f"{val_kcl_loss:.2e}"))
                    # KCL diagnostics: per-component breakdown, GT floor, drop stats
                    try:
                        with torch.no_grad():
                            _vb = next(iter(val_loader))
                            if hasattr(_vb, 'to'):
                                _vb = _vb.to(args.device)
                            _out = model(_vb)
                            _kcl_args = dict(
                                node_currents=_out['node_currents'],
                                edge_index=_vb.edge_index,
                                num_terminals=_vb.num_terminals,
                                train_mask=_vb.train_mask,
                                ptr=_vb.ptr,
                                terminal_current_sign=getattr(_vb, 'terminal_current_sign', None),
                                current_mean=current_mean,
                                current_std=current_std,
                                gt_currents=_vb.node_current_targets if hasattr(_vb, 'node_current_targets') else None,
                                kcl_include_mask=getattr(_vb, 'kcl_include_mask', None),
                                return_stats=True,
                                kcl_mode=kcl_mode,
                            )
                            _, _, _ks = compute_kcl_loss(**_kcl_args)
                        # Per-net KCL relative violations (averaged across batch)
                        if hasattr(_vb, 'node_names') and _vb.node_names is not None:
                            _n_graphs = len(_vb.ptr) - 1
                            _pred_accum = {}
                            _common_args = dict(
                                edge_index=_vb.edge_index,
                                num_terminals=_vb.num_terminals,
                                train_mask=_vb.train_mask,
                                ptr=_vb.ptr,
                                terminal_current_sign=getattr(_vb, 'terminal_current_sign', None),
                                current_mean=current_mean,
                                current_std=current_std,
                                kcl_include_mask=getattr(_vb, 'kcl_include_mask', None),
                                node_names=_vb.node_names,
                            )
                            for _gi in range(_n_graphs):
                                _pv = compute_kcl_per_net_debug(node_currents=_out['node_currents'], graph_idx=_gi, **_common_args)
                                for _name, _val in _pv.items():
                                    _pred_accum.setdefault(_name, []).append(_val)
                            _pred_mean = {k: sum(v)/len(v) for k, v in _pred_accum.items()}
                            _pred_str = " ".join(f"{k}:{v*100:.1f}%" for k, v in sorted(_pred_mean.items(), key=lambda x: -x[1]))
                            rows.append(("  KCL/net", "", _pred_str))
                    except Exception as _e:
                        rows.append(("  detail", f"err: {_e}", ""))
                if ss_loss_weight > 0:
                    rows.append(("SS", f"{tr_ss_loss:.2e}", f"{val_ss_loss:.2e}"))
                    # SS accuracy on train and val
                    def _eval_ss(loader):
                        _gm_e, _gds_e = [], []
                        with torch.no_grad():
                            for _b in loader:
                                if hasattr(_b, 'to'):
                                    _b = _b.to(args.device)
                                _out = model(_b)
                                _gm_p, _gds_p = _out.get('mosfet_gm_pred'), _out.get('mosfet_gds_pred')
                                if _gm_p is None: break
                                _m = getattr(_b, 'mosfet_drain_mask', None)
                                if _m is not None and _m.any():
                                    _gm_e.extend((_gm_p[_m] * ss_gm_std + ss_gm_mean - (_b.node_log_gm[_m] * ss_gm_std + ss_gm_mean)).abs().cpu().tolist())
                                    _gds_e.extend((_gds_p[_m] * ss_gds_std + ss_gds_mean - (_b.node_log_gds[_m] * ss_gds_std + ss_gds_mean)).abs().cpu().tolist())
                        return np.array(_gm_e) if _gm_e else None, np.array(_gds_e) if _gds_e else None
                    _tr_gm, _tr_gds = _eval_ss(train_loader)
                    _vl_gm, _vl_gds = _eval_ss(val_loader)
                    if _vl_gm is not None:
                        rows.append(("  gm", f"MAE={_tr_gm.mean():.3f}log  1.5x={100*np.mean(_tr_gm<np.log10(1.5)):.0f}%  2x={100*np.mean(_tr_gm<np.log10(2)):.0f}%", f"MAE={_vl_gm.mean():.3f}log  1.5x={100*np.mean(_vl_gm<np.log10(1.5)):.0f}%  2x={100*np.mean(_vl_gm<np.log10(2)):.0f}%"))
                        rows.append(("  gds", f"MAE={_tr_gds.mean():.3f}log  1.5x={100*np.mean(_tr_gds<np.log10(1.5)):.0f}%  2x={100*np.mean(_tr_gds<np.log10(2)):.0f}%", f"MAE={_vl_gds.mean():.3f}log  1.5x={100*np.mean(_vl_gds<np.log10(1.5)):.0f}%  2x={100*np.mean(_vl_gds<np.log10(2)):.0f}%"))
                if gm_physics_loss_weight > 0:
                    rows.append(("GmPhy", f"{tr_gm_physics_loss:.2e}", f"{val_gm_physics_loss:.2e}"))
                if triode_physics_loss_weight > 0:
                    rows.append(("TriPhy", f"{tr_triode_physics_loss:.2e}", f"{val_triode_physics_loss:.2e}"))
                    rows.append(("  eq1gm", f"{tr_triode_eq1:.2e}", f"{val_triode_eq1_loss:.2e}"))
                    rows.append(("  eq2gds", f"{tr_triode_eq2:.2e}", f"{val_triode_eq2_loss:.2e}"))
                    rows.append(("  eq3sc", f"{tr_triode_eq3:.2e}", f"{val_triode_eq3_loss:.2e}"))
                if cutoff_physics_loss_weight > 0:
                    rows.append(("CutPhy", f"{tr_cutoff_physics_loss:.2e}", f"{val_cutoff_physics_loss:.2e}"))
                if region_loss_weight > 0:
                    rows.append(("Region", f"{tr_region_loss:.2e}", f"{val_region_loss:.2e}"))
                    # Quick region accuracy on train and val
                    def _eval_region(loader):
                        _preds, _labels = [], []
                        with torch.no_grad():
                            for _b in loader:
                                if hasattr(_b, 'to'):
                                    _b = _b.to(args.device)
                                _out = model(_b)
                                _rpred = _out.get('mosfet_region_pred')
                                if _rpred is None: break
                                _m = _b.mosfet_drain_mask & (_b.node_region_labels >= 0)
                                if _m.any():
                                    # Ordinal: <0.5 → cutoff(0), 0.5-1.5 → triode(1), >1.5 → sat(2)
                                    _preds.append(_rpred[_m].round().clamp(0, 2).long().cpu())
                                    _labels.append(_b.node_region_labels[_m].cpu())
                        if not _preds: return None
                        _p = torch.cat(_preds); _l = torch.cat(_labels)
                        _acc = 100 * (_p == _l).float().mean().item()
                        _accs = []
                        for _c in range(3):
                            _cm = _l == _c
                            _accs.append(100 * ((_p == _c) & _cm).sum().item() / _cm.sum().item() if _cm.any() else 0)
                        return _acc, _accs  # overall, [cutoff, triode, sat]
                    _tr_r = _eval_region(train_loader)
                    _vl_r = _eval_region(val_loader)
                    if _tr_r and _vl_r:
                        rows.append(("", f"Acc={_tr_r[0]:.0f}%  cut={_tr_r[1][0]:.0f}% tri={_tr_r[1][1]:.0f}% sat={_tr_r[1][2]:.0f}%",
                                        f"Acc={_vl_r[0]:.0f}%  cut={_vl_r[1][0]:.0f}% tri={_vl_r[1][1]:.0f}% sat={_vl_r[1][2]:.0f}%"))
                if ac_loss_weight > 0:
                    rows.append(("AC", f"{tr_ac_loss:.2e}", f"{val_ac_loss:.2e}"))
                    # AC accuracy on train and val
                    def _eval_ac(loader):
                        _errs = {c: [] for c in ac_components}
                        with torch.no_grad():
                            for _b in loader:
                                if hasattr(_b, 'to'):
                                    _b = _b.to(args.device)
                                _out = model(_b)
                                _ac_p = _out.get('ac_pred')
                                if _ac_p is None: break
                                _v = _b.ac_valid if hasattr(_b, 'ac_valid') else None
                                if _v is None or not _v.any(): continue
                                _pd = _ac_p[_v] * ac_std.to(args.device) + ac_mean.to(args.device)
                                for _i, _c in enumerate(ac_components):
                                    if _c == 'ugbw':
                                        _t = torch.log10(_b.ac_ugbw[_v].clamp(min=1.0).to(args.device))
                                    elif _c == 'pm':
                                        _t = _b.ac_pm[_v].to(args.device)
                                    elif _c == 'am':
                                        _t = _b.ac_am[_v].to(args.device)
                                    _errs[_c].extend((_pd[:, _i] - _t).abs().cpu().tolist())
                        return {c: np.array(v) for c, v in _errs.items() if v}
                    _tr_ac = _eval_ac(train_loader)
                    _vl_ac = _eval_ac(val_loader)
                    for _c in ac_components:
                        if _c in _vl_ac:
                            _te, _ve = _tr_ac.get(_c, _vl_ac[_c]), _vl_ac[_c]
                            if _c == 'ugbw':
                                rows.append(("  UGBW", f"MAE={_te.mean():.3f}log  1.5x={100*np.mean(_te<np.log10(1.5)):.0f}%  2x={100*np.mean(_te<np.log10(2)):.0f}%", f"MAE={_ve.mean():.3f}log  1.5x={100*np.mean(_ve<np.log10(1.5)):.0f}%  2x={100*np.mean(_ve<np.log10(2)):.0f}%"))
                            elif _c == 'pm':
                                rows.append(("  PM", f"MAE={_te.mean():.1f}d  <5={100*np.mean(_te<5):.0f}%  <10={100*np.mean(_te<10):.0f}%", f"MAE={_ve.mean():.1f}d  <5={100*np.mean(_ve<5):.0f}%  <10={100*np.mean(_ve<10):.0f}%"))
                            elif _c == 'am':
                                rows.append(("  AM", f"MAE={_te.mean():.1f}dB  <3={100*np.mean(_te<3):.0f}%  <6={100*np.mean(_te<6):.0f}%", f"MAE={_ve.mean():.1f}dB  <3={100*np.mean(_ve<3):.0f}%  <6={100*np.mean(_ve<6):.0f}%"))
                if constraint_weight > 0:
                    rows.append(("DiffPr", f"{tr_dp_loss:.2e}", f"{val_dp_loss:.2e}"))
                    rows.append(("Mirror", f"{tr_mirror_loss:.2e}", f"{val_mirror_loss:.2e}"))
                    rows.append(("OutStg", f"{tr_os_loss:.2e}", f"{val_os_loss:.2e}"))
                    rows.append(("LamMir", f"{tr_lm_loss:.2e}", f"{val_lm_loss:.2e}"))

                # Print with dynamic alignment
                label_w = max(len(r[0]) for r in rows)
                tr_w = max(len(r[1]) for r in rows)
                print(f"\nEpoch {epoch:3d} | LR={lr:.2e} | BestLoss={best_val_loss:.4f}")
                print(f"  {'':>{label_w}}   {'Train':^{tr_w}} | {'Val'}")
                for label, tr_val, val_val in rows:
                    print(f"  {label:>{label_w}}   {tr_val:<{tr_w}} | {val_val}")
                tr_total = f"Loss={tr_loss:.4f}"
                val_total = f"Loss={val_loss:.4f}"
                print(f"  {'Total':>{label_w}}   {tr_total:<{tr_w}} | {val_total}")

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save best model with full config
    if best_model_state is not None:
        save_path = output_path / 'best_model.pt'

        full_config = build_full_config(
            args, total_input_dim, predict_currents, voltage_head_config,
            current_head_config, current_weight_target, loss_type, huber_delta,
            val_freq, early_stopping_patience, dataset_path,
            derive_currents_from_voltage, args.mosfet_current_mlp_config,
            getattr(args, 'use_gnn_current_prediction', False),
            getattr(args, 'current_gnn_config', {}),
            getattr(args, 'use_frozen_device_mlp', False),
            getattr(args, 'frozen_device_mlp_config', {}),
            getattr(args, 'use_refinement_pass', False),
            getattr(args, 'refinement_config', {}),
        )
        stats = {
            'vdc': {'mean': vdc_mean, 'std': vdc_std},
            'current': {'mean': current_mean, 'std': current_std}
        }

        # Add finetune provenance info
        if hasattr(args, '_finetune_source'):
            full_config['finetune_source'] = args._finetune_source
            full_config['finetune_source_epoch'] = args._finetune_source_epoch
            full_config['freeze_backbone'] = getattr(args, 'freeze_backbone', False)
            full_config['backbone_lr_scale'] = getattr(args, 'backbone_lr_scale', 0.1)

        save_checkpoint(save_path, best_model_state, best_epoch, best_val_loss, full_config, stats)
        print(f"\nSaved best model (epoch {best_epoch}) to {save_path}")
        print(f"Saved config to {output_path / 'config.yaml'}")

    # Print summary
    training_time = time.time() - training_start_time
    print(f"\n{'='*50}\n=== TRAINING COMPLETE ===\n{'='*50}")
    print(f"Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best Val MAE: {best_metrics['val_mae_mv']:.2f}mV")
    if predict_currents:
        print(f"Best Val Current MAE: {best_metrics['val_current_mae_ua']:.1f}µA")
    if predict_currents:
        print(f"Val Accuracy@80mV: {best_metrics['acc80']:5.2f}% | Current @50uA: {best_metrics['current_acc50']:5.2f}%")
        print(f"Val Accuracy@50mV: {best_metrics['acc50']:5.2f}% | Current @20uA: {best_metrics['current_acc20']:5.2f}%")
        print(f"Val Accuracy@20mV: {best_metrics['acc20']:5.2f}% | Current @5uA:  {best_metrics['current_acc5']:5.2f}%")
        print(f"Val Accuracy@10mV: {best_metrics['acc10']:5.2f}% | Current @2uA:  {best_metrics['current_acc2']:5.2f}%")
    else:
        print(f"Val Accuracy@80mV: {best_metrics['acc80']:.2f}%")
        print(f"Val Accuracy@50mV: {best_metrics['acc50']:.2f}%")
        print(f"Val Accuracy@20mV: {best_metrics['acc20']:.2f}%")
        print(f"Val Accuracy@10mV: {best_metrics['acc10']:.2f}%")

    # End-of-training SS and AC evaluation on best model
    if best_model_state is not None and val_loader:
        model.load_state_dict(best_model_state)
        model.eval()

        # SS evaluation: denormalized MAE in log10 space
        if ss_loss_weight_target > 0 and has_ss:
            all_gm_errors = []
            all_gds_errors = []
            with torch.no_grad():
                for batch in val_loader:
                    if hasattr(batch, 'to'):
                        batch = batch.to(args.device)
                    out_dict = model(batch)
                    gm_pred = out_dict.get('mosfet_gm_pred')
                    gds_pred = out_dict.get('mosfet_gds_pred')
                    if gm_pred is None:
                        break
                    mask = batch.mosfet_drain_mask
                    if mask.any():
                        # Denormalize: z-score back to log10 space
                        gm_pred_log = gm_pred[mask] * ss_gm_std + ss_gm_mean
                        gm_target_log = batch.node_log_gm[mask] * ss_gm_std + ss_gm_mean
                        gds_pred_log = gds_pred[mask] * ss_gds_std + ss_gds_mean
                        gds_target_log = batch.node_log_gds[mask] * ss_gds_std + ss_gds_mean
                        all_gm_errors.extend((gm_pred_log - gm_target_log).abs().cpu().tolist())
                        all_gds_errors.extend((gds_pred_log - gds_target_log).abs().cpu().tolist())

            if all_gm_errors:
                gm_errs = np.array(all_gm_errors)
                gds_errs = np.array(all_gds_errors)
                # log10 error of X means prediction is off by 10^X factor
                print(f"\n--- SS Evaluation (best model, val set) ---")
                print(f"  gm  MAE: {gm_errs.mean():.3f} log10  (median {np.median(gm_errs):.3f})")
                print(f"  gds MAE: {gds_errs.mean():.3f} log10  (median {np.median(gds_errs):.3f})")
                print(f"  gm  within 1.5x: {100*np.mean(gm_errs < np.log10(1.5)):.1f}%")
                print(f"  gm  within 2x:   {100*np.mean(gm_errs < np.log10(2)):.1f}%")
                print(f"  gds within 1.5x: {100*np.mean(gds_errs < np.log10(1.5)):.1f}%")
                print(f"  gds within 2x:   {100*np.mean(gds_errs < np.log10(2)):.1f}%")

        # AC evaluation: denormalized MAE in real units
        if ac_loss_weight_target > 0 and ac_mean is not None and ac_components:
            ac_errors = {comp: [] for comp in ac_components}
            with torch.no_grad():
                for batch in val_loader:
                    if hasattr(batch, 'to'):
                        batch = batch.to(args.device)
                    out_dict = model(batch)
                    ac_pred = out_dict.get('ac_pred')
                    if ac_pred is None:
                        break
                    valid = batch.ac_valid if hasattr(batch, 'ac_valid') else None
                    if valid is None or not valid.any():
                        continue
                    # Denormalize predictions
                    pred_denorm = ac_pred[valid] * ac_std.to(args.device) + ac_mean.to(args.device)
                    # Compute errors for each enabled component
                    for i, comp in enumerate(ac_components):
                        if comp == 'ugbw':
                            target = torch.log10(batch.ac_ugbw[valid].clamp(min=1.0).to(args.device))
                        elif comp == 'pm':
                            target = batch.ac_pm[valid].to(args.device)
                        elif comp == 'am':
                            target = batch.ac_am[valid].to(args.device)
                        ac_errors[comp].extend((pred_denorm[:, i] - target).abs().cpu().tolist())

            if ac_errors[ac_components[0]]:
                print(f"\n--- AC Evaluation (best model, val set) ---")
                for comp in ac_components:
                    errs = np.array(ac_errors[comp])
                    if comp == 'ugbw':
                        print(f"  UGBW  MAE: {errs.mean():.3f} log10(Hz)  (median {np.median(errs):.3f})")
                        print(f"  UGBW  within 1.5x: {100*np.mean(errs < np.log10(1.5)):.1f}%")
                        print(f"  UGBW  within 2x:   {100*np.mean(errs < np.log10(2)):.1f}%")
                    elif comp == 'pm':
                        print(f"  PM    MAE: {errs.mean():.1f} deg  (median {np.median(errs):.1f})")
                        print(f"  PM    within 5d:   {100*np.mean(errs < 5):.1f}%")
                        print(f"  PM    within 10d:  {100*np.mean(errs < 10):.1f}%")
                    elif comp == 'am':
                        print(f"  AM    MAE: {errs.mean():.1f} dB  (median {np.median(errs):.1f})")
                        print(f"  AM    within 3dB:  {100*np.mean(errs < 3):.1f}%")
                        print(f"  AM    within 6dB:  {100*np.mean(errs < 6):.1f}%")

        # Region classification evaluation
        if region_loss_weight_target > 0:
            region_names = ['cutoff', 'triode', 'saturation']
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    if hasattr(batch, 'to'):
                        batch = batch.to(args.device)
                    out_dict = model(batch)
                    rpred = out_dict.get('mosfet_region_pred')
                    if rpred is None:
                        break
                    mask = batch.mosfet_drain_mask & (batch.node_region_labels >= 0)
                    if mask.any():
                        preds = rpred[mask].round().clamp(0, 2).long()
                        labels = batch.node_region_labels[mask]
                        all_preds.append(preds.cpu())
                        all_labels.append(labels.cpu())

            if all_preds:
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                total_correct = (all_preds == all_labels).sum().item()
                total_samples = len(all_labels)
                overall_acc = 100 * total_correct / total_samples

                print(f"\n--- Region Classification (best model, val set) ---")
                print(f"  Overall accuracy: {overall_acc:.1f}% ({total_correct}/{total_samples})")
                for c in range(3):
                    c_mask = all_labels == c
                    c_total = c_mask.sum().item()
                    if c_total > 0:
                        c_correct = ((all_preds == c) & c_mask).sum().item()
                        c_acc = 100 * c_correct / c_total
                        print(f"  {region_names[c]:>10s}: {c_acc:5.1f}% ({c_correct}/{c_total})")
                    else:
                        print(f"  {region_names[c]:>10s}: N/A (0 samples)")

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_voltage_losses, val_voltage_losses,
        train_maes, val_maes, learning_rates, best_metrics, best_val_loss, best_epoch,
        output_path / 'training_curve.png', val_freq=val_freq, predict_currents=predict_currents,
        train_current_losses=train_current_losses, val_current_losses=val_current_losses,
        train_current_maes=train_current_maes, val_current_maes=val_current_maes,
        train_ss_losses=train_ss_losses, val_ss_losses=val_ss_losses,
        train_kcl_losses=train_kcl_losses, val_kcl_losses=val_kcl_losses,
        train_ac_losses=train_ac_losses, val_ac_losses=val_ac_losses,
        train_region_losses=train_region_losses, val_region_losses=val_region_losses,
        train_gm_physics_losses=train_gm_physics_losses, val_gm_physics_losses=val_gm_physics_losses,
        train_triode_physics_losses=train_triode_physics_losses if triode_physics_loss_weight_target > 0 else None,
        val_triode_physics_losses=val_triode_physics_losses if triode_physics_loss_weight_target > 0 else None,
        train_cutoff_physics_losses=train_cutoff_physics_losses if cutoff_physics_loss_weight_target > 0 else None,
        val_cutoff_physics_losses=val_cutoff_physics_losses if cutoff_physics_loss_weight_target > 0 else None,
        num_train_batches=len(train_loader), num_val_batches=len(val_loader) if val_loader else 0
    )


if __name__ == '__main__':
    main()
