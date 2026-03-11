#!/usr/bin/env python3
"""
Evaluate a trained GNN model on validation or test data.

Usage:
    python scripts/evaluate.py --checkpoint path/to/best_model.pt
    python scripts/evaluate.py --checkpoint path/to/best_model.pt --split test
    python scripts/evaluate.py --checkpoint path/to/best_model.pt --dataset path/to/dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch

from src.training.checkpoint import load_checkpoint
from src.training.data_loading import (
    PrebatchedLoader,
    load_prebatched_variant,
    normalize_batches_vdc,
    normalize_batches_current,
)
from src.training.loops import validate


def evaluate_model(model, data_path, split, stats, config, device):
    """
    Evaluate model on specified data split.

    Args:
        model: Loaded model
        data_path: Path to dataset directory
        split: 'val' or 'test'
        stats: Normalization stats
        config: Model config
        device: Device to use

    Returns:
        Dict of evaluation metrics
    """
    data_path = Path(data_path)
    split_dir = data_path / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    print(f"\nLoading {split} data from: {split_dir}")
    batches = load_prebatched_variant(split_dir, variant_id=0, device=device)
    print(f"  Loaded {len(batches)} batches")

    # Get normalization stats
    vdc_stats = stats.get('vdc', {})
    current_stats = stats.get('current', stats.get('curr', {}))

    vdc_mean = vdc_stats.get('mean', 0.0)
    vdc_std = vdc_stats.get('std', 1.0)
    current_mean = current_stats.get('mean', 0.0)
    current_std = current_stats.get('std', 1.0)

    print(f"\nNormalization stats:")
    print(f"  VDC: mean={vdc_mean:.4f}, std={vdc_std:.4f}")
    print(f"  Current: mean={current_mean:.4f}, std={current_std:.4f} (log scale)")

    # Always normalize targets (prebatched data contains raw values)
    print("  Normalizing targets...")
    normalize_batches_vdc(batches, vdc_mean, vdc_std)

    sample = batches[0]
    if hasattr(sample, 'node_current_targets') and sample.node_current_targets is not None:
        normalize_batches_current(batches, current_mean, current_std)

    loader = PrebatchedLoader(batches, shuffle=False)

    # Run evaluation
    predict_currents = config.get('predict_currents', False)
    current_weight = config.get('current_weight', 1.0)
    loss_type = config.get('loss_type', 'mse')
    huber_delta = config.get('huber_delta', 1.0)

    print(f"\nRunning evaluation...")
    results = validate(
        model, loader, device,
        vdc_mean, vdc_std, current_mean, current_std,
        predict_currents=predict_currents,
        current_weight=current_weight,
        loss_type=loss_type,
        huber_delta=huber_delta
    )

    # Unpack results (must match validate() return signature)
    (avg_loss, mae_mv, avg_voltage_loss, avg_current_loss, current_mae_ua,
     acc80, acc50, acc20, acc10, current_acc50, current_acc20, current_acc10, current_acc5,
     avg_kcl_loss, *_rest) = results

    return {
        'loss': avg_loss,
        'voltage_loss': avg_voltage_loss,
        'current_loss': avg_current_loss,
        'mae_mv': mae_mv,
        'current_mae_ua': current_mae_ua,
        'acc80': acc80,
        'acc50': acc50,
        'acc20': acc20,
        'acc10': acc10,
        'current_acc50': current_acc50,
        'current_acc20': current_acc20,
        'current_acc10': current_acc10,
        'current_acc5': current_acc5,
        'predict_currents': predict_currents,
    }


def print_results(metrics, split):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS ({split})")
    print(f"{'='*50}")

    print(f"\nVoltage Prediction:")
    print(f"  MAE: {metrics['mae_mv']:.2f} mV")
    print(f"  Accuracy @80mV: {metrics['acc80']:.1f}%")
    print(f"  Accuracy @50mV: {metrics['acc50']:.1f}%")
    print(f"  Accuracy @20mV: {metrics['acc20']:.1f}%")
    print(f"  Loss (MSE): {metrics['voltage_loss']:.6f}")

    if metrics['predict_currents']:
        print(f"\nCurrent Prediction:")
        print(f"  MAE: {metrics['current_mae_ua']:.2f} µA")
        print(f"  Accuracy @50%: {metrics['current_acc50']:.1f}%")
        print(f"  Accuracy @20%: {metrics['current_acc20']:.1f}%")
        print(f"  Accuracy @10%: {metrics['current_acc10']:.1f}%")
        print(f"  Accuracy @5%: {metrics['current_acc5']:.1f}%")
        print(f"  Loss (MSE): {metrics['current_loss']:.6f}")

    print(f"\nCombined Loss: {metrics['loss']:.6f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained GNN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best_model.pt)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset (default: infer from checkpoint path)')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Data split to evaluate on (default: test, falls back to val)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Infer dataset path from checkpoint if not provided
    if args.dataset is None:
        args.dataset = Path(args.checkpoint).parent

    # Load model using checkpoint module (handles VN detection properly)
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config, stats = load_checkpoint(args.checkpoint, args.device)

    # Print detected architecture
    detected = config.get('_detected', {})
    print(f"\nModel architecture:")
    print(f"  input_dim: {detected.get('input_dim', 'unknown')}")
    print(f"  hidden_dim: {config.get('hidden', config.get('hidden_dim', 128))}")
    print(f"  num_layers: {config.get('layers', config.get('num_layers', 15))}")
    print(f"  jk_mode: {config.get('jk_mode', 'cat')}, jk_attention: {config.get('jk_attention', False)}")
    print(f"  use_virtual_node: {detected.get('has_virtual_node', False)}")
    print(f"  predict_currents: {config.get('predict_currents', False)}")
    print(f"  Model class: {detected.get('model_class', 'unknown')}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Check split exists and has data, fall back to val if test not found or empty
    split = args.split
    split_dir = Path(args.dataset) / split
    if split == 'test':
        if not split_dir.exists():
            print(f"\nTest split not found at {split_dir}, falling back to val")
            split = 'val'
        elif not list(split_dir.glob('*.pt')):
            print(f"\nTest split is empty at {split_dir}, falling back to val")
            split = 'val'

    # Evaluate
    metrics = evaluate_model(model, args.dataset, split, stats, config, args.device)

    # Print results
    print_results(metrics, split)

    return metrics


if __name__ == '__main__':
    main()
