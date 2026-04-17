#!/usr/bin/env python3
"""
Re-evaluate 10k and 30k models to compute missing metrics
that weren't logged in the older training script.

Usage:
    python scripts/eval_missing_metrics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.training.checkpoint import create_model
from src.training.data_loading import (
    PrebatchedLoader,
    load_prebatched_variant,
    normalize_batches_vdc,
    normalize_batches_current,
    add_ss_node_targets,
    compute_ss_normalization,
)
from src.training.loops import validate


EXPERIMENTS = [
    {
        'name': '10k (3stage_10k_vi_ss)',
        'dataset': 'datasets/opamp_3stage_fan_smc_v8_10k_nofil',
        'checkpoint': 'datasets/opamp_3stage_fan_smc_v8_10k_nofil/experiments/3stage_10k_vi_ss/best_model.pt',
        'expected_v_mae': 14.84,
        'expected_gm_mae': 0.066,
        'expected_gds_mae': 0.093,
    },
    {
        'name': '30k (3stage_30k_vi_ss)',
        'dataset': 'datasets/opamp_3stage_fan_smc_v8_30k_nofil',
        'checkpoint': 'datasets/opamp_3stage_fan_smc_v8_30k_nofil/experiments/3stage_30k_vi_ss/best_model.pt',
        'expected_v_mae': 11.10,
        'expected_gm_mae': 0.056,
        'expected_gds_mae': 0.077,
    },
]


def load_old_model(checkpoint_path, device):
    """Load old DeepGENConv checkpoint, inferring SS head architecture from state_dict."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    stats = ckpt.get('stats', {})
    state_dict = ckpt['model_state_dict']

    # Detect architecture from state_dict
    has_vn_keys = any('vn_' in k or 'virtual_node' in k for k in state_dict.keys())
    use_virtual_node = config.get('virtual_node', config.get('use_virtual_node', has_vn_keys))

    if 'input_proj.weight' in state_dict:
        input_dim = state_dict['input_proj.weight'].shape[1]
    elif 'input_linear.weight' in state_dict:
        input_dim = state_dict['input_linear.weight'].shape[1]
    else:
        input_dim = config.get('input_dim', 20)

    # Detect SS head config from state_dict shapes
    ss_head_config = config.get('ss_head_config', {})
    if 'gm_head.0.weight' in state_dict:
        ss_input_dim = state_dict['gm_head.0.weight'].shape[1]
        ss_hidden_dim = state_dict['gm_head.0.weight'].shape[0]
        # Count layers: look for gm_head.N.weight where N is a Linear layer
        num_linear = sum(1 for k in state_dict if k.startswith('gm_head.') and k.endswith('.weight')
                        and state_dict[k].dim() == 2)
        ss_head_config['enabled'] = True
        ss_head_config['hidden_dim'] = ss_hidden_dim
        # num_linear includes the final output Linear, but _build_ss_head's num_layers
        # counts only hidden layers (it adds a final Linear(hidden→1) separately)
        ss_head_config['num_layers'] = max(num_linear - 1, 1)
        # Detect pool mode from input dim
        # JK cat with 15 layers → output_dim ~= hidden_dim + skip = 149
        # If input matches ~1x output_dim → drain only; if ~3x → gate+drain+source
        jk_output_dim = ss_hidden_dim + 21  # approximate: hidden + skip (varies)
        if ss_input_dim < 2 * jk_output_dim:
            ss_head_config['pool_mode'] = 'drain'
        else:
            ss_head_config['pool_mode'] = 'concat'
        print(f"  Detected SS head: input={ss_input_dim}, hidden={ss_hidden_dim}, "
              f"layers={num_linear}, pool_mode={ss_head_config.get('pool_mode', 'unknown')}")

    model, model_class = create_model(
        node_feature_dim=input_dim,
        hidden_dim=config.get('hidden', config.get('hidden_dim', 128)),
        num_layers=config.get('layers', config.get('num_layers', 15)),
        dropout=config.get('dropout', 0.0),
        genconv_num_layers=config.get('genconv_num_layers', 2),
        num_mlp_layers=config.get('num_mlp_layers', 3),
        jk_mode=config.get('jk_mode', 'cat'),
        jk_attention=config.get('jk_attention', False),
        jk_learn_temperature=config.get('jk_learn_temperature', False),
        norm_type=config.get('norm_type', 'layer'),
        skip_connection=config.get('skip_connection', True),
        predict_currents=config.get('predict_currents', False),
        voltage_head_config=config.get('voltage_head_config', config.get('voltage_head', {})),
        current_head_config=config.get('current_head_config', config.get('current_head', {})),
        use_virtual_node=use_virtual_node,
        use_attention_pooling=config.get('use_attention_pooling', True),
        vn_learn_temperature=config.get('vn_learn_temperature', False),
        gradient_checkpointing=False,
        device='cpu',
        model_type='deepgen',
        derive_currents_from_voltage=config.get('derive_currents_from_voltage', False),
        mosfet_current_mlp_config=config.get('mosfet_current_mlp_config', {}),
        use_gnn_current_prediction=config.get('use_gnn_current_prediction', False),
        current_gnn_config=config.get('current_gnn_config', {}),
        use_frozen_device_mlp=config.get('use_frozen_device_mlp', False),
        frozen_device_mlp_config=config.get('frozen_device_mlp_config', {}),
        use_refinement_pass=config.get('use_refinement_pass', False),
        refinement_config=config.get('refinement_config', {}),
        ac_head_config=config.get('ac_head_config', {}),
        ss_head_config=ss_head_config,
        region_head_config=config.get('region_head_config', {}),
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, config, stats


def evaluate_experiment(exp, device):
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp['name']}")
    print(f"{'='*60}")

    # 1. Load checkpoint with architecture detection
    print(f"Loading checkpoint: {exp['checkpoint']}")
    model, config, stats = load_old_model(exp['checkpoint'], device)

    # Get normalization stats from checkpoint
    vdc_stats = stats.get('vdc', {})
    current_stats = stats.get('current', stats.get('curr', {}))
    vdc_mean = vdc_stats.get('mean', 0.0)
    vdc_std = vdc_stats.get('std', 1.0)
    current_mean = current_stats.get('mean', 0.0)
    current_std = current_stats.get('std', 1.0)
    print(f"  VDC: mean={vdc_mean:.4f}, std={vdc_std:.4f}")
    print(f"  Current: mean={current_mean:.4f}, std={current_std:.4f}")

    dataset_path = Path(exp['dataset'])

    # 2. Load train data for SS normalization stats
    print("Loading train data for SS normalization stats...")
    train_batches = load_prebatched_variant(dataset_path / 'train', variant_id=0, device=device)
    add_ss_node_targets(train_batches)
    gm_mean, gm_std, gds_mean, gds_std = compute_ss_normalization([train_batches])
    print(f"  SS stats: gm_mean={gm_mean:.4f}, gm_std={gm_std:.4f}, gds_mean={gds_mean:.4f}, gds_std={gds_std:.4f}")

    # Free train data
    del train_batches
    torch.cuda.empty_cache()

    # 3. Load val data
    print("Loading val data...")
    val_batches = load_prebatched_variant(dataset_path / 'val', variant_id=0, device=device)
    add_ss_node_targets(val_batches)
    normalize_batches_vdc(val_batches, vdc_mean, vdc_std)

    sample = val_batches[0]
    if hasattr(sample, 'node_current_targets') and sample.node_current_targets is not None:
        normalize_batches_current(val_batches, current_mean, current_std)

    loader = PrebatchedLoader(val_batches, shuffle=False)

    # 4. Run validate with SS params
    print("Running validation...")
    results = validate(
        model, loader, device,
        vdc_mean, vdc_std, current_mean, current_std,
        predict_currents=True,
        current_weight=1.0,
        loss_type='mse',
        huber_delta=1.0,
        ss_gm_loss_weight=1.0,
        ss_gds_loss_weight=1.0,
        ss_gm_mean=gm_mean,
        ss_gm_std=gm_std,
        ss_gds_mean=gds_mean,
        ss_gds_std=gds_std,
    )

    # 5. Extract all metrics
    (avg_loss, mae_mv, avg_voltage_loss, avg_current_loss, current_mae_ua,
     acc80, acc50, acc20, acc10,
     current_acc50, current_acc20, current_acc10, current_acc5,
     avg_kcl_loss, avg_diff_pair_loss, avg_mirror_loss, avg_output_stage_loss,
     avg_lambda_mirror_loss, avg_gm_physics_loss, avg_ac_loss,
     avg_ss_gm_loss, avg_ss_gds_loss,
     avg_triode_physics_loss, avg_triode_eq1_loss, avg_triode_eq2_loss, avg_triode_eq3_loss,
     avg_cutoff_physics_loss, avg_region_loss,
     rel_metrics, avg_vov_loss, avg_vth_loss) = results

    ss = rel_metrics.get('ss_metrics', {}) or {}

    # 6. Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {exp['name']}")
    print(f"{'='*60}")

    print(f"\n--- Voltage ---")
    print(f"  Val MAE: {mae_mv:.2f}mV")
    print(f"  Median rel: {rel_metrics['v_rel_median']:.2f}%")
    print(f"  @80mV: {acc80:.2f}%  @50mV: {acc50:.2f}%  @20mV: {acc20:.2f}%  @10mV: {acc10:.2f}%")
    print(f"  Rel @1%: {rel_metrics['v_rel_acc'][1]:.2f}%  @5%: {rel_metrics['v_rel_acc'][5]:.2f}%  @10%: {rel_metrics['v_rel_acc'][10]:.2f}%  @20%: {rel_metrics['v_rel_acc'][20]:.2f}%")

    print(f"\n--- Current ---")
    print(f"  Val MAE: {current_mae_ua:.1f}µA")
    print(f"  Median rel: {rel_metrics['i_rel_median']:.2f}%")
    print(f"  @50uA: {rel_metrics['i_abs_acc'][50]:.2f}%  @20uA: {rel_metrics['i_abs_acc'][20]:.2f}%  @5uA: {rel_metrics['i_abs_acc'][5]:.2f}%  @2uA: {rel_metrics['i_abs_acc'][2]:.2f}%")
    print(f"  Rel @1%: {rel_metrics['i_rel_acc'][1]:.2f}%  @5%: {rel_metrics['i_rel_acc'][5]:.2f}%  @10%: {rel_metrics['i_rel_acc'][10]:.2f}%  @20%: {rel_metrics['i_rel_acc'][20]:.2f}%")

    print(f"\n--- gm ---")
    print(f"  MAE: {ss.get('gm_log_mae', 0):.3f} log10")
    print(f"  Median: {ss.get('gm_log_median', 0):.3f} log10")
    print(f"  Median rel: {ss.get('gm_median', 0):.1f}%")
    gm_acc = ss.get('gm_acc', {})
    print(f"  @10%: {gm_acc.get(10, 0):.1f}%  @20%: {gm_acc.get(20, 0):.1f}%  @50%: {gm_acc.get(50, 0):.1f}%")

    print(f"\n--- gds ---")
    print(f"  MAE: {ss.get('gds_log_mae', 0):.3f} log10")
    print(f"  Median: {ss.get('gds_log_median', 0):.3f} log10")
    print(f"  Median rel: {ss.get('gds_median', 0):.1f}%")
    gds_acc = ss.get('gds_acc', {})
    print(f"  @10%: {gds_acc.get(10, 0):.1f}%  @20%: {gds_acc.get(20, 0):.1f}%  @50%: {gds_acc.get(50, 0):.1f}%")

    print(f"\n--- Training ---")
    print(f"  Best val loss: {avg_loss:.4f}")

    # 7. Verification against known values
    print(f"\n--- Verification ---")
    v_ok = abs(mae_mv - exp['expected_v_mae']) < 0.5
    gm_ok = abs(ss.get('gm_log_mae', 0) - exp['expected_gm_mae']) < 0.005
    gds_ok = abs(ss.get('gds_log_mae', 0) - exp['expected_gds_mae']) < 0.005
    print(f"  V MAE: {mae_mv:.2f} vs expected {exp['expected_v_mae']:.2f} -> {'OK' if v_ok else 'MISMATCH!'}")
    print(f"  gm MAE: {ss.get('gm_log_mae', 0):.3f} vs expected {exp['expected_gm_mae']:.3f} -> {'OK' if gm_ok else 'MISMATCH!'}")
    print(f"  gds MAE: {ss.get('gds_log_mae', 0):.3f} vs expected {exp['expected_gds_mae']:.3f} -> {'OK' if gds_ok else 'MISMATCH!'}")

    if not (v_ok and gm_ok and gds_ok):
        print("  WARNING: Some metrics don't match! Check normalization.")

    return rel_metrics, ss


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    for exp in EXPERIMENTS:
        evaluate_experiment(exp, device)


if __name__ == '__main__':
    main()
