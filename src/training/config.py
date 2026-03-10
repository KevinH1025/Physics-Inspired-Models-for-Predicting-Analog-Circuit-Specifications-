"""
Configuration loading utilities.

Handles YAML config parsing for training scripts.
"""

import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse a YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse config dict into flat args-style dict.

    Args:
        config: Raw config dict from YAML

    Returns:
        Flat dict with all training parameters
    """
    args = {}

    # Data - support both nested and flat formats
    data_cfg = config.get('data', {})
    train_cfg_for_data = config.get('training', {})
    args['dataset'] = data_cfg.get('path', train_cfg_for_data.get('dataset_path'))
    args['use_fixed_topology'] = data_cfg.get('fixed_topology', False)
    args['use_prebatched'] = data_cfg.get('prebatched', True) and not args['use_fixed_topology']
    args['max_preload_variants'] = data_cfg.get('max_preload_variants', 10)
    args['preload_to_gpu'] = data_cfg.get('preload_to_gpu', True)

    # Model
    model_cfg = config.get('model', {})
    args['hidden'] = model_cfg.get('hidden_dim', 128)
    args['layers'] = model_cfg.get('num_layers', 15)
    args['dropout'] = model_cfg.get('dropout', 0.0)
    args['genconv_num_layers'] = model_cfg.get('genconv_num_layers', 2)
    args['norm_type'] = model_cfg.get('norm_type', 'layer')
    args['skip_connection'] = model_cfg.get('skip_connection', True)
    args['gradient_checkpointing'] = model_cfg.get('gradient_checkpointing', False)

    # Virtual node - support both nested and flat formats
    vn_cfg = model_cfg.get('virtual_node', {})
    args['virtual_node'] = vn_cfg.get('enabled', model_cfg.get('use_virtual_node', False))
    args['vn_learn_temperature'] = vn_cfg.get('learn_temperature', model_cfg.get('vn_learn_temperature', False))
    args['use_attention_pooling'] = vn_cfg.get('attention_pooling', model_cfg.get('use_attention_pooling', True))

    # JK - support both nested and flat formats
    jk_cfg = model_cfg.get('jk', {})
    args['jk_mode'] = jk_cfg.get('mode', model_cfg.get('jk_mode', 'last'))
    args['jk_attention'] = jk_cfg.get('attention', model_cfg.get('jk_attention', False))
    args['jk_learn_temperature'] = jk_cfg.get('learn_temperature', model_cfg.get('jk_learn_temperature', False))

    # Heads - support both nested and flat formats
    heads_cfg = model_cfg.get('heads', {})
    args['voltage_head_config'] = heads_cfg.get('voltage', model_cfg.get('voltage_head', {}))
    current_cfg = heads_cfg.get('current', {})
    args['predict_currents'] = current_cfg.get('enabled', model_cfg.get('predict_currents', False))
    args['current_head_config'] = {k: v for k, v in current_cfg.items() if k != 'enabled'}
    if not args['current_head_config']:
        args['current_head_config'] = model_cfg.get('current_head', {})

    # Voltage-derived currents - support both nested and flat formats
    derive_cfg = heads_cfg.get('derive_from_voltage', {})
    args['derive_currents_from_voltage'] = derive_cfg.get('enabled', model_cfg.get('derive_currents_from_voltage', False))
    mlp_cfg = model_cfg.get('mosfet_current_mlp_config', {})
    args['mosfet_current_mlp_config'] = {
        'hidden_dim': derive_cfg.get('hidden_dim', mlp_cfg.get('hidden_dim', 64)),
        'num_layers': derive_cfg.get('num_layers', mlp_cfg.get('num_layers', 2)),
        'dropout': derive_cfg.get('dropout', mlp_cfg.get('dropout', 0.0)),
        'use_wl_ratio': derive_cfg.get('use_wl_ratio', mlp_cfg.get('use_wl_ratio', True)),
    }

    # GNN-based current prediction - support both nested and flat formats
    current_gnn_cfg = heads_cfg.get('current_gnn', {})
    args['use_gnn_current_prediction'] = current_gnn_cfg.get('enabled', model_cfg.get('use_gnn_current_prediction', False))
    args['current_gnn_config'] = {
        'hidden_dim': current_gnn_cfg.get('hidden_dim', model_cfg.get('hidden_dim', 128)),
        'num_layers': current_gnn_cfg.get('num_layers', 3),
        'dropout': current_gnn_cfg.get('dropout', model_cfg.get('dropout', 0.0)),
        'genconv_num_layers': current_gnn_cfg.get('genconv_num_layers', model_cfg.get('genconv_num_layers', 2)),
        'head_layers': current_gnn_cfg.get('head_layers', 2),
    }

    # Frozen Device MLP for MOSFET current prediction
    frozen_mlp_cfg = heads_cfg.get('frozen_device_mlp', {})
    args['use_frozen_device_mlp'] = frozen_mlp_cfg.get('enabled', model_cfg.get('use_frozen_device_mlp', False))
    args['frozen_device_mlp_config'] = {
        'checkpoint': frozen_mlp_cfg.get('checkpoint', None),
        'hidden_dim': frozen_mlp_cfg.get('hidden_dim', 128),
        'num_layers': frozen_mlp_cfg.get('num_layers', 3),
        'use_polynomial_features': frozen_mlp_cfg.get('use_polynomial_features', False),
        'use_separate_heads': frozen_mlp_cfg.get('use_separate_heads', False),
        'use_residual': frozen_mlp_cfg.get('use_residual', False),
    }

    # AC readout head
    ac_head_cfg = heads_cfg.get('ac', {})
    args['ac_head_config'] = {
        'enabled': ac_head_cfg.get('enabled', False),
        'hidden_dim': ac_head_cfg.get('hidden_dim', 128),
        'num_layers': ac_head_cfg.get('num_layers', 3),
        'dropout': ac_head_cfg.get('dropout', 0.1),
        'predict_ugbw': ac_head_cfg.get('predict_ugbw', True),
        'predict_pm': ac_head_cfg.get('predict_pm', True),
        'predict_am': ac_head_cfg.get('predict_am', False),
        'readout': ac_head_cfg.get('readout', 'vn'),  # 'vn' or 'pool'
    }

    # gm/gds (small-signal) prediction head
    ss_head_cfg = heads_cfg.get('ss', {})
    args['ss_head_config'] = {
        'enabled': ss_head_cfg.get('enabled', False),
        'hidden_dim': ss_head_cfg.get('hidden_dim', model_cfg.get('hidden_dim', 128)),
        'num_layers': ss_head_cfg.get('num_layers', 2),
        'dropout': ss_head_cfg.get('dropout', 0.0),
    }

    # Region classification head
    region_head_cfg = heads_cfg.get('region', {})
    args['region_head_config'] = {
        'enabled': region_head_cfg.get('enabled', False),
        'hidden_dim': region_head_cfg.get('hidden_dim', model_cfg.get('hidden_dim', 128)),
        'num_layers': region_head_cfg.get('num_layers', 1),
        'dropout': region_head_cfg.get('dropout', 0.0),
    }

    # Refinement pass (two-pass architecture)
    refinement_cfg = heads_cfg.get('refinement', {})
    args['use_refinement_pass'] = refinement_cfg.get('enabled', model_cfg.get('use_refinement_pass', False))
    args['refinement_config'] = {
        'hidden_dim': refinement_cfg.get('hidden_dim', model_cfg.get('hidden_dim', 128)),
        'num_layers': refinement_cfg.get('num_layers', 3),
        'dropout': refinement_cfg.get('dropout', model_cfg.get('dropout', 0.0)),
        'genconv_num_layers': refinement_cfg.get('genconv_num_layers', model_cfg.get('genconv_num_layers', 2)),
        'head_layers': refinement_cfg.get('head_layers', 2),
    }

    # Z-space KCL projection (architectural enforcement for 2-term nets)
    args['kcl_zspace_projection'] = model_cfg.get('kcl_zspace_projection', False)
    args['kcl_blend_alpha'] = model_cfg.get('kcl_blend_alpha', 0.0)

    # Loss - support both nested and flat formats
    loss_cfg = config.get('loss', {})
    train_cfg_preview = config.get('training', {})
    args['loss_type'] = loss_cfg.get('type', loss_cfg.get('loss_type', 'mse'))
    args['huber_delta'] = loss_cfg.get('huber_delta', 1.0)
    args['current_weight'] = loss_cfg.get('current_weight', train_cfg_preview.get('current_weight', 1.0))
    args['kcl_weight'] = loss_cfg.get('kcl_weight', train_cfg_preview.get('kcl_weight', 0.0))
    # Current loss warmup - ramp current_weight from 0 to target over N epochs
    args['current_warmup_epochs'] = loss_cfg.get('current_warmup_epochs', train_cfg_preview.get('current_warmup_epochs', 0))
    # KCL loss warmup - ramp kcl_weight from 0 to target over N epochs
    args['kcl_warmup_epochs'] = loss_cfg.get('kcl_warmup_epochs', 0)
    # KCL start epoch - delay KCL until model is well-trained (0 = start after current warmup)
    args['kcl_start_epoch'] = loss_cfg.get('kcl_start_epoch', 0)
    # Minimum total current for KCL nets (filters noisy low-current nets)
    args['kcl_min_current'] = loss_cfg.get('kcl_min_current', 1e-9)  # 1nA default
    args['kcl_mode'] = loss_cfg.get('kcl_mode', 'logsumexp')  # logsumexp, z_diff, denorm
    args['kcl_exclusive'] = loss_cfg.get('kcl_exclusive', False)  # exclude KCL terminals from current MSE
    args['kcl_detach_backbone'] = loss_cfg.get('kcl_detach_backbone', False)  # stop KCL gradient to backbone
    args['kcl_violation_threshold'] = loss_cfg.get('kcl_violation_threshold', 0.0)  # min relative violation to penalize
    args['kcl_huber_delta'] = loss_cfg.get('kcl_huber_delta', 0.0)  # 0 = disabled (use MSE), >0 = Huber delta for 3-term KCL
    args['kcl_conservation'] = loss_cfg.get('kcl_conservation', False)  # structural KCL enforcement via projection
    args['kcl_skip_two_term'] = loss_cfg.get('kcl_skip_two_term', False)  # skip 2-term KCL loss (when enforced in architecture)
    args['kcl_only_two_term'] = loss_cfg.get('kcl_only_two_term', False)  # only compute 2-term KCL loss, skip 3+ term nets
    # Soft z-score clipping for current targets (0 to disable)
    args['current_z_clip'] = loss_cfg.get('current_z_clip', 0.0)

    # Terminal voltage supervision (loss on terminal nodes instead of net nodes)
    args['use_terminal_voltage_loss'] = loss_cfg.get('use_terminal_voltage_loss', False)

    # Stage 2 node weighting (upweight critical output nodes)
    args['stage2_weight'] = loss_cfg.get('stage2_weight', 1.0)
    args['stage2_nodes'] = loss_cfg.get('stage2_nodes', None)

    # Physics-based current constraints (diff pair, current mirrors)
    constraint_cfg = loss_cfg.get('current_constraints', {})
    args['constraint_weight'] = constraint_cfg.get('weight', 0.0)
    args['constraint_warmup_epochs'] = constraint_cfg.get('warmup_epochs', 0)
    args['constraint_start_epoch'] = constraint_cfg.get('start_epoch', 0)
    args['constraint_config'] = constraint_cfg.get('config', None)  # None = use default opamp constraints

    # gm self-consistency physics loss (gm = 2*I_D / Vov, saturation only)
    gm_phy_cfg = loss_cfg.get('gm_physics_loss', {})
    args['gm_physics_loss_weight'] = gm_phy_cfg.get('weight', 0.0)
    args['gm_physics_loss_start_epoch'] = gm_phy_cfg.get('start_epoch', 0)
    args['gm_physics_loss_warmup_epochs'] = gm_phy_cfg.get('warmup_epochs', 0)
    args['gm_physics_min_vov'] = gm_phy_cfg.get('min_vov', 0.0)
    args['gm_physics_use_clm'] = gm_phy_cfg.get('use_clm', False)
    args['gm_physics_use_gt_voltages'] = gm_phy_cfg.get('use_gt_voltages', True)

    # Triode physics regularizer loss (3 equations, training only)
    triode_phy_cfg = loss_cfg.get('triode_physics_loss', {})
    args['triode_physics_loss_weight'] = triode_phy_cfg.get('weight', 0.0)
    args['triode_physics_loss_start_epoch'] = triode_phy_cfg.get('start_epoch', 0)
    args['triode_physics_loss_warmup_epochs'] = triode_phy_cfg.get('warmup_epochs', 0)
    args['triode_physics_config'] = {
        'min_vov': triode_phy_cfg.get('min_vov', 0.0),
        'eq1_enabled': triode_phy_cfg.get('eq1_gm', {}).get('enabled', True),
        'eq1_weight': triode_phy_cfg.get('eq1_gm', {}).get('weight', 1.0),
        'eq2_enabled': triode_phy_cfg.get('eq2_gds', {}).get('enabled', True),
        'eq2_weight': triode_phy_cfg.get('eq2_gds', {}).get('weight', 1.0),
        'eq2_max_vds_vov': triode_phy_cfg.get('eq2_gds', {}).get('max_vds_vov', 1.0),
        'eq3_enabled': triode_phy_cfg.get('eq3_self', {}).get('enabled', True),
        'eq3_weight': triode_phy_cfg.get('eq3_self', {}).get('weight', 1.0),
    }

    # Cutoff/subthreshold physics loss
    cutoff_phy_cfg = loss_cfg.get('cutoff_physics_loss', {})
    args['cutoff_physics_loss_weight'] = cutoff_phy_cfg.get('weight', 0.0)
    args['cutoff_physics_loss_start_epoch'] = cutoff_phy_cfg.get('start_epoch', 0)
    args['cutoff_physics_loss_warmup_epochs'] = cutoff_phy_cfg.get('warmup_epochs', 0)
    args['cutoff_physics_n_nmos'] = cutoff_phy_cfg.get('n_nmos', 1.5)
    args['cutoff_physics_n_pmos'] = cutoff_phy_cfg.get('n_pmos', 2.0)

    # AC prediction loss
    ac_cfg = loss_cfg.get('ac_loss', {})
    args['ac_loss_weight'] = ac_cfg.get('weight', 0.0)
    args['ac_loss_start_epoch'] = ac_cfg.get('start_epoch', 0)
    args['ac_loss_warmup_epochs'] = ac_cfg.get('warmup_epochs', 0)

    # Region classification loss
    region_cfg = loss_cfg.get('region_loss', {})
    args['region_loss_weight'] = region_cfg.get('weight', 0.0)
    args['region_loss_start_epoch'] = region_cfg.get('start_epoch', 0)

    # Supervised gm/gds prediction loss
    ss_cfg = loss_cfg.get('ss_loss', {})
    args['ss_loss_weight'] = ss_cfg.get('weight', 0.0)
    args['ss_loss_start_epoch'] = ss_cfg.get('start_epoch', 0)
    args['ss_loss_warmup_epochs'] = ss_cfg.get('warmup_epochs', 0)

    # Optimizer - support both nested and flat formats
    optim_cfg = config.get('optimizer', {})
    train_cfg_preview = config.get('training', {})
    args['lr'] = optim_cfg.get('lr', train_cfg_preview.get('learning_rate', 0.003))
    args['weight_decay'] = optim_cfg.get('weight_decay', train_cfg_preview.get('weight_decay', 0.0))

    # Scheduler - support both nested and flat formats
    sched_cfg = config.get('scheduler', {})
    train_cfg = config.get('training', {})
    sched_type = sched_cfg.get('type', train_cfg.get('scheduler', 'plateau'))
    args['scheduler'] = {'plateau': 'plateau', 'cosine': 'cosine', 'poly': 'poly'}.get(sched_type, 'none')
    args['warmup'] = sched_cfg.get('warmup_epochs', train_cfg.get('warmup_epochs', 0))
    args['end_lr'] = float(sched_cfg.get('min_lr', 1e-6))
    args['plateau_factor'] = sched_cfg.get('factor', 0.5)
    args['plateau_patience'] = sched_cfg.get('patience', 10)

    # Training
    args['epochs'] = train_cfg.get('epochs', 700)
    args['batch_size'] = train_cfg.get('batch_size', 1024)
    args['gradient_clip'] = train_cfg.get('gradient_clip', 1.0)
    args['val_freq'] = train_cfg.get('val_freq', 1)
    args['early_stopping_patience'] = train_cfg.get('early_stopping_patience', 80)
    args['seed'] = train_cfg.get('seed', 42)

    # Normalization
    norm_cfg = config.get('normalization', {})
    args['target_norm_type'] = norm_cfg.get('target_type', 'zscore')
    args['vdd'] = norm_cfg.get('vdd', 1.8)

    return args
