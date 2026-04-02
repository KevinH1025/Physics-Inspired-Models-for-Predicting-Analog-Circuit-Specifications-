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
    args['model_type'] = model_cfg.get('type', 'deepgen')
    args['hidden'] = model_cfg.get('hidden_dim', 128)
    args['layers'] = model_cfg.get('num_layers', 15)
    args['dropout'] = model_cfg.get('dropout', 0.0)
    args['genconv_num_layers'] = model_cfg.get('genconv_num_layers', 2)
    args['norm_type'] = model_cfg.get('norm_type', 'layer')
    args['act_type'] = model_cfg.get('act_type', 'relu')
    args['skip_connection'] = model_cfg.get('skip_connection', True)
    args['gradient_checkpointing'] = model_cfg.get('gradient_checkpointing', False)

    # Edge features
    args['use_edge_features'] = model_cfg.get('edge_features', False)
    args['input_dropout'] = model_cfg.get('input_dropout', 0.0)

    # Virtual node - support both nested and flat formats
    vn_cfg = model_cfg.get('virtual_node', {})
    args['virtual_node'] = vn_cfg.get('enabled', model_cfg.get('use_virtual_node', False))
    args['vn_learn_temperature'] = vn_cfg.get('learn_temperature', model_cfg.get('vn_learn_temperature', False))
    args['vn_gate_broadcast'] = vn_cfg.get('gate_broadcast', False)
    args['vn_mode'] = vn_cfg.get('mode', 'default')
    args['vn_num_heads'] = vn_cfg.get('num_heads', 4)
    args['vn_head_dim'] = vn_cfg.get('head_dim', 32)
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

    # Device-level pooling current head
    args['use_device_pooling_current'] = current_cfg.get('device_pooling', False)
    # Autograd SS: derive gm/gds via autograd through current head
    if current_cfg.get('autograd_ss', False):
        args['current_head_config']['autograd_ss'] = True

    # Device aggregation layer (device virtual node)
    device_agg_cfg = model_cfg.get('device_aggregation', {})
    args['device_aggregation_config'] = {
        'enabled': device_agg_cfg.get('enabled', False),
        'after_layer': device_agg_cfg.get('after_layer', None),
        'after_layers': device_agg_cfg.get('after_layers', None),
        'attention': device_agg_cfg.get('attention', False),
        'separate_mlps': device_agg_cfg.get('separate_mlps', False),
    }

    # Intermediate voltage prediction + feedback
    intermediate_v_cfg = model_cfg.get('intermediate_voltage', {})
    args['intermediate_voltage_config'] = {
        'enabled': intermediate_v_cfg.get('enabled', False),
        'after_layer': intermediate_v_cfg.get('after_layer', None),
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
        'dropout': ac_head_cfg.get('dropout', 0.0),
        'predict_ugbw': ac_head_cfg.get('predict_ugbw', True),
        'predict_pm': ac_head_cfg.get('predict_pm', True),
        'predict_am': ac_head_cfg.get('predict_am', False),
        'readout': ac_head_cfg.get('readout', 'vn'),  # 'vn', 'pool', or 'cross_attn'
        'num_heads': ac_head_cfg.get('num_heads', 8),
        'head_dim': ac_head_cfg.get('head_dim', 48),
    }

    # DC gain prediction head (graph-level, from sensitivity tower)
    dc_gain_cfg = heads_cfg.get('dc_gain', {})
    args['dc_gain_config'] = {
        'enabled': dc_gain_cfg.get('enabled', False),
        'mode': dc_gain_cfg.get('mode', 'cross_attn'),  # 'cross_attn' or 'physics'
        'num_heads': dc_gain_cfg.get('num_heads', 4),
        'head_dim': dc_gain_cfg.get('head_dim', 32),
        'dropout': dc_gain_cfg.get('dropout', 0.0),
        # Physics mode settings
        'hidden_dim': dc_gain_cfg.get('hidden_dim', 64),
        'num_layers': dc_gain_cfg.get('num_layers', 2),
        'detach_ss': dc_gain_cfg.get('detach_ss', False),
        'r_in': dc_gain_cfg.get('r_in', 50000.0),
        'r_f': dc_gain_cfg.get('r_f', 50000.0),
        'use_vn_context': dc_gain_cfg.get('use_vn_context', dc_gain_cfg.get('use_vn', False)),
        'vn_projection_dim': dc_gain_cfg.get('vn_projection_dim', 0),
        'use_device_context': dc_gain_cfg.get('use_device_context', False),
        'device_proj_dim': dc_gain_cfg.get('device_proj_dim', 32),
        'rout1_formula': dc_gain_cfg.get('rout1_formula', 'cascode'),
    }

    # gm/Id auxiliary prediction head
    gm_id_cfg = heads_cfg.get('gm_id', {})
    args['gm_id_head_config'] = {
        'enabled': gm_id_cfg.get('enabled', False),
        'hidden_dim': gm_id_cfg.get('hidden_dim', 128),
        'num_layers': gm_id_cfg.get('num_layers', 2),
        'as_feature': gm_id_cfg.get('as_feature', False),
    }

    # gm/gds (small-signal) prediction head
    ss_head_cfg = heads_cfg.get('ss', {})
    args['ss_head_config'] = {
        'enabled': ss_head_cfg.get('enabled', False),
        'hidden_dim': ss_head_cfg.get('hidden_dim', model_cfg.get('hidden_dim', 128)),
        'num_layers': ss_head_cfg.get('num_layers', 2),
        'dropout': ss_head_cfg.get('dropout', 0.0),
        'state_conditioned': ss_head_cfg.get('state_conditioned', False),
        'detach_state': ss_head_cfg.get('detach_state', True),
        'diff_features': ss_head_cfg.get('diff_features', False),
        'include_bulk': ss_head_cfg.get('include_bulk', False),
        'state_context': ss_head_cfg.get('state_context', False),
        'physics_pool': ss_head_cfg.get('physics_pool', False),
        'voltage_context': ss_head_cfg.get('voltage_context', False),
        'cross_attention': ss_head_cfg.get('cross_attention', False),
        'pairwise': ss_head_cfg.get('pairwise', False),
        'current_context': ss_head_cfg.get('current_context', False),
        'vov_context': ss_head_cfg.get('vov_context', False),
        'head_type': ss_head_cfg.get('head_type', 'mlp'),
        'num_attn_layers': ss_head_cfg.get('num_attn_layers', 1),
        'num_heads': ss_head_cfg.get('num_heads', 4),
        'shared_layers': ss_head_cfg.get('shared_layers', 0),
        'sensitivity_branch_layers': ss_head_cfg.get('sensitivity_branch_layers', 0),
        'source': ss_head_cfg.get('source', 'sensitivity_tower'),
        'region_context': ss_head_cfg.get('region_context', False),
        'wl_context': ss_head_cfg.get('wl_context', False),
        'vth_context': ss_head_cfg.get('vth_context', False),
        'mixture_of_experts': ss_head_cfg.get('mixture_of_experts', False),
        'expert_hidden_dim': ss_head_cfg.get('expert_hidden_dim', 256),
        'expert_num_layers': ss_head_cfg.get('expert_num_layers', 2),
        'predict_gm_id': ss_head_cfg.get('predict_gm_id', False),
    }
    # Differentiable I-V model config (nested under ss head)
    iv_cfg = ss_head_cfg.get('iv_model', {})
    args['ss_head_config']['iv_model'] = {
        'enabled': iv_cfg.get('enabled', False),
        'hidden_dim': iv_cfg.get('hidden_dim', 256),
        'num_layers': iv_cfg.get('num_layers', 3),
        'dropout': iv_cfg.get('dropout', 0.0),
        'detach_voltages': iv_cfg.get('detach_voltages', True),
        'detach_embeddings': iv_cfg.get('detach_embeddings', True),
        'residual': iv_cfg.get('residual', False),
        'embed_dim': iv_cfg.get('embed_dim', 64),
    }

    # Vov prediction head
    vov_head_cfg = heads_cfg.get('vov', {})
    args['vov_head_config'] = {
        'enabled': vov_head_cfg.get('enabled', False),
        'hidden_dim': vov_head_cfg.get('hidden_dim', 128),
    }

    # Vth prediction head
    vth_head_cfg = heads_cfg.get('vth', {})
    args['vth_head_config'] = {
        'enabled': vth_head_cfg.get('enabled', False),
        'hidden_dim': vth_head_cfg.get('hidden_dim', 128),
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

    # Tower architecture config (for tower_genconv)
    tower_cfg = model_cfg.get('tower', {})
    args['backbone_layers'] = tower_cfg.get('backbone_layers', 6)
    args['state_tower_layers'] = tower_cfg.get('state_tower_layers', 2)
    args['sensitivity_tower_layers'] = tower_cfg.get('sensitivity_tower_layers', 2)
    args['backbone_jk_config'] = tower_cfg.get('backbone_jk', {})
    args['state_tower_jk_config'] = tower_cfg.get('state_tower_jk', {})
    args['sensitivity_tower_jk_config'] = tower_cfg.get('sensitivity_tower_jk', {})

    # Loop attention config
    loop_attn_cfg = tower_cfg.get('loop_attention', {})
    args['loop_attention_config'] = {
        'enabled': loop_attn_cfg.get('enabled', False),
        'num_heads': loop_attn_cfg.get('num_heads', 4),
        'head_dim': loop_attn_cfg.get('head_dim', 32),
        'apply_to': loop_attn_cfg.get('apply_to', 'backbone'),
        'fusion': loop_attn_cfg.get('fusion', 'add'),
        'warmup_epochs': loop_attn_cfg.get('warmup_epochs', 0),
        'warmup_duration': loop_attn_cfg.get('warmup_duration', 0),
        'level': loop_attn_cfg.get('level', 'device'),
        'pool_mode': loop_attn_cfg.get('pool_mode', 'mean'),
    }

    # Loss - support both nested and flat formats
    loss_cfg = config.get('loss', {})
    train_cfg_preview = config.get('training', {})
    args['loss_type'] = loss_cfg.get('type', loss_cfg.get('loss_type', 'mse'))
    args['huber_delta'] = loss_cfg.get('huber_delta', 1.0)
    args['intermediate_v_weight'] = loss_cfg.get('intermediate_v_weight', 0.0)
    args['current_weight'] = loss_cfg.get('current_weight', train_cfg_preview.get('current_weight', 1.0))
    args['voltage_weight'] = loss_cfg.get('voltage_weight', 1.0)
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
    args['kcl_gt_filter'] = loss_cfg.get('kcl_gt_filter', 0.1)  # max GT relative violation for multi-term nets (0 = disabled)
    args['kcl_conservation'] = loss_cfg.get('kcl_conservation', False)  # structural KCL enforcement via projection
    args['kcl_skip_two_term'] = loss_cfg.get('kcl_skip_two_term', False)  # skip 2-term KCL loss (when enforced in architecture)
    args['kcl_only_two_term'] = loss_cfg.get('kcl_only_two_term', False)  # only compute 2-term KCL loss, skip 3+ term nets
    args['kcl_intermediate_weight'] = loss_cfg.get('kcl_intermediate_weight', 0.0)  # deep supervision: KCL on intermediate state tower layer
    # Soft z-score clipping for current targets (0 to disable)
    args['current_z_clip'] = loss_cfg.get('current_z_clip', 0.0)

    # Terminal voltage supervision (loss on terminal nodes instead of net nodes)
    args['use_terminal_voltage_loss'] = loss_cfg.get('use_terminal_voltage_loss', False)

    # Stage 2 node weighting (upweight critical output nodes)
    args['stage2_weight'] = loss_cfg.get('stage2_weight', 1.0)
    args['stage2_nodes'] = loss_cfg.get('stage2_nodes', None)
    args['node_weights'] = loss_cfg.get('node_weights', None)

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
    args['ac_pred_filter'] = ac_cfg.get('pred_filter', False)

    # DC gain prediction loss
    dc_gain_loss_cfg = loss_cfg.get('dc_gain_loss', {})
    args['dc_gain_loss_weight'] = dc_gain_loss_cfg.get('weight', 0.0)
    args['dc_gain_warmup_epochs'] = dc_gain_loss_cfg.get('warmup_epochs', 0)
    args['dc_gain_start_epoch'] = dc_gain_loss_cfg.get('start_epoch', 0)

    # gm/Id consistency loss
    gm_id_cfg = loss_cfg.get('gm_id_consistency', {})
    args['gm_id_consistency_weight'] = gm_id_cfg.get('weight', 0.0)

    # gm/Id auxiliary loss
    gm_id_aux_cfg = loss_cfg.get('gm_id_aux', {})
    args['gm_id_aux_weight'] = gm_id_aux_cfg.get('weight', 0.0)

    # Device consistency loss
    args['device_consistency_weight'] = loss_cfg.get('device_consistency_weight', 0.0)

    # Region classification loss
    region_cfg = loss_cfg.get('region_loss', {})
    args['region_loss_weight'] = region_cfg.get('weight', 0.0)
    args['region_loss_start_epoch'] = region_cfg.get('start_epoch', 0)

    # Supervised gm/gds prediction loss (separate weights)
    ss_cfg = loss_cfg.get('ss_loss', {})
    args['ss_gm_loss_weight'] = ss_cfg.get('gm_weight', ss_cfg.get('weight', 0.0))
    args['ss_gds_loss_weight'] = ss_cfg.get('gds_weight', ss_cfg.get('weight', 0.0))
    args['ss_loss_start_epoch'] = ss_cfg.get('start_epoch', 0)
    args['ss_loss_warmup_epochs'] = ss_cfg.get('warmup_epochs', 0)
    args['ss_huber_delta'] = ss_cfg.get('huber_delta', 0.0)  # 0 = MSE, >0 = Huber
    args['ss_per_region_norm'] = ss_cfg.get('per_region_norm', False)

    # IV model ID prediction loss (drain current from differentiable I-V model)
    iv_id_cfg = loss_cfg.get('iv_id_loss', {})
    args['iv_id_loss_weight'] = iv_id_cfg.get('weight', 0.0)

    # Vov prediction loss
    vov_loss_cfg = loss_cfg.get('vov_loss', {})
    args['vov_loss_weight'] = vov_loss_cfg.get('weight', 0.0)

    # Vth prediction loss
    vth_loss_cfg = loss_cfg.get('vth_loss', {})
    args['vth_loss_weight'] = vth_loss_cfg.get('weight', 0.0)

    # Uncertainty weighting (Kendall et al. 2018)
    args['use_uncertainty_weighting'] = loss_cfg.get('uncertainty_weighting', False)

    # Optimizer - support both nested and flat formats
    optim_cfg = config.get('optimizer', {})
    train_cfg_preview = config.get('training', {})
    args['lr'] = optim_cfg.get('lr', train_cfg_preview.get('learning_rate', 0.003))
    args['weight_decay'] = optim_cfg.get('weight_decay', train_cfg_preview.get('weight_decay', 0.0))
    args['adam_eps'] = optim_cfg.get('eps', 1e-8)

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

    # Output
    output_cfg = config.get('output', {})
    args['name'] = output_cfg.get('name', None)

    # Normalization
    norm_cfg = config.get('normalization', {})
    args['target_norm_type'] = norm_cfg.get('target_type', 'zscore')
    args['vdd'] = norm_cfg.get('vdd', 1.8)

    return args
