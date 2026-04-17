"""
Checkpoint save/load utilities for GNN models.

Handles saving complete configs and loading models with architecture auto-detection.
"""

from pathlib import Path
import torch
import yaml

from src.gnn import get_model, list_models


def create_model(
    node_feature_dim,
    hidden_dim=128,
    num_layers=15,
    dropout=0.0,
    genconv_num_layers=2,
    conv_type='genconv',
    conv_num_heads=4,
    mlp_expansion=2,
    mlp_depth=2,
    num_mlp_layers=3,
    jk_mode='cat',
    jk_attention=False,
    jk_learn_temperature=False,
    norm_type='layer',
    act_type='relu',
    skip_connection=True,
    predict_currents=False,
    voltage_head_config=None,
    current_head_config=None,
    use_virtual_node=False,
    use_attention_pooling=True,
    vn_learn_temperature=False,
    vn_gate_broadcast=False,
    vn_mode='default',
    vn_num_heads=4,
    vn_head_dim=32,
    gradient_checkpointing=False,
    device='cuda',
    model_type=None,
    derive_currents_from_voltage=False,
    mosfet_current_mlp_config=None,
    use_gnn_current_prediction=False,
    current_gnn_config=None,
    use_frozen_device_mlp=False,
    frozen_device_mlp_config=None,
    use_device_pooling_current=False,
    device_aggregation_config=None,
    intermediate_voltage_config=None,
    use_edge_features=False,
    input_dropout=0.0,
    use_refinement_pass=False,
    refinement_config=None,
    ac_head_config=None,
    ss_head_config=None,
    region_head_config=None,
    kcl_zspace_projection=False,
    kcl_blend_alpha=0.0,
    # Tower architecture options
    backbone_layers=6,
    state_tower_layers=2,
    sensitivity_tower_layers=2,
    backbone_jk_config=None,
    state_tower_jk_config=None,
    sensitivity_tower_jk_config=None,
    vov_head_config=None,
    vth_head_config=None,
    loop_attention_config=None,
    dc_gain_config=None,
    gm_id_head_config=None,
    subcircuit_dag_config=None,
    vgsvds_config=None,
):
    """
    Create a GNN model with the specified configuration.

    Args:
        node_feature_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate
        genconv_num_layers: Layers in each GENConv
        num_mlp_layers: MLP layers in head
        jk_mode: Jumping knowledge mode ('cat', 'last', 'max', 'sum')
        jk_attention: Use attention for JK aggregation
        jk_learn_temperature: Learn JK attention temperature
        norm_type: Normalization type ('layer' or 'batch')
        skip_connection: Concatenate input to head
        predict_currents: Enable current prediction head
        voltage_head_config: Config dict for voltage head
        current_head_config: Config dict for current head
        use_virtual_node: Enable virtual node (config option, not separate model)
        use_attention_pooling: Use attention pooling for virtual node
        vn_learn_temperature: Learn virtual node attention temperature
        gradient_checkpointing: Enable gradient checkpointing
        device: Device to place model on
        model_type: Model type from registry (default: 'deepgen')
        derive_currents_from_voltage: Derive MOSFET currents from predicted voltages
        mosfet_current_mlp_config: Config dict for MOSFET current MLP
        use_gnn_current_prediction: Use GNN layers for current prediction
        current_gnn_config: Config dict for current GNN backbone
        use_frozen_device_mlp: Use pre-trained frozen Device MLP for current prediction
        frozen_device_mlp_config: Config dict for frozen Device MLP
        use_refinement_pass: Enable two-pass refinement architecture
        refinement_config: Config dict for refinement GNN
        ss_head_config: Config dict for gm/gds prediction head

    Returns:
        model: Created model on specified device
        model_class: String name of model class used
    """
    # Default to 'deepgen' - VN is now a config option, not a separate model
    if model_type is None:
        model_type = 'deepgen'

    # All kwargs passed to model - VN is handled internally
    model_kwargs = dict(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        genconv_num_layers=genconv_num_layers,
        conv_type=conv_type,
        conv_num_heads=conv_num_heads,
        mlp_expansion=mlp_expansion,
        mlp_depth=mlp_depth,
        num_mlp_layers=num_mlp_layers,
        jk_mode=jk_mode,
        jk_attention=jk_attention,
        jk_learn_temperature=jk_learn_temperature,
        norm_type=norm_type,
        act_type=act_type,
        skip_connection=skip_connection,
        predict_currents=predict_currents,
        voltage_head_config=voltage_head_config or {},
        current_head_config=current_head_config or {},
        # Virtual node options (model handles these internally)
        use_virtual_node=use_virtual_node,
        use_attention_pooling=use_attention_pooling,
        vn_learn_temperature=vn_learn_temperature,
        vn_gate_broadcast=vn_gate_broadcast,
        vn_mode=vn_mode,
        vn_num_heads=vn_num_heads,
        vn_head_dim=vn_head_dim,
        gradient_checkpointing=gradient_checkpointing,
        # Voltage-derived current options
        derive_currents_from_voltage=derive_currents_from_voltage,
        mosfet_current_mlp_config=mosfet_current_mlp_config or {},
        # GNN-based current prediction options
        use_gnn_current_prediction=use_gnn_current_prediction,
        current_gnn_config=current_gnn_config or {},
        # Frozen Device MLP options
        use_frozen_device_mlp=use_frozen_device_mlp,
        frozen_device_mlp_config=frozen_device_mlp_config or {},
        # Device-level pooling current head
        use_device_pooling_current=use_device_pooling_current,
        # Device aggregation layer
        device_aggregation_config=device_aggregation_config or {},
        # Intermediate voltage prediction
        intermediate_voltage_config=intermediate_voltage_config or {},
        # Edge features
        use_edge_features=use_edge_features,
        input_dropout=input_dropout,
        # Refinement pass options
        use_refinement_pass=use_refinement_pass,
        refinement_config=refinement_config or {},
        # AC readout head
        ac_head_config=ac_head_config or {},
        # gm/gds prediction head
        ss_head_config=ss_head_config or {},
        # Region classification head
        region_head_config=region_head_config or {},
        # Z-space KCL projection
        kcl_zspace_projection=kcl_zspace_projection,
        kcl_blend_alpha=kcl_blend_alpha,
        # Tower architecture options
        backbone_layers=backbone_layers,
        state_tower_layers=state_tower_layers,
        sensitivity_tower_layers=sensitivity_tower_layers,
        backbone_jk_config=backbone_jk_config or {},
        state_tower_jk_config=state_tower_jk_config or {},
        sensitivity_tower_jk_config=sensitivity_tower_jk_config or {},
        # Vov prediction head
        vov_head_config=vov_head_config or {},
        # Vth prediction head
        vth_head_config=vth_head_config or {},
        # Loop attention config
        loop_attention_config=loop_attention_config or {},
        # DC gain prediction head
        dc_gain_config=dc_gain_config or {},
        # gm/Id auxiliary head
        gm_id_head_config=gm_id_head_config or {},
        # Subcircuit DAG
        subcircuit_dag_config=subcircuit_dag_config or {},
        # Vgs/Vds prediction head
        vgsvds_config=vgsvds_config or {},
    )

    # Create model using registry
    model = get_model(model_type, **model_kwargs)
    model_class = type(model).__name__

    model = model.to(device)
    return model, model_class


def create_model_from_args(args, input_dim, device='cuda'):
    """
    Create model from argparse args (convenience wrapper for training scripts).

    Args:
        args: Argument namespace with model config
        input_dim: Input feature dimension
        device: Device to place model on

    Returns:
        model: Created model
        model_class: String name of model class
    """
    # Get model_type from args (default: 'deepgen')
    model_type = getattr(args, 'model_type', 'deepgen')

    return create_model(
        node_feature_dim=input_dim,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        genconv_num_layers=getattr(args, 'genconv_num_layers', 2),
        conv_type=getattr(args, 'conv_type', 'genconv'),
        conv_num_heads=getattr(args, 'conv_num_heads', 4),
        mlp_expansion=getattr(args, 'mlp_expansion', 2),
        mlp_depth=getattr(args, 'mlp_depth', 2),
        num_mlp_layers=getattr(args, 'num_mlp_layers', 3),
        jk_mode=args.jk_mode,
        jk_attention=args.jk_attention,
        jk_learn_temperature=getattr(args, 'jk_learn_temperature', False),
        norm_type=getattr(args, 'norm_type', 'layer'),
        act_type=getattr(args, 'act_type', 'relu'),
        skip_connection=getattr(args, 'skip_connection', True),
        predict_currents=getattr(args, 'predict_currents', False),
        voltage_head_config=getattr(args, 'voltage_head_config', {}),
        current_head_config=getattr(args, 'current_head_config', {}),
        use_virtual_node=args.virtual_node,
        use_attention_pooling=getattr(args, 'use_attention_pooling', True),
        vn_learn_temperature=getattr(args, 'vn_learn_temperature', False),
        vn_gate_broadcast=getattr(args, 'vn_gate_broadcast', False),
        vn_mode=getattr(args, 'vn_mode', 'default'),
        vn_num_heads=getattr(args, 'vn_num_heads', 4),
        vn_head_dim=getattr(args, 'vn_head_dim', 32),
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
        device=device,
        model_type=model_type,
        derive_currents_from_voltage=getattr(args, 'derive_currents_from_voltage', False),
        mosfet_current_mlp_config=getattr(args, 'mosfet_current_mlp_config', {}),
        use_gnn_current_prediction=getattr(args, 'use_gnn_current_prediction', False),
        current_gnn_config=getattr(args, 'current_gnn_config', {}),
        use_frozen_device_mlp=getattr(args, 'use_frozen_device_mlp', False),
        frozen_device_mlp_config=getattr(args, 'frozen_device_mlp_config', {}),
        use_device_pooling_current=getattr(args, 'use_device_pooling_current', False),
        device_aggregation_config=getattr(args, 'device_aggregation_config', {}),
        intermediate_voltage_config=getattr(args, 'intermediate_voltage_config', {}),
        use_edge_features=getattr(args, 'use_edge_features', False),
        input_dropout=getattr(args, 'input_dropout', 0.0),
        use_refinement_pass=getattr(args, 'use_refinement_pass', False),
        refinement_config=getattr(args, 'refinement_config', {}),
        ac_head_config=getattr(args, 'ac_head_config', {}),
        ss_head_config=getattr(args, 'ss_head_config', {}),
        region_head_config=getattr(args, 'region_head_config', {}),
        kcl_zspace_projection=getattr(args, 'kcl_zspace_projection', False),
        kcl_blend_alpha=getattr(args, 'kcl_blend_alpha', 0.0),
        backbone_layers=getattr(args, 'backbone_layers', 6),
        state_tower_layers=getattr(args, 'state_tower_layers', 2),
        sensitivity_tower_layers=getattr(args, 'sensitivity_tower_layers', 2),
        backbone_jk_config=getattr(args, 'backbone_jk_config', {}),
        state_tower_jk_config=getattr(args, 'state_tower_jk_config', {}),
        sensitivity_tower_jk_config=getattr(args, 'sensitivity_tower_jk_config', {}),
        vov_head_config=getattr(args, 'vov_head_config', {}),
        vth_head_config=getattr(args, 'vth_head_config', {}),
        loop_attention_config=getattr(args, 'loop_attention_config', {}),
        dc_gain_config=getattr(args, 'dc_gain_config', {}),
        gm_id_head_config=getattr(args, 'gm_id_head_config', {}),
        subcircuit_dag_config=getattr(args, 'subcircuit_dag_config', {}),
        vgsvds_config=getattr(args, 'vgsvds_config', {}),
    )


def save_checkpoint(
    save_path,
    model_state_dict,
    epoch,
    val_loss,
    config,
    stats,
    save_yaml=True,
):
    """
    Save model checkpoint with full config.

    Args:
        save_path: Path to save checkpoint (.pt file)
        model_state_dict: Model state dict
        epoch: Best epoch number
        val_loss: Best validation loss
        config: Dict with all model and training config
        stats: Dict with normalization stats {'vdc': {...}, 'curr': {...}}
        save_yaml: Also save config.yaml alongside checkpoint
    """
    save_path = Path(save_path)

    torch.save({
        'model_state_dict': model_state_dict,
        'epoch': epoch,
        'val_loss': val_loss,
        'config': config,
        'stats': stats,
    }, save_path)

    if save_yaml:
        config_yaml = build_config_yaml(config, stats)
        config_path = save_path.parent / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False, sort_keys=False)
        return save_path, config_path

    return save_path, None


def build_config_yaml(config, stats):
    """Build YAML-friendly config dict from flat config."""
    return {
        'model': {
            'hidden_dim': config.get('hidden', config.get('hidden_dim', 128)),
            'num_layers': config.get('layers', config.get('num_layers', 15)),
            'dropout': config.get('dropout', 0.0),
            'genconv_num_layers': config.get('genconv_num_layers', 2),
            'use_virtual_node': config.get('virtual_node', config.get('use_virtual_node', False)),
            'predict_currents': config.get('predict_currents', False),
            'jk_mode': config.get('jk_mode', 'cat'),
            'jk_attention': config.get('jk_attention', False),
            'jk_learn_temperature': config.get('jk_learn_temperature', False),
            'norm_type': config.get('norm_type', 'layer'),
            'skip_connection': config.get('skip_connection', True),
            'use_attention_pooling': config.get('use_attention_pooling', True),
            'vn_learn_temperature': config.get('vn_learn_temperature', False),
            'voltage_head': config.get('voltage_head_config', config.get('voltage_head', {})),
            'current_head': config.get('current_head_config', config.get('current_head', {})),
            'derive_currents_from_voltage': config.get('derive_currents_from_voltage', False),
            'mosfet_current_mlp_config': config.get('mosfet_current_mlp_config', {}),
        },
        'training': {
            'dataset_path': config.get('dataset', ''),
            'learning_rate': config.get('learning_rate', config.get('lr', 0.003)),
            'weight_decay': config.get('weight_decay', 0.0),
            'gradient_clip': config.get('gradient_clip', 1.0),
            'scheduler': config.get('scheduler', 'plateau'),
            'warmup_epochs': config.get('warmup_epochs', config.get('warmup', 0)),
            'current_weight': config.get('current_weight', 1.0),
            'early_stopping_patience': config.get('early_stopping_patience', 80),
            'val_freq': config.get('val_freq', 1),
            'seed': config.get('seed', 42),
        },
        'normalization': {
            'vdc_mean': stats.get('vdc', {}).get('mean', 0.0),
            'vdc_std': stats.get('vdc', {}).get('std', 1.0),
            'current_mean': stats.get('current', stats.get('curr', {})).get('mean', 0.0),
            'current_std': stats.get('current', stats.get('curr', {})).get('std', 1.0),
        },
        'loss': {
            'loss_type': config.get('loss_type', 'mse'),
            'huber_delta': config.get('huber_delta', 1.0),
        },
    }


def load_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint with architecture auto-detection.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        model: Loaded model in eval mode
        config: Config dict
        stats: Normalization stats dict
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = ckpt.get('config', {})
    stats = ckpt.get('stats', {})
    state_dict = ckpt['model_state_dict']

    # Get model_type from config (default: 'deepgen')
    model_type = config.get('model_type', 'deepgen')

    # Detect virtual node from state_dict keys (for backward compatibility)
    # This handles old checkpoints that used separate deepgen_vn model
    has_vn_keys = any('vn_' in k or 'virtual_node' in k for k in state_dict.keys())

    # Get use_virtual_node from config, or infer from state_dict
    use_virtual_node = config.get('virtual_node', config.get('use_virtual_node', has_vn_keys))

    # Infer input dim from state_dict
    if 'input_proj.weight' in state_dict:
        input_dim = state_dict['input_proj.weight'].shape[1]
    elif 'input_linear.weight' in state_dict:
        input_dim = state_dict['input_linear.weight'].shape[1]
    else:
        input_dim = config.get('input_dim', 20)

    # Create model - VN is now a config option on 'deepgen'
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
        gradient_checkpointing=False,  # Not needed for inference
        device='cpu',  # Load to CPU first, then transfer after loading weights
        model_type=model_type,  # Use detected model_type from config
        derive_currents_from_voltage=config.get('derive_currents_from_voltage', False),
        mosfet_current_mlp_config=config.get('mosfet_current_mlp_config', {}),
        use_gnn_current_prediction=config.get('use_gnn_current_prediction', False),
        current_gnn_config=config.get('current_gnn_config', {}),
        use_frozen_device_mlp=config.get('use_frozen_device_mlp', False),
        frozen_device_mlp_config=config.get('frozen_device_mlp_config', {}),
        use_refinement_pass=config.get('use_refinement_pass', False),
        refinement_config=config.get('refinement_config', {}),
        ac_head_config=config.get('ac_head_config', {}),
        ss_head_config=config.get('ss_head_config', {}),
        region_head_config=config.get('region_head_config', {}),
        use_device_pooling_current=config.get('use_device_pooling_current',
            any('device_current_head.' in k for k in state_dict.keys())),
        device_aggregation_config=config.get('device_aggregation_config',
            {'enabled': True} if any('device_agg.' in k for k in state_dict.keys()) else {}),
        backbone_layers=config.get('backbone_layers', 6),
        state_tower_layers=config.get('state_tower_layers', 2),
        sensitivity_tower_layers=config.get('sensitivity_tower_layers', 2),
        backbone_jk_config=config.get('backbone_jk_config', {}),
        state_tower_jk_config=config.get('state_tower_jk_config', {}),
        sensitivity_tower_jk_config=config.get('sensitivity_tower_jk_config', {}),
        vov_head_config=config.get('vov_head_config', {}),
        vth_head_config=config.get('vth_head_config', {}),
        dc_gain_config=config.get('dc_gain_config', {}),
        act_type=config.get('act_type', 'gelu' if any('gelu' in k.lower() for k in state_dict.keys()) else 'relu'),
        loop_attention_config=config.get('loop_attention_config',
            {'enabled': True, 'num_heads': 4, 'head_dim': 32, 'apply_to': 'backbone', 'fusion': 'gate'}
            if any('loop_attn' in k for k in state_dict.keys()) else {}),
        gm_id_head_config=config.get('gm_id_head_config',
            {'enabled': True} if any('gm_id_head.' in k for k in state_dict.keys()) else {}),
        vn_mode=config.get('vn_config', {}).get('mode',
            'mha' if any('virtual_node.W_q' in k for k in state_dict.keys()) else 'default'),
        vn_num_heads=config.get('vn_config', {}).get('num_heads', 4),
        vn_head_dim=config.get('vn_config', {}).get('head_dim', 32),
        vn_gate_broadcast=config.get('vn_config', {}).get('gate_broadcast',
            any('virtual_node.gate_projs' in k for k in state_dict.keys())),
    )

    # Load weights and move to device
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Add detection info to config for reference
    config['_detected'] = {
        'input_dim': input_dim,
        'has_virtual_node': use_virtual_node,
        'model_class': model_class,
        'model_type': model_type,
    }

    return model, config, stats


def build_full_config(args, total_input_dim, predict_currents, voltage_head_config,
                      current_head_config, current_weight, loss_type, huber_delta,
                      val_freq, early_stopping_patience, dataset_path,
                      derive_currents_from_voltage=False, mosfet_current_mlp_config=None,
                      use_gnn_current_prediction=False, current_gnn_config=None,
                      use_frozen_device_mlp=False, frozen_device_mlp_config=None,
                      use_refinement_pass=False, refinement_config=None):
    """
    Build complete config dict from training args for saving.

    Args:
        args: Argument namespace from argparse
        total_input_dim: Total input feature dimension
        predict_currents: Whether model predicts currents
        voltage_head_config: Voltage head config dict
        current_head_config: Current head config dict
        current_weight: Weight for current loss
        loss_type: Loss type string
        huber_delta: Huber delta value
        val_freq: Validation frequency
        early_stopping_patience: Early stopping patience
        dataset_path: Path to dataset
        derive_currents_from_voltage: Derive MOSFET currents from predicted voltages
        mosfet_current_mlp_config: Config dict for MOSFET current MLP
        use_gnn_current_prediction: Use GNN layers for current prediction
        current_gnn_config: Config dict for current GNN backbone
        use_frozen_device_mlp: Use pre-trained frozen Device MLP for current prediction
        frozen_device_mlp_config: Config dict for frozen Device MLP
        use_refinement_pass: Enable two-pass refinement architecture
        refinement_config: Config dict for refinement GNN

    Returns:
        Dict with all config values
    """
    model_type = getattr(args, 'model_type', 'deepgen')

    return {
        # Model architecture
        'model_type': model_type,
        'hidden': args.hidden,
        'layers': args.layers,
        'dropout': args.dropout,
        'jk_mode': args.jk_mode,
        'jk_attention': args.jk_attention,
        'jk_learn_temperature': getattr(args, 'jk_learn_temperature', False),
        'genconv_num_layers': getattr(args, 'genconv_num_layers', 2),
        'num_mlp_layers': getattr(args, 'num_mlp_layers', 3),
        'norm_type': getattr(args, 'norm_type', 'layer'),
        'skip_connection': getattr(args, 'skip_connection', True),
        # Virtual node config (option on model, not separate model type)
        'virtual_node': args.virtual_node,
        'use_attention_pooling': getattr(args, 'use_attention_pooling', True),
        'vn_learn_temperature': getattr(args, 'vn_learn_temperature', False),
        'gradient_checkpointing': getattr(args, 'gradient_checkpointing', False),
        # Prediction heads
        'predict_currents': predict_currents,
        'voltage_head_config': voltage_head_config,
        'current_head_config': current_head_config,
        # Voltage-derived current config
        'derive_currents_from_voltage': derive_currents_from_voltage,
        'mosfet_current_mlp_config': mosfet_current_mlp_config or {},
        # GNN-based current prediction config
        'use_gnn_current_prediction': use_gnn_current_prediction,
        'current_gnn_config': current_gnn_config or {},
        # Frozen Device MLP config
        'use_frozen_device_mlp': use_frozen_device_mlp,
        'frozen_device_mlp_config': frozen_device_mlp_config or {},
        # Refinement pass config
        'use_refinement_pass': use_refinement_pass,
        'refinement_config': refinement_config or {},
        # AC readout head config
        'ac_head_config': getattr(args, 'ac_head_config', {}),
        # gm/gds prediction head config
        'ss_head_config': getattr(args, 'ss_head_config', {}),
        # Region classification head config
        'region_head_config': getattr(args, 'region_head_config', {}),
        # Tower architecture config
        'backbone_layers': getattr(args, 'backbone_layers', 6),
        'state_tower_layers': getattr(args, 'state_tower_layers', 2),
        'sensitivity_tower_layers': getattr(args, 'sensitivity_tower_layers', 2),
        'backbone_jk_config': getattr(args, 'backbone_jk_config', {}),
        'state_tower_jk_config': getattr(args, 'state_tower_jk_config', {}),
        'sensitivity_tower_jk_config': getattr(args, 'sensitivity_tower_jk_config', {}),
        # DC gain prediction head config
        'dc_gain_config': getattr(args, 'dc_gain_config', {}),
        # Activation type
        'act_type': getattr(args, 'act_type', 'relu'),
        # Loop attention config
        'loop_attention_config': getattr(args, 'loop_attention_config', {}),
        # gm/Id auxiliary head config
        'gm_id_head_config': getattr(args, 'gm_id_head_config', {}),
        # Virtual node config
        'vn_config': getattr(args, 'vn_config', {}),
        # Device pooling current
        'use_device_pooling_current': getattr(args, 'use_device_pooling_current', False),
        # Training params
        'learning_rate': args.lr,
        'weight_decay': getattr(args, 'weight_decay', 0.0),
        'gradient_clip': args.gradient_clip,
        'scheduler': args.scheduler,
        'warmup_epochs': args.warmup,
        'current_weight': current_weight,
        'loss_type': loss_type,
        'huber_delta': huber_delta,
        'val_freq': val_freq,
        'early_stopping_patience': early_stopping_patience,
        'seed': args.seed,
        # Data info
        'dataset': str(dataset_path),
        'input_dim': total_input_dim,
    }
