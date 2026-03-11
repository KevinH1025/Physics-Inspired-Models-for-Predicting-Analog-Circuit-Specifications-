"""
Base GNN model for circuit node voltage and current prediction.

Supports optional Virtual Node for improved global information flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional

from ..registry import register_model
from ..components.layers import build_mlp, create_deepgcn_layer
from ..components.virtual_node import VirtualNode
from ..components.current_from_voltage import (
    MOSFETCurrentMLP,
    compute_mosfet_currents,
    compute_resistor_currents,
    compute_capacitor_currents,
    compute_isource_currents,
)
from ..components.current_gnn import CurrentGNNBackbone, propagate_voltages_to_terminals
from .base import BaseGNN

# Import for frozen device MLP (lazy to avoid circular imports if needed)
def _load_frozen_device_mlp(config: dict):
    """Load and freeze a pre-trained Device MLP."""
    from src.models.device_mlp import DeviceMLP
    checkpoint_path = config.get('checkpoint')
    if not checkpoint_path:
        raise ValueError("frozen_device_mlp_config must specify 'checkpoint' path")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model_state_dict', checkpoint)
    stats = checkpoint.get('stats', {})

    # Get model config from checkpoint or config file
    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 3)
    use_polynomial_features = config.get('use_polynomial_features', False)
    use_separate_heads = config.get('use_separate_heads', False)
    use_residual = config.get('use_residual', False)

    # Create model with matching architecture
    device_mlp = DeviceMLP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
        use_polynomial_features=use_polynomial_features,
        use_separate_heads=use_separate_heads,
        use_residual=use_residual,
    )
    device_mlp.load_state_dict(model_state)

    # Freeze parameters
    device_mlp.eval()
    for param in device_mlp.parameters():
        param.requires_grad = False

    return device_mlp, stats


@register_model("deepgen")
class DeepGENConv(BaseGNN):
    """
    Deep GNN for node-level prediction using GENConv layers.

    Architecture:
    - Input projection: Linear(node_features, hidden_dim)
    - N DeepGCNLayers with GENConv (res+ blocks, LayerNorm, ReLU)
    - Optional Virtual Node for global information aggregation
    - Jumping Knowledge aggregation (last/cat/max/sum with optional attention)
    - Prediction heads (voltage, optionally current)

    Args:
        node_feature_dim: Input feature dimension
        hidden_dim: Hidden dimension for all layers
        num_layers: Number of message passing layers
        dropout: Dropout probability
        genconv_num_layers: MLP layers within GENConv
        num_mlp_layers: Default layers in prediction head
        jk_mode: JK aggregation mode ('last', 'cat', 'max', 'sum')
        jk_attention: Use attention weighting for JK
        jk_learn_temperature: Learnable attention temperature
        norm_type: 'layer' or 'batch' normalization
        skip_connection: Concat input features to final representation
        predict_currents: Whether to predict node currents
        voltage_head_config: Config dict for voltage head
        current_head_config: Config dict for current head
        use_virtual_node: Enable virtual node for global information flow
        use_attention_pooling: Use attention pooling for VN aggregation
        vn_learn_temperature: Learnable VN attention temperature
        gradient_checkpointing: Use gradient checkpointing to save memory
        derive_currents_from_voltage: If True, derive MOSFET currents from predicted
            voltages using a small MLP instead of independent current head
        mosfet_current_mlp_config: Config dict for MOSFET current MLP (when deriving)
        use_gnn_current_prediction: If True, use GNN layers for current prediction
            instead of MLP. Provides context-aware current prediction.
        current_gnn_config: Config dict for current GNN backbone (when using GNN)
        use_frozen_device_mlp: If True, use pre-trained frozen Device MLP for MOSFET
            current prediction. The MLP is frozen but gradients flow through it.
        frozen_device_mlp_config: Config dict for frozen Device MLP (checkpoint path, etc.)
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 15,
        dropout: float = 0.0,
        genconv_num_layers: int = 2,
        num_mlp_layers: int = 3,
        jk_mode: str = 'last',
        jk_attention: bool = False,
        jk_learn_temperature: bool = False,
        norm_type: str = 'layer',
        skip_connection: bool = True,
        predict_currents: bool = False,
        voltage_head_config: dict = None,
        current_head_config: dict = None,
        # Virtual Node options
        use_virtual_node: bool = False,
        use_attention_pooling: bool = True,
        vn_learn_temperature: bool = False,
        gradient_checkpointing: bool = False,
        # Voltage-derived current options
        derive_currents_from_voltage: bool = False,
        mosfet_current_mlp_config: dict = None,
        # GNN-based current prediction options
        use_gnn_current_prediction: bool = False,
        current_gnn_config: dict = None,
        # Frozen Device MLP options
        use_frozen_device_mlp: bool = False,
        frozen_device_mlp_config: dict = None,
        # Device-level pooling current head
        use_device_pooling_current: bool = False,
        # Edge features
        use_edge_features: bool = False,
        edge_feature_dim: int = 6,
        # Input feature dropout
        input_dropout: float = 0.0,
        # Device aggregation layer (device virtual node)
        device_aggregation_config: dict = None,
        # Intermediate voltage prediction + feedback
        intermediate_voltage_config: dict = None,
        # Refinement pass options (two-pass architecture)
        use_refinement_pass: bool = False,
        refinement_config: dict = None,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.jk_mode = jk_mode
        self.jk_attention = jk_attention
        self.skip_connection = skip_connection
        self.node_feature_dim = node_feature_dim
        self.predict_currents = predict_currents
        self.norm_type = norm_type
        self.use_virtual_node = use_virtual_node
        self.gradient_checkpointing = gradient_checkpointing
        self.derive_currents_from_voltage = derive_currents_from_voltage
        self.use_gnn_current_prediction = use_gnn_current_prediction
        self.use_frozen_device_mlp = use_frozen_device_mlp
        self.use_device_pooling_current = use_device_pooling_current
        self.use_refinement_pass = use_refinement_pass
        self.kcl_zspace_projection = kwargs.get('kcl_zspace_projection', False)
        self.kcl_blend_alpha = kwargs.get('kcl_blend_alpha', 0.0)

        voltage_head_config = voltage_head_config or {}
        refinement_config = refinement_config or {}
        frozen_device_mlp_config = frozen_device_mlp_config or {}
        current_head_config = current_head_config or {}
        mosfet_current_mlp_config = mosfet_current_mlp_config or {}
        current_gnn_config = current_gnn_config or {}

        # Device aggregation layer (device virtual node)
        device_aggregation_config = device_aggregation_config or {}
        if device_aggregation_config.get('enabled', False):
            from src.gnn.components.device_aggregation import DeviceAggregationLayer
            self.device_agg = DeviceAggregationLayer(
                hidden_dim, norm_type=norm_type,
                attention=device_aggregation_config.get('attention', False),
                separate_mlps=device_aggregation_config.get('separate_mlps', False),
            )
            # Support both single layer and multiple layers
            after_layers = device_aggregation_config.get('after_layers', None)
            if after_layers is not None:
                self.device_agg_after_layers = set(after_layers)
            else:
                single = device_aggregation_config.get('after_layer', num_layers // 2)
                self.device_agg_after_layers = {single}
        else:
            self.device_agg = None
            self.device_agg_after_layers = set()

        # Intermediate voltage prediction + feedback
        intermediate_voltage_config = intermediate_voltage_config or {}
        if intermediate_voltage_config.get('enabled', False):
            self.intermediate_v_head = nn.Linear(hidden_dim, 1)
            self.intermediate_v_proj = nn.Linear(1, hidden_dim)
            self.intermediate_v_after_layer = intermediate_voltage_config.get('after_layer', num_layers // 2)
        else:
            self.intermediate_v_head = None

        # Input projection
        self.input_linear = nn.Linear(node_feature_dim, hidden_dim)

        # Message passing layers
        self.use_edge_features = use_edge_features
        self.input_dropout = input_dropout
        edge_dim = edge_feature_dim if use_edge_features else None
        self.layers = nn.ModuleList([
            create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        # Virtual Node (optional) - uses VirtualNode component
        if use_virtual_node:
            self.virtual_node = VirtualNode(
                hidden_dim=hidden_dim,
                use_attention_pooling=use_attention_pooling,
                learn_temperature=vn_learn_temperature,
            )
        else:
            self.virtual_node = None

        # JK aggregation
        self._init_jk_aggregation(jk_learn_temperature)

        # Prediction heads
        mlp_input_dim = hidden_dim + node_feature_dim if skip_connection else hidden_dim

        v_layers = voltage_head_config.get('num_layers', num_mlp_layers)
        v_hidden = voltage_head_config.get('hidden_dim', hidden_dim)
        v_dropout = voltage_head_config.get('dropout', 0.0)
        self.voltage_head = build_mlp(v_layers, mlp_input_dim, v_hidden, 1, norm_type, v_dropout)

        if predict_currents:
            if use_gnn_current_prediction:
                # GNN-based current prediction: additional GEN layers after voltage
                # Input: node embeddings + voltage feature
                gnn_input_dim = mlp_input_dim + 1  # +1 for voltage
                gnn_hidden = current_gnn_config.get('hidden_dim', hidden_dim)
                gnn_layers = current_gnn_config.get('num_layers', 3)
                gnn_dropout = current_gnn_config.get('dropout', dropout)
                gnn_genconv_layers = current_gnn_config.get('genconv_num_layers', genconv_num_layers)
                gnn_head_layers = current_gnn_config.get('head_layers', 2)

                self.current_gnn_backbone = CurrentGNNBackbone(
                    input_dim=gnn_input_dim,
                    hidden_dim=gnn_hidden,
                    num_layers=gnn_layers,
                    dropout=gnn_dropout,
                    genconv_num_layers=gnn_genconv_layers,
                    norm_type=norm_type,
                    current_head_layers=gnn_head_layers,
                )
                # Virtual node for current GNN (carries over backbone VN embedding)
                if self.virtual_node is not None:
                    self.current_vn = VirtualNode(
                        gnn_hidden,
                        use_attention_pooling=use_attention_pooling,
                        learn_temperature=vn_learn_temperature,
                    )
                else:
                    self.current_vn = None
                self.current_head = None
                self.mosfet_current_mlp = None
            elif derive_currents_from_voltage:
                # Use voltage-derived current prediction (MOSFET only for now)
                mlp_hidden = mosfet_current_mlp_config.get('hidden_dim', 64)
                mlp_layers = mosfet_current_mlp_config.get('num_layers', 2)
                mlp_dropout = mosfet_current_mlp_config.get('dropout', 0.0)
                use_wl_ratio = mosfet_current_mlp_config.get('use_wl_ratio', True)
                self.mosfet_current_mlp = MOSFETCurrentMLP(
                    hidden_dim=mlp_hidden,
                    num_layers=mlp_layers,
                    dropout=mlp_dropout,
                    use_wl_ratio=use_wl_ratio,
                )
                self.current_head = None  # Not used when deriving from voltage
                self.current_gnn_backbone = None
                self.frozen_device_mlp = None
            elif use_frozen_device_mlp:
                # Pre-trained frozen Device MLP for physics-informed current prediction
                # The MLP is frozen but gradients flow through it to voltage predictions
                self.frozen_device_mlp, self.frozen_device_mlp_stats = _load_frozen_device_mlp(
                    frozen_device_mlp_config
                )
                self.current_head = None
                self.mosfet_current_mlp = None
                self.current_gnn_backbone = None
                self.device_current_head = None
            elif use_device_pooling_current:
                # Device-level pooling: predict one current per device from terminal embeddings
                from src.gnn.components.device_current_head import DevicePoolingCurrentHead
                c_hidden = current_head_config.get('hidden_dim', 256)
                c_dropout = current_head_config.get('dropout', 0.0)
                self.device_current_head = DevicePoolingCurrentHead(
                    embed_dim=mlp_input_dim,
                    hidden_dim=c_hidden,
                    dropout=c_dropout,
                    norm_type=norm_type,
                )
                self.current_head = None
                self.mosfet_current_mlp = None
                self.current_gnn_backbone = None
                self.frozen_device_mlp = None
            else:
                # Traditional independent current head
                c_layers = current_head_config.get('num_layers', num_mlp_layers)
                c_hidden = current_head_config.get('hidden_dim', hidden_dim)
                c_dropout = current_head_config.get('dropout', 0.0)
                self.current_head = build_mlp(c_layers, mlp_input_dim, c_hidden, 1, norm_type, c_dropout)
                self.mosfet_current_mlp = None
                self.current_gnn_backbone = None
                self.frozen_device_mlp = None
                self.device_current_head = None

        # Refinement pass: uses V_pred₁ and I_pred₁ to predict ΔV
        # V_pred₂ = V_pred₁ + ΔV, then re-run current GNN for I_pred₂
        if use_refinement_pass and use_gnn_current_prediction:
            # Input: embeddings + V_pred₁ + I_pred₁
            ref_input_dim = mlp_input_dim + 2  # +1 for voltage, +1 for current
            ref_hidden = refinement_config.get('hidden_dim', hidden_dim)
            ref_layers = refinement_config.get('num_layers', 3)
            ref_dropout = refinement_config.get('dropout', dropout)
            ref_genconv_layers = refinement_config.get('genconv_num_layers', genconv_num_layers)
            ref_head_layers = refinement_config.get('head_layers', 2)

            # Refinement GNN backbone
            self.refinement_gnn_backbone = CurrentGNNBackbone(
                input_dim=ref_input_dim,
                hidden_dim=ref_hidden,
                num_layers=ref_layers,
                dropout=ref_dropout,
                genconv_num_layers=ref_genconv_layers,
                norm_type=norm_type,
                current_head_layers=ref_head_layers,
            )
            # ΔV head predicts voltage correction
            self.delta_v_head = build_mlp(
                ref_head_layers, ref_hidden, ref_hidden, 1, norm_type, ref_dropout
            )
        else:
            self.refinement_gnn_backbone = None
            self.delta_v_head = None

        # gm/gds prediction head (optional): predicts per-MOSFET small-signal params
        # from drain terminal node embeddings (after backbone + JK + skip)
        # Output: [log10(gm), log10(gds)] per MOSFET device
        ss_head_config = kwargs.get('ss_head_config', None)
        self.predict_ss = ss_head_config is not None and ss_head_config.get('enabled', False)
        if self.predict_ss:
            ss_hidden = ss_head_config.get('hidden_dim', hidden_dim)
            ss_layers = ss_head_config.get('num_layers', 2)
            ss_dropout = ss_head_config.get('dropout', 0.0)
            # Input: drain terminal embedding from backbone (mlp_input_dim)
            self.gm_head = build_mlp(ss_layers, mlp_input_dim, ss_hidden, 1, norm_type, ss_dropout)
            self.gds_head = build_mlp(ss_layers, mlp_input_dim, ss_hidden, 1, norm_type, ss_dropout)

        # Region head (optional): ordinal regression for operating region per MOSFET
        # Output: single scalar per node (cutoff=0, triode=1, saturation=2)
        region_head_config = kwargs.get('region_head_config', None)
        self.predict_region = region_head_config is not None and region_head_config.get('enabled', False)
        if self.predict_region:
            region_layers = region_head_config.get('num_layers', 1)
            region_hidden = region_head_config.get('hidden_dim', hidden_dim)
            region_dropout = region_head_config.get('dropout', 0.0)
            self.region_head = build_mlp(region_layers, mlp_input_dim, region_hidden, 1, norm_type, region_dropout)

        # AC readout head (optional): predicts graph-level AC quantities
        # Each component (UGBW, PM, AM) can be individually enabled
        # Readout modes:
        #   'vn'   - use virtual node embedding (128-dim)
        #   'pool' - mean+max pool over final node embeddings (mlp_input_dim * 2 = 288-dim)
        ac_head_config = kwargs.get('ac_head_config', None)
        self.predict_ac = ac_head_config is not None and ac_head_config.get('enabled', False)
        if self.predict_ac:
            # Build list of enabled AC components (order matters for output indexing)
            self.ac_components = []
            if ac_head_config.get('predict_ugbw', True):
                self.ac_components.append('ugbw')
            if ac_head_config.get('predict_pm', True):
                self.ac_components.append('pm')
            if ac_head_config.get('predict_am', False):
                self.ac_components.append('am')
            ac_output_dim = len(self.ac_components)
            if ac_output_dim == 0:
                self.predict_ac = False
            else:
                ac_hidden = ac_head_config.get('hidden_dim', 128)
                ac_layers = ac_head_config.get('num_layers', 3)
                ac_dropout = ac_head_config.get('dropout', 0.1)
                self.ac_readout = ac_head_config.get('readout', 'vn')
                self.ac_detach_vn = ac_head_config.get('detach_vn', False)
                if self.ac_readout == 'pool':
                    # mean+max pool over final node embeddings → 2 * mlp_input_dim
                    ac_input_dim = mlp_input_dim * 2
                else:
                    # VN embedding (hidden_dim) or current GNN VN
                    ac_input_dim = current_gnn_config.get('hidden_dim', hidden_dim) if use_gnn_current_prediction else hidden_dim
                self.ac_head = build_mlp(ac_layers, ac_input_dim, ac_hidden, ac_output_dim, norm_type, ac_dropout)

    def _init_jk_aggregation(self, learn_temperature: bool = False):
        """Initialize Jumping Knowledge aggregation components."""
        num_outputs = self.num_layers + 1

        if self.jk_attention:
            self.jk_attn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )
            if learn_temperature:
                self.jk_temperature = nn.Parameter(torch.tensor(1.0))
            else:
                self.register_buffer('jk_temperature', torch.tensor(1.0))

            if self.jk_mode == 'cat':
                self.jk_linear = nn.Linear(self.hidden_dim * num_outputs, self.hidden_dim)
            else:
                self.jk_linear = None
        elif self.jk_mode == 'cat':
            self.jk_linear = nn.Linear(self.hidden_dim * num_outputs, self.hidden_dim)
            self.jk_attn = None
        else:
            self.jk_linear = None
            self.jk_attn = None

    def _apply_jk(self, layer_outputs: list) -> torch.Tensor:
        """Apply Jumping Knowledge aggregation to layer outputs."""
        if self.jk_attention:
            layer_stack = torch.stack(layer_outputs, dim=0)
            attn_scores = self.jk_attn(layer_stack)
            temperature = self.jk_temperature.clamp(min=0.1)
            attn_weights = F.softmax(attn_scores / temperature, dim=0)

            if self.jk_linear is not None:
                weighted_stack = layer_stack * attn_weights
                x = weighted_stack.permute(1, 0, 2).reshape(layer_stack.size(1), -1)
                x = self.jk_linear(x)
            else:
                x = (layer_stack * attn_weights).sum(dim=0)

        elif self.jk_mode == 'cat':
            x = torch.cat(layer_outputs, dim=-1)
            x = self.jk_linear(x)

        elif self.jk_mode == 'max':
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]

        elif self.jk_mode == 'sum':
            x = torch.stack(layer_outputs, dim=0).sum(dim=0)

        else:  # 'last'
            x = layer_outputs[-1]

        return x

    def _get_num_graphs(self, data, batch: torch.Tensor) -> int:
        """Get number of graphs in batch."""
        if hasattr(data, 'num_graphs'):
            return data.num_graphs
        if hasattr(data, 'ptr'):
            return len(data.ptr) - 1
        return int(batch.max()) + 1

    def _message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        data=None,
    ) -> list:
        """Run message passing layers and collect outputs.

        If virtual node is enabled, broadcasts VN to nodes before each layer
        and aggregates from nodes after each layer.
        """
        layer_outputs = [x]

        # Initialize virtual node embedding if enabled
        vn_emb = None
        if self.virtual_node is not None:
            vn_emb = self.virtual_node.init_embedding(num_graphs)

        for i, layer in enumerate(self.layers):
            # Broadcast VN to nodes (if enabled)
            if self.virtual_node is not None:
                x = x + self.virtual_node.broadcast(vn_emb, batch)

            # Message passing (with optional gradient checkpointing)
            edge_attr = getattr(data, 'edge_attr', None) if self.use_edge_features and data is not None else None
            if self.gradient_checkpointing and self.training:
                if edge_attr is not None:
                    x = checkpoint(layer, x, edge_index, edge_attr, use_reentrant=False)
                else:
                    x = checkpoint(layer, x, edge_index, use_reentrant=False)
            else:
                if edge_attr is not None:
                    x = layer(x, edge_index, edge_attr)
                else:
                    x = layer(x, edge_index)

            # Update VN from nodes (if enabled)
            if self.virtual_node is not None:
                vn_emb, _ = self.virtual_node(x, vn_emb, batch, num_graphs)

            # Device aggregation: inject device context after specified layer
            if self.device_agg is not None and i in self.device_agg_after_layers and data is not None:
                x = self.device_agg(
                    x,
                    mosfet_info=getattr(data, 'mosfet_info', None),
                    resistor_info=getattr(data, 'resistor_info', None),
                    capacitor_info=getattr(data, 'capacitor_info', None),
                    vsource_info=getattr(data, 'vsource_info', None),
                    isource_info=getattr(data, 'isource_info', None),
                    ptr=getattr(data, 'ptr', None),
                    mosfet_ptr=getattr(data, 'mosfet_ptr', None),
                    resistor_ptr=getattr(data, 'resistor_ptr', None),
                    capacitor_ptr=getattr(data, 'capacitor_ptr', None),
                    isource_ptr=getattr(data, 'isource_ptr', None),
                )

            # Intermediate voltage prediction + feedback
            if self.intermediate_v_head is not None and i == self.intermediate_v_after_layer:
                v_mid = self.intermediate_v_head(x).squeeze(-1)  # [num_nodes]
                self._intermediate_v_pred = v_mid
                x = x + self.intermediate_v_proj(v_mid.unsqueeze(-1))

            layer_outputs.append(x)

        return layer_outputs, vn_emb

    def _get_final_representation(self, layer_outputs: list, x_in: torch.Tensor) -> torch.Tensor:
        """Get final node representation after JK and optional skip connection."""
        x = self._apply_jk(layer_outputs)
        x = self.layers[0].act(self.layers[0].norm(x))

        if self.skip_connection:
            x = torch.cat([x, x_in], dim=-1)

        return x

    def _enforce_per_device_current(self, node_currents: torch.Tensor, data) -> torch.Tensor:
        """Enforce one current per device by averaging terminal predictions.

        For MOSFETs: average drain and source predictions.
        For 2-terminal devices (R, C, V, I): average p and n predictions.
        """
        out = node_currents.clone()

        # MOSFETs: average drain + source
        if hasattr(data, 'mosfet_info') and data.mosfet_info is not None and data.mosfet_info.numel() > 0:
            drain_idx = data.mosfet_info[:, 1].long()
            source_idx = data.mosfet_info[:, 2].long()
            avg = (out[drain_idx] + out[source_idx]) / 2
            out[drain_idx] = avg
            out[source_idx] = avg

        # 2-terminal devices: average p + n
        for attr in ('resistor_info', 'capacitor_info', 'vsource_info'):
            info = getattr(data, attr, None)
            if info is not None and info.numel() > 0:
                p_idx = info[:, 0].long()
                n_idx = info[:, 1].long()
                avg = (out[p_idx] + out[n_idx]) / 2
                out[p_idx] = avg
                out[n_idx] = avg

        if hasattr(data, 'isource_info') and data.isource_info is not None and data.isource_info.numel() > 0:
            p_idx = data.isource_info[:, 0].long()
            n_idx = data.isource_info[:, 1].long()
            avg = (out[p_idx] + out[n_idx]) / 2
            out[p_idx] = avg
            out[n_idx] = avg

        return out

    def _apply_kcl_zspace_projection(self, currents: torch.Tensor, data) -> torch.Tensor:
        """
        Enforce KCL in z-space for 2-term and 3-term nets.
        2-term: z_avg = (z_A + z_B) / 2, both set to z_avg. Gradient = 0.5.
        3-term: predict majority pair independently, derive minority from KCL
        via logaddexp. Gradient = |I_i|/sum(|I|), bounded [0,1].
        """
        ptr = getattr(data, 'ptr', None)
        num_terminals = getattr(data, 'num_terminals', None)
        train_mask = getattr(data, 'train_mask', None)
        terminal_current_sign = getattr(data, 'terminal_current_sign', None)
        kcl_include_mask = getattr(data, 'kcl_include_mask', None)

        if ptr is None or train_mask is None or terminal_current_sign is None:
            return currents

        num_nodes = train_mask.size(0)
        graph_sizes = ptr[1:] - ptr[:-1]
        batch_idx = torch.repeat_interleave(
            torch.arange(len(graph_sizes), device=currents.device), graph_sizes
        )
        local_idx = torch.arange(num_nodes, device=currents.device) - ptr[batch_idx]

        if isinstance(num_terminals, int):
            terminal_mask = local_idx < num_terminals
        else:
            terminal_mask = local_idx < num_terminals[batch_idx]

        internal_net_mask = train_mask & ~terminal_mask

        src, dst = data.edge_index
        valid_edges = terminal_mask[src] & internal_net_mask[dst]
        if not valid_edges.any():
            return currents

        valid_src, valid_dst = src[valid_edges], dst[valid_edges]
        if kcl_include_mask is not None:
            keep = kcl_include_mask[valid_src]
            valid_src, valid_dst = valid_src[keep], valid_dst[keep]

        if valid_src.numel() == 0:
            return currents

        # Ensure float32 for scatter operations (model may output bfloat16 under AMP)
        currents = currents.float()

        # Count terminals per net and check for both signs (2-term net criterion)
        terms = torch.zeros(num_nodes, device=currents.device, dtype=torch.long)
        terms.scatter_add_(0, valid_dst, torch.ones_like(valid_dst))

        sign_at_src = terminal_current_sign[valid_src]
        pos = torch.zeros(num_nodes, device=currents.device, dtype=torch.long)
        neg = torch.zeros(num_nodes, device=currents.device, dtype=torch.long)
        pos.scatter_add_(0, valid_dst, (sign_at_src > 0).long())
        neg.scatter_add_(0, valid_dst, (sign_at_src < 0).long())

        projected = currents.clone()

        # 2-term nets: average both terminals
        two_term_mask = (terms == 2) & (pos > 0) & (neg > 0) & internal_net_mask
        if two_term_mask.any():
            edge_valid = two_term_mask[valid_dst]
            t_src = valid_src[edge_valid]
            t_dst = valid_dst[edge_valid]

            z_sum = torch.zeros(num_nodes, device=currents.device)
            z_sum.scatter_add_(0, t_dst, currents[t_src])
            z_avg = z_sum / 2.0

            projected[t_src] = z_avg[t_dst]

        # 3-term nets: blend minority terminal with KCL-derived value
        if self.kcl_blend_alpha > 0:
            three_term_mask = (terms == 3) & (pos > 0) & (neg > 0) & internal_net_mask
            if three_term_mask.any():
                edge_valid = three_term_mask[valid_dst]
                t_src = valid_src[edge_valid]
                t_dst = valid_dst[edge_valid]
                t_sign = sign_at_src[edge_valid]
                t_z = currents[t_src]  # read from original, not projected

                # Identify minority sign per net (the sign with count 1)
                net_pos = pos[t_dst]
                net_neg = neg[t_dst]
                # minority_is_pos=True means positive terminals are minority (1 pos, 2 neg)
                minority_is_pos = (net_pos == 1)

                is_minority = ((t_sign > 0) & minority_is_pos) | ((t_sign < 0) & ~minority_is_pos)
                is_majority = ~is_minority

                # Compute logaddexp of majority pair per net to derive minority value
                # logaddexp(a, b) = log10(10^a + 10^b) in z-space
                # First, gather majority z-values per net
                maj_z = t_z.clone()
                maj_z[is_minority] = float('-inf')  # exclude minority from sum

                # scatter logsumexp over majority terminals per net
                unique_nets, inv = t_dst.unique(return_inverse=True)
                n_nets = unique_nets.size(0)

                # Convert to natural log for logsumexp, then back
                ln10 = 2.302585093
                current_std = getattr(data, 'current_std', None)
                current_mean = getattr(data, 'current_mean', None)
                if current_std is None or current_mean is None:
                    return projected  # can't do 3-term without normalization stats

                # Convert z-space to log10|I|: log10|I| = z * std + mean
                maj_log10 = maj_z * current_std + current_mean
                # Convert to natural log for torch.scatter with logsumexp
                maj_ln = maj_log10 * ln10

                # Scatter logsumexp: for each net, compute log(sum(exp(ln_vals))) over majority
                max_per_net = torch.full((n_nets,), float('-inf'), device=currents.device)
                max_per_net.scatter_reduce_(0, inv, maj_ln, reduce='amax', include_self=False)

                # Only majority contribute
                maj_ln_shifted = maj_ln - max_per_net[inv]
                maj_exp = torch.where(is_majority, torch.exp(maj_ln_shifted), torch.zeros_like(maj_ln_shifted))

                sum_exp = torch.zeros(n_nets, device=currents.device)
                sum_exp.scatter_add_(0, inv, maj_exp)

                log_sum_ln = max_per_net + torch.log(sum_exp + 1e-30)
                # Convert back: log10|I_derived| = log_sum_ln / ln10
                derived_log10 = log_sum_ln / ln10
                # Convert to z-space: z_derived = (log10|I| - mean) / std
                z_derived = (derived_log10 - current_mean) / current_std

                # Blend minority terminals
                alpha = self.kcl_blend_alpha
                minority_mask_global = is_minority
                minority_src = t_src[minority_mask_global]
                minority_inv = inv[minority_mask_global]
                z_pred = currents[minority_src]
                z_kcl = z_derived[minority_inv]
                projected[minority_src] = alpha * z_kcl + (1 - alpha) * z_pred

        return projected

    def forward(self, data) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, and optional type_tens/net_type/batch

        Returns:
            Dict with 'node_voltages' and optionally 'node_currents'
        """
        x_in = self._get_input_features(data)
        if self.input_dropout > 0:
            x_in = F.dropout(x_in, p=self.input_dropout, training=self.training)
        x = self.input_linear(x_in)

        # Get batch info for virtual node (if enabled)
        batch = None
        num_graphs = None
        if self.virtual_node is not None:
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(
                x_in.size(0), dtype=torch.long, device=x_in.device
            )
            num_graphs = self._get_num_graphs(data, batch)

        layer_outputs, vn_emb = self._message_passing(x, data.edge_index, batch, num_graphs, data=data)
        x = self._get_final_representation(layer_outputs, x_in)

        result = {'node_voltages': self.voltage_head(x).squeeze(-1)}
        result['node_embeddings'] = x  # Pre-head embeddings for detached KCL

        # Include intermediate voltage prediction for auxiliary loss
        if self.intermediate_v_head is not None and hasattr(self, '_intermediate_v_pred'):
            result['intermediate_voltages'] = self._intermediate_v_pred

        current_vn_emb = None  # Will be set by current GNN if VN is enabled

        if self.predict_currents:
            if self.use_gnn_current_prediction:
                # GNN-based current prediction: use additional GEN layers
                # that can see circuit context (neighbors, topology patterns)
                pred_voltages = result['node_voltages']
                ptr = getattr(data, 'ptr', None)
                num_terminals = getattr(data, 'num_terminals', None)

                # Propagate voltages from NET nodes to TERMINAL nodes
                # so the current GNN has voltage info at all nodes
                node_voltages = propagate_voltages_to_terminals(
                    pred_voltages=pred_voltages,
                    edge_index=data.edge_index,
                    num_terminals=num_terminals,
                    ptr=ptr,
                )

                # Augment node embeddings with voltage feature
                x_augmented = torch.cat([x, node_voltages.unsqueeze(-1)], dim=-1)

                # Run current GNN backbone (no detach - end-to-end training)
                # Pass VN embedding from backbone for global context continuity
                current_pred, current_vn_emb = self.current_gnn_backbone(
                    x_augmented, data.edge_index,
                    virtual_node=self.current_vn,
                    vn_emb=vn_emb,
                    batch=batch,
                    num_graphs=num_graphs,
                )
                result['node_currents'] = current_pred.squeeze(-1)

                # === PASS 2: Refinement ===
                # Use initial current predictions to refine voltages
                if self.use_refinement_pass and self.refinement_gnn_backbone is not None:
                    # Store pass 1 predictions
                    v_pred_1 = result['node_voltages']
                    i_pred_1 = result['node_currents']

                    # Propagate currents from TERMINAL nodes to all nodes
                    # (similar to voltage propagation)
                    node_currents_all = propagate_voltages_to_terminals(
                        pred_voltages=i_pred_1,  # reusing function for currents
                        edge_index=data.edge_index,
                        num_terminals=num_terminals,
                        ptr=ptr,
                    )

                    # Augment embeddings with V_pred₁ and I_pred₁
                    x_refinement = torch.cat([
                        x,
                        node_voltages.unsqueeze(-1),
                        node_currents_all.unsqueeze(-1),
                    ], dim=-1)

                    # Run refinement GNN to get intermediate features
                    ref_features = self.refinement_gnn_backbone.get_embeddings(
                        x_refinement, data.edge_index
                    )

                    # Predict ΔV correction
                    delta_v = self.delta_v_head(ref_features).squeeze(-1)

                    # V_pred₂ = V_pred₁ + ΔV (only for NET nodes, not terminals)
                    v_pred_2 = v_pred_1 + delta_v

                    # Propagate refined voltages
                    node_voltages_2 = propagate_voltages_to_terminals(
                        pred_voltages=v_pred_2,
                        edge_index=data.edge_index,
                        num_terminals=num_terminals,
                        ptr=ptr,
                    )

                    # Re-augment embeddings with refined voltage
                    x_augmented_2 = torch.cat([x, node_voltages_2.unsqueeze(-1)], dim=-1)

                    # Re-run current GNN (same weights) for I_pred₂
                    current_pred_2, current_vn_emb = self.current_gnn_backbone(
                        x_augmented_2, data.edge_index,
                        virtual_node=self.current_vn,
                        vn_emb=vn_emb,  # re-init from backbone VN (not pass 1 current VN)
                        batch=batch,
                        num_graphs=num_graphs,
                    )
                    i_pred_2 = current_pred_2.squeeze(-1)

                    # Store refined predictions (overwrite pass 1)
                    result['node_voltages'] = v_pred_2
                    result['node_currents'] = i_pred_2

                    # Keep pass 1 for potential auxiliary loss
                    result['node_voltages_pass1'] = v_pred_1
                    result['node_currents_pass1'] = i_pred_1
                    result['delta_v'] = delta_v

            elif self.derive_currents_from_voltage:
                # Hybrid physics-informed current prediction:
                # - MOSFETs: MLP learns I_ds = f(Vgs, Vds, W/L, is_nmos)
                # - Resistors: I = (V_p - V_n) / R (Ohm's law)
                # - Capacitors: I = 0 (DC analysis - open circuit)
                # - I-sources: I = i_ref (known constant)
                # - V-sources: Excluded (not predictable from voltage)

                pred_voltages = result['node_voltages']

                # Phase 1: Detach voltages so current loss doesn't interfere with voltage learning
                # Phase 2: Don't detach - voltage is frozen anyway, gradients flow to current MLP
                phase2_mode = getattr(self, 'phase2_mode', False)
                if phase2_mode:
                    # Phase 2: voltage frozen, allow gradients to current MLP
                    pred_voltages_for_current = pred_voltages
                else:
                    # Phase 1: detach to isolate voltage learning from current loss
                    pred_voltages_for_current = pred_voltages.detach()
                use_terminal_indices = False  # Use net indices for predicted voltages
                num_nodes = len(pred_voltages)
                device = pred_voltages.device
                dtype = pred_voltages.dtype
                ptr = getattr(data, 'ptr', None)

                # Get normalization stats (for physics-based current computation)
                # These should be set on data object by the dataloader
                voltage_mean = getattr(data, 'voltage_mean', 0.0)
                voltage_std = getattr(data, 'voltage_std', 1.0)
                current_mean = getattr(data, 'current_mean', 0.0)
                current_std = getattr(data, 'current_std', 1.0)

                # Initialize combined currents and mask for voltage-derived predictions
                combined_currents = torch.zeros(num_nodes, device=device, dtype=dtype)
                voltage_derived_mask = torch.zeros(num_nodes, device=device, dtype=torch.bool)

                # 1. MOSFET currents via MLP
                mosfet_info = getattr(data, 'mosfet_info', None)
                num_terminals = getattr(data, 'num_terminals', None)
                if mosfet_info is not None and num_terminals is not None and len(mosfet_info) > 0:
                    mosfet_currents = compute_mosfet_currents(
                        pred_voltages=pred_voltages_for_current,
                        mosfet_info=mosfet_info,
                        num_terminals=num_terminals,
                        current_mlp=self.mosfet_current_mlp,
                        ptr=ptr,
                        terminal_features=data.x,
                        num_nodes=num_nodes,
                        use_terminal_indices=use_terminal_indices,
                        voltage_mean=voltage_mean,
                        voltage_std=voltage_std,
                        mosfet_ptr=getattr(data, 'mosfet_ptr', None),
                    )
                    if mosfet_currents is not None:
                        combined_currents = combined_currents + mosfet_currents
                        # Mark MOSFET drain/source terminals in mask
                        voltage_derived_mask = voltage_derived_mask | (mosfet_currents != 0)

                # 2. Resistor currents via Ohm's law
                resistor_info = getattr(data, 'resistor_info', None)
                if resistor_info is not None and len(resistor_info) > 0:
                    resistor_currents = compute_resistor_currents(
                        pred_voltages=pred_voltages_for_current,
                        resistor_info=resistor_info,
                        num_nodes=num_nodes,
                        ptr=ptr,
                        voltage_mean=voltage_mean,
                        voltage_std=voltage_std,
                        current_mean=current_mean,
                        current_std=current_std,
                        resistor_ptr=getattr(data, 'resistor_ptr', None),
                    )
                    combined_currents = combined_currents + resistor_currents
                    voltage_derived_mask = voltage_derived_mask | (resistor_currents != 0)

                # 3. Capacitor currents (zeros for DC analysis)
                capacitor_info = getattr(data, 'capacitor_info', None)
                if capacitor_info is not None and len(capacitor_info) > 0:
                    capacitor_currents = compute_capacitor_currents(
                        capacitor_info=capacitor_info,
                        num_nodes=num_nodes,
                        device=device,
                        dtype=dtype,
                    )
                    combined_currents = combined_currents + capacitor_currents

                # 4. I-source currents (known constants)
                isource_info = getattr(data, 'isource_info', None)
                if isource_info is not None and len(isource_info) > 0:
                    isource_currents = compute_isource_currents(
                        isource_info=isource_info,
                        num_nodes=num_nodes,
                        ptr=ptr,
                        current_mean=current_mean,
                        current_std=current_std,
                        isource_ptr=getattr(data, 'isource_ptr', None),
                    )
                    combined_currents = combined_currents + isource_currents
                    voltage_derived_mask = voltage_derived_mask | (isource_currents != 0)

                result['node_currents'] = combined_currents
                result['voltage_derived_current_mask'] = voltage_derived_mask

            elif self.use_frozen_device_mlp:
                # Pre-trained frozen Device MLP for physics-informed current prediction
                # The MLP is frozen but gradients flow through it to voltage predictions
                # This teaches the GNN: "wrong voltages → wrong currents"

                pred_voltages = result['node_voltages']
                # NO detach - gradients flow through frozen MLP to voltage predictions
                num_nodes = len(pred_voltages)
                device = pred_voltages.device
                dtype = pred_voltages.dtype
                ptr = getattr(data, 'ptr', None)

                # Get normalization stats
                voltage_mean = getattr(data, 'voltage_mean', 0.0)
                voltage_std = getattr(data, 'voltage_std', 1.0)
                current_mean = getattr(data, 'current_mean', 0.0)
                current_std = getattr(data, 'current_std', 1.0)

                # Get Device MLP training stats for proper denormalization
                mlp_stats = getattr(self, 'frozen_device_mlp_stats', {})
                mlp_Vgs_mean = mlp_stats.get('Vgs_mean', 0.0)
                mlp_Vgs_std = mlp_stats.get('Vgs_std', 1.0)
                mlp_Vds_mean = mlp_stats.get('Vds_mean', 0.0)
                mlp_Vds_std = mlp_stats.get('Vds_std', 1.0)
                mlp_log_W_mean = mlp_stats.get('log_W_mean', -5.0)
                mlp_log_W_std = mlp_stats.get('log_W_std', 0.5)
                mlp_log_L_mean = mlp_stats.get('log_L_mean', -7.0)
                mlp_log_L_std = mlp_stats.get('log_L_std', 0.3)
                mlp_log_I_mean = mlp_stats.get('log_I_mean', -5.0)
                mlp_log_I_std = mlp_stats.get('log_I_std', 1.0)

                # Initialize combined currents
                combined_currents = torch.zeros(num_nodes, device=device, dtype=dtype)
                voltage_derived_mask = torch.zeros(num_nodes, device=device, dtype=torch.bool)

                # 1. MOSFET currents via frozen Device MLP
                mosfet_info = getattr(data, 'mosfet_info', None)
                num_terminals = getattr(data, 'num_terminals', None)
                if mosfet_info is not None and num_terminals is not None and len(mosfet_info) > 0:
                    mosfet_currents = self._compute_mosfet_currents_frozen_device_mlp(
                        pred_voltages=pred_voltages,
                        mosfet_info=mosfet_info,
                        terminal_features=data.x,
                        ptr=ptr,
                        num_nodes=num_nodes,
                        voltage_mean=voltage_mean,
                        voltage_std=voltage_std,
                        current_mean=current_mean,
                        current_std=current_std,
                        mlp_stats=mlp_stats,
                    )
                    if mosfet_currents is not None:
                        combined_currents = combined_currents + mosfet_currents
                        voltage_derived_mask = voltage_derived_mask | (mosfet_currents != 0)

                # 2. Resistor currents via Ohm's law
                resistor_info = getattr(data, 'resistor_info', None)
                if resistor_info is not None and len(resistor_info) > 0:
                    resistor_currents = compute_resistor_currents(
                        pred_voltages=pred_voltages,
                        resistor_info=resistor_info,
                        num_nodes=num_nodes,
                        ptr=ptr,
                        voltage_mean=voltage_mean,
                        voltage_std=voltage_std,
                        current_mean=current_mean,
                        current_std=current_std,
                        resistor_ptr=getattr(data, 'resistor_ptr', None),
                    )
                    combined_currents = combined_currents + resistor_currents
                    voltage_derived_mask = voltage_derived_mask | (resistor_currents != 0)

                # 3. Capacitor currents (zeros for DC analysis)
                capacitor_info = getattr(data, 'capacitor_info', None)
                if capacitor_info is not None and len(capacitor_info) > 0:
                    capacitor_currents = compute_capacitor_currents(
                        capacitor_info=capacitor_info,
                        num_nodes=num_nodes,
                        device=device,
                        dtype=dtype,
                    )
                    combined_currents = combined_currents + capacitor_currents

                # 4. I-source currents (known constants)
                isource_info = getattr(data, 'isource_info', None)
                if isource_info is not None and len(isource_info) > 0:
                    isource_currents = compute_isource_currents(
                        isource_info=isource_info,
                        num_nodes=num_nodes,
                        ptr=ptr,
                        current_mean=current_mean,
                        current_std=current_std,
                        isource_ptr=getattr(data, 'isource_ptr', None),
                    )
                    combined_currents = combined_currents + isource_currents
                    voltage_derived_mask = voltage_derived_mask | (isource_currents != 0)

                result['node_currents'] = combined_currents
                result['voltage_derived_current_mask'] = voltage_derived_mask

            elif self.use_device_pooling_current:
                # Device-level pooling: one current per device
                result['node_currents'] = self.device_current_head(
                    x=x,
                    mosfet_info=getattr(data, 'mosfet_info', None),
                    resistor_info=getattr(data, 'resistor_info', None),
                    capacitor_info=getattr(data, 'capacitor_info', None),
                    vsource_info=getattr(data, 'vsource_info', None),
                    isource_info=getattr(data, 'isource_info', None),
                    num_nodes=x.size(0),
                    ptr=getattr(data, 'ptr', None),
                    mosfet_ptr=getattr(data, 'mosfet_ptr', None),
                    resistor_ptr=getattr(data, 'resistor_ptr', None),
                    capacitor_ptr=getattr(data, 'capacitor_ptr', None),
                    isource_ptr=getattr(data, 'isource_ptr', None),
                )
            else:
                # Traditional independent current prediction
                result['node_currents'] = self.current_head(x).squeeze(-1)

            # Z-space KCL projection for 2-term nets: enforce z_A = z_B by averaging
            if self.kcl_zspace_projection:
                result['node_currents_raw'] = result['node_currents']  # pre-projection for KCL loss
                result['node_currents'] = self._apply_kcl_zspace_projection(
                    result['node_currents'], data
                )

        # gm/gds prediction head: predict for ALL nodes, mask to drains in loss
        # (same pattern as voltage/current heads — avoids manual batch index offsetting)
        if self.predict_ss:
            result['mosfet_gm_pred'] = self.gm_head(x).squeeze(-1)  # [num_nodes]
            result['mosfet_gds_pred'] = self.gds_head(x).squeeze(-1)  # [num_nodes]

        # Region head: ordinal regression for ALL nodes, mask to drains in loss
        if self.predict_region:
            result['mosfet_region_pred'] = self.region_head(x).squeeze(-1)  # [num_nodes]

        # AC readout head: graph-level AC quantity prediction
        if self.predict_ac:
            if self.ac_readout == 'pool':
                # Mean+max pool over final node embeddings (same info as voltage head)
                if batch is None:
                    batch = data.batch if hasattr(data, 'batch') else torch.zeros(
                        x.size(0), dtype=torch.long, device=x.device
                    )
                from torch_geometric.nn import global_mean_pool, global_max_pool
                x_mean = global_mean_pool(x, batch)   # [B, mlp_input_dim]
                x_max = global_max_pool(x, batch)     # [B, mlp_input_dim]
                ac_input = torch.cat([x_mean, x_max], dim=-1)  # [B, 2 * mlp_input_dim]
                result['ac_pred'] = self.ac_head(ac_input)
                result['ac_components'] = self.ac_components
            else:
                # VN readout: prefers current GNN's VN, falls back to backbone VN
                ac_input = current_vn_emb if current_vn_emb is not None else vn_emb
                if ac_input is not None and self.ac_detach_vn:
                    ac_input = ac_input.detach()
                if ac_input is not None:
                    result['ac_pred'] = self.ac_head(ac_input)  # [B, num_ac_components]
                    result['ac_components'] = self.ac_components

        return result

    def get_node_embeddings(self, data) -> torch.Tensor:
        """Get node embeddings before prediction heads."""
        x_in = self._get_input_features(data)
        x = self.input_linear(x_in)

        # Get batch info for virtual node (if enabled)
        batch = None
        num_graphs = None
        if self.virtual_node is not None:
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(
                x_in.size(0), dtype=torch.long, device=x_in.device
            )
            num_graphs = self._get_num_graphs(data, batch)

        layer_outputs, _ = self._message_passing(x, data.edge_index, batch, num_graphs)
        x = self._apply_jk(layer_outputs)
        x = self.layers[0].act(self.layers[0].norm(x))

        return x

    def _compute_mosfet_currents_frozen_device_mlp(
        self,
        pred_voltages: torch.Tensor,
        mosfet_info: torch.Tensor,
        terminal_features: torch.Tensor,
        ptr: Optional[torch.Tensor],
        num_nodes: int,
        voltage_mean: float,
        voltage_std: float,
        current_mean: float,
        current_std: float,
        mlp_stats: dict,
    ) -> torch.Tensor:
        """
        Compute MOSFET currents using frozen Device MLP.

        The frozen Device MLP was pre-trained on SPICE data to learn I_ds = f(Vgs, Vds, W, L, is_nmos).
        Gradients flow through the frozen MLP to voltage predictions, providing physics-informed
        gradient signal to the voltage backbone.

        Args:
            pred_voltages: Predicted voltages (normalized z-score) [num_nodes]
            mosfet_info: MOSFET info [num_mosfets, 7] with columns:
                [gate_term, drain_term, source_term, gate_net, drain_net, source_net, is_nmos]
            terminal_features: Node features [num_nodes, feature_dim]
            ptr: Node boundaries per graph [batch_size + 1] or None for single graph
            num_nodes: Total number of nodes
            voltage_mean: Mean for voltage denormalization
            voltage_std: Std for voltage denormalization
            current_mean: Mean for current normalization (log10 scale)
            current_std: Std for current normalization (log10 scale)
            mlp_stats: Normalization stats from Device MLP training

        Returns:
            MOSFET currents (normalized z-score) [num_nodes] with non-zero values at drain terminals
        """
        device = pred_voltages.device
        dtype = pred_voltages.dtype
        num_mosfets = len(mosfet_info)

        # Handle batching: add node offsets to indices
        if ptr is not None and len(ptr) > 2:
            # Batched data - need to add offsets
            num_graphs = len(ptr) - 1
            mosfet_ptr = getattr(data, 'mosfet_ptr', None) if hasattr(data, 'mosfet_ptr') else None

            # Compute graph assignment for each MOSFET
            if mosfet_ptr is not None:
                mosfet_graph_idx = torch.bucketize(
                    torch.arange(num_mosfets, device=device),
                    mosfet_ptr[1:].to(device), right=True)
            else:
                mosfets_per_graph = num_mosfets // num_graphs
                mosfet_graph_idx = torch.arange(num_mosfets, device=device) // mosfets_per_graph

            node_offsets = ptr[mosfet_graph_idx]

            # Build offset-adjusted indices
            gate_net_idx = mosfet_info[:, 3] + node_offsets
            drain_net_idx = mosfet_info[:, 4] + node_offsets
            source_net_idx = mosfet_info[:, 5] + node_offsets
            drain_term_idx = mosfet_info[:, 1] + node_offsets
        else:
            gate_net_idx = mosfet_info[:, 3]
            drain_net_idx = mosfet_info[:, 4]
            source_net_idx = mosfet_info[:, 5]
            drain_term_idx = mosfet_info[:, 1]

        is_nmos = mosfet_info[:, 6].float().to(device)

        # Gather voltages at gate, drain, source nets
        V_g_norm = pred_voltages[gate_net_idx]
        V_d_norm = pred_voltages[drain_net_idx]
        V_s_norm = pred_voltages[source_net_idx]

        # Denormalize voltages to actual volts
        V_g = V_g_norm * voltage_std + voltage_mean
        V_d = V_d_norm * voltage_std + voltage_mean
        V_s = V_s_norm * voltage_std + voltage_mean

        # Compute Vgs and Vds in actual volts
        Vgs = V_g - V_s
        Vds = V_d - V_s

        # Get W, L from terminal features
        # Features are stored at drain terminal, typically as normalized log values
        # Feature indices: 0=W, 1=L, 2=W/L (normalized)
        wl_feat = terminal_features[drain_term_idx, 2]  # W/L ratio (normalized)

        # Denormalize W/L: W/L = 10^(wl_norm * 3.0) based on graph_builder normalization
        wl_ratio = torch.pow(10, 3.0 * wl_feat)

        # Get individual W and L from features if available, otherwise derive from W/L
        # Assuming typical L range around 100nm-1um, use W/L and a reference L
        w_feat = terminal_features[drain_term_idx, 0]  # Normalized W
        l_feat = terminal_features[drain_term_idx, 1]  # Normalized L

        # The normalization in graph_builder uses min-max for W, L
        # For Device MLP we need log10(W) and log10(L) in meters
        # Approximate using typical ranges: W: 0.1um-100um, L: 0.05um-1um
        # log10(W) range: ~[-7, -4], log10(L) range: ~[-7.3, -6]
        log_W = w_feat * 3.0 - 7.0  # Approximate mapping: normalized [0,1] -> log10 [-7, -4]
        log_L = l_feat * 1.3 - 7.3  # Approximate mapping: normalized [0,1] -> log10 [-7.3, -6]

        # Normalize inputs for Device MLP using its training stats
        mlp_Vgs_mean = mlp_stats.get('Vgs_mean', 0.0)
        mlp_Vgs_std = mlp_stats.get('Vgs_std', 1.0)
        mlp_Vds_mean = mlp_stats.get('Vds_mean', 0.0)
        mlp_Vds_std = mlp_stats.get('Vds_std', 1.0)
        mlp_log_W_mean = mlp_stats.get('log_W_mean', -5.5)
        mlp_log_W_std = mlp_stats.get('log_W_std', 0.8)
        mlp_log_L_mean = mlp_stats.get('log_L_mean', -6.8)
        mlp_log_L_std = mlp_stats.get('log_L_std', 0.3)
        mlp_log_I_mean = mlp_stats.get('log_I_mean', -5.0)
        mlp_log_I_std = mlp_stats.get('log_I_std', 1.0)

        Vgs_norm = (Vgs - mlp_Vgs_mean) / mlp_Vgs_std
        Vds_norm = (Vds - mlp_Vds_mean) / mlp_Vds_std
        log_W_norm = (log_W - mlp_log_W_mean) / mlp_log_W_std
        log_L_norm = (log_L - mlp_log_L_mean) / mlp_log_L_std

        # Forward through frozen Device MLP
        # Output is normalized log10(I_ds)
        log_I_ds_norm = self.frozen_device_mlp(Vgs_norm, Vds_norm, log_W_norm, log_L_norm, is_nmos)

        # Denormalize to actual log10(I_ds)
        log_I_ds = log_I_ds_norm * mlp_log_I_std + mlp_log_I_mean

        # Normalize to GNN current target format (z-score of log10)
        I_ds_gnn_norm = (log_I_ds - current_mean) / current_std

        # Scatter currents to drain terminal positions
        currents = torch.zeros(num_nodes, device=device, dtype=dtype)
        currents.scatter_(0, drain_term_idx.to(device), I_ds_gnn_norm)

        return currents
