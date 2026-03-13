"""
Tower GNN architecture: shared backbone + task-specific towers.

Shared backbone (6 GENConv layers) learns general circuit representations,
then splits into:
  - State tower (2 layers): predicts voltage (V) and current (I)
  - Sensitivity tower (2 layers): predicts gm and gds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional

from ..registry import register_model
from ..components.layers import build_mlp, create_deepgcn_layer
from ..components.virtual_node import VirtualNode
from .base import BaseGNN


class JKAggregation(nn.Module):
    """Reusable Jumping Knowledge aggregation module."""

    def __init__(
        self,
        hidden_dim: int,
        num_outputs: int,
        mode: str = 'last',
        attention: bool = False,
        learn_temperature: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.attention = attention
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs

        if attention:
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if learn_temperature:
                self.temperature = nn.Parameter(torch.tensor(1.0))
            else:
                self.register_buffer('temperature', torch.tensor(1.0))

            if mode == 'cat':
                self.linear = nn.Linear(hidden_dim * num_outputs, hidden_dim)
            else:
                self.linear = None
        elif mode == 'cat':
            self.linear = nn.Linear(hidden_dim * num_outputs, hidden_dim)
            self.attn = None
        else:
            self.linear = None
            self.attn = None

    def forward(self, layer_outputs: list) -> torch.Tensor:
        if self.attention:
            layer_stack = torch.stack(layer_outputs, dim=0)
            attn_scores = self.attn(layer_stack)
            temperature = self.temperature.clamp(min=0.1)
            attn_weights = F.softmax(attn_scores / temperature, dim=0)

            if self.linear is not None:
                weighted_stack = layer_stack * attn_weights
                x = weighted_stack.permute(1, 0, 2).reshape(layer_stack.size(1), -1)
                x = self.linear(x)
            else:
                x = (layer_stack * attn_weights).sum(dim=0)

        elif self.mode == 'cat':
            x = torch.cat(layer_outputs, dim=-1)
            x = self.linear(x)

        elif self.mode == 'max':
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]

        elif self.mode == 'sum':
            x = torch.stack(layer_outputs, dim=0).sum(dim=0)

        else:  # 'last'
            x = layer_outputs[-1]

        return x


@register_model("tower_genconv")
class TowerGENConv(BaseGNN):
    """
    Tower GNN: shared backbone + state tower (V/I) + sensitivity tower (gm/gds).

    Architecture:
        Input → Linear → [Backbone: N GENConv layers + VN + JK]
                              │
                    ┌─────────┴──────────┐
                    │                    │
              [State Tower]      [Sensitivity Tower]
              M GENConv layers   M GENConv layers
                    │                    │
              voltage_head          gm_head
              current_head          gds_head
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        # Backbone config
        backbone_layers: int = 6,
        # Tower config
        state_tower_layers: int = 2,
        sensitivity_tower_layers: int = 2,
        # Standard GENConv options
        dropout: float = 0.0,
        genconv_num_layers: int = 2,
        norm_type: str = 'layer',
        skip_connection: bool = True,
        gradient_checkpointing: bool = False,
        # Edge features
        use_edge_features: bool = False,
        edge_feature_dim: int = 6,
        input_dropout: float = 0.0,
        # Virtual Node (backbone only)
        use_virtual_node: bool = False,
        use_attention_pooling: bool = True,
        vn_learn_temperature: bool = False,
        # JK configs per section
        backbone_jk_config: dict = None,
        state_tower_jk_config: dict = None,
        sensitivity_tower_jk_config: dict = None,
        # Prediction heads
        predict_currents: bool = False,
        voltage_head_config: dict = None,
        current_head_config: dict = None,
        ss_head_config: dict = None,
        # Accept and ignore kwargs for compatibility with create_model
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim
        self.skip_connection = skip_connection
        self.predict_currents = predict_currents
        self.gradient_checkpointing = gradient_checkpointing
        self.use_edge_features = use_edge_features
        self.input_dropout = input_dropout
        self.norm_type = norm_type
        self.use_virtual_node = use_virtual_node
        self.backbone_num_layers = backbone_layers
        self.state_tower_num_layers = state_tower_layers
        self.sensitivity_tower_num_layers = sensitivity_tower_layers

        voltage_head_config = voltage_head_config or {}
        current_head_config = current_head_config or {}
        ss_head_config = ss_head_config or {}
        backbone_jk_config = backbone_jk_config or {}
        state_tower_jk_config = state_tower_jk_config or {}
        sensitivity_tower_jk_config = sensitivity_tower_jk_config or {}

        # Input projection
        self.input_linear = nn.Linear(node_feature_dim, hidden_dim)

        # Edge feature dimension
        edge_dim = edge_feature_dim if use_edge_features else None

        # --- Backbone layers ---
        self.backbone = nn.ModuleList([
            create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim)
            for _ in range(backbone_layers)
        ])

        # Virtual Node (backbone only)
        if use_virtual_node:
            self.virtual_node = VirtualNode(
                hidden_dim=hidden_dim,
                use_attention_pooling=use_attention_pooling,
                learn_temperature=vn_learn_temperature,
            )
        else:
            self.virtual_node = None

        # Backbone JK aggregation (over backbone_layers + 1 outputs)
        self.backbone_jk = JKAggregation(
            hidden_dim=hidden_dim,
            num_outputs=backbone_layers + 1,
            mode=backbone_jk_config.get('mode', 'cat'),
            attention=backbone_jk_config.get('attention', True),
            learn_temperature=backbone_jk_config.get('learn_temperature', False),
        )

        # --- State Tower (V/I) ---
        self.state_tower = nn.ModuleList([
            create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim)
            for _ in range(state_tower_layers)
        ])

        self.state_jk = JKAggregation(
            hidden_dim=hidden_dim,
            num_outputs=state_tower_layers + 1,
            mode=state_tower_jk_config.get('mode', 'last'),
            attention=state_tower_jk_config.get('attention', False),
            learn_temperature=state_tower_jk_config.get('learn_temperature', False),
        )

        # Prediction heads input dim
        mlp_input_dim = hidden_dim + node_feature_dim if skip_connection else hidden_dim

        # Voltage head
        v_layers = voltage_head_config.get('num_layers', 1)
        v_hidden = voltage_head_config.get('hidden_dim', hidden_dim)
        v_dropout = voltage_head_config.get('dropout', 0.0)
        self.voltage_head = build_mlp(v_layers, mlp_input_dim, v_hidden, 1, norm_type, v_dropout)

        # Current head (optional)
        if predict_currents:
            c_layers = current_head_config.get('num_layers', 2)
            c_hidden = current_head_config.get('hidden_dim', hidden_dim)
            c_dropout = current_head_config.get('dropout', 0.0)
            self.current_head = build_mlp(c_layers, mlp_input_dim, c_hidden, 1, norm_type, c_dropout)
            # Auxiliary current head for intermediate KCL (after state tower layer 0)
            if state_tower_layers >= 2:
                self.aux_current_head = build_mlp(1, hidden_dim, hidden_dim, 1, norm_type, 0.0)
            else:
                self.aux_current_head = None
        else:
            self.current_head = None
            self.aux_current_head = None

        # --- Sensitivity Tower (gm/gds) ---
        self.has_sensitivity_tower = ss_head_config.get('enabled', False)
        self.state_conditioned_sens = ss_head_config.get('state_conditioned', False)
        self.detach_state_for_sens = ss_head_config.get('detach_state', True)
        if self.has_sensitivity_tower:
            # Optional: fuse state tower embeddings into sensitivity tower input
            if self.state_conditioned_sens:
                self.state_sens_proj = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                )

            self.sensitivity_tower = nn.ModuleList([
                create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim)
                for _ in range(sensitivity_tower_layers)
            ])

            self.sensitivity_jk = JKAggregation(
                hidden_dim=hidden_dim,
                num_outputs=sensitivity_tower_layers + 1,
                mode=sensitivity_tower_jk_config.get('mode', 'last'),
                attention=sensitivity_tower_jk_config.get('attention', False),
                learn_temperature=sensitivity_tower_jk_config.get('learn_temperature', False),
            )

            ss_hidden = ss_head_config.get('hidden_dim', hidden_dim)
            ss_layers = ss_head_config.get('num_layers', 2)
            ss_dropout = ss_head_config.get('dropout', 0.0)
            self.gm_head = build_mlp(ss_layers, mlp_input_dim, ss_hidden, 1, norm_type, ss_dropout)
            self.gds_head = build_mlp(ss_layers, mlp_input_dim, ss_hidden, 1, norm_type, ss_dropout)
        else:
            self.sensitivity_tower = None
            self.sensitivity_jk = None
            self.gm_head = None
            self.gds_head = None

        # Store config for checkpoint save/load
        self._config = {
            'node_feature_dim': node_feature_dim,
            'hidden_dim': hidden_dim,
            'backbone_layers': backbone_layers,
            'state_tower_layers': state_tower_layers,
            'sensitivity_tower_layers': sensitivity_tower_layers,
            'dropout': dropout,
            'genconv_num_layers': genconv_num_layers,
            'norm_type': norm_type,
            'skip_connection': skip_connection,
            'gradient_checkpointing': gradient_checkpointing,
            'use_edge_features': use_edge_features,
            'use_virtual_node': use_virtual_node,
            'predict_currents': predict_currents,
            'has_sensitivity_tower': self.has_sensitivity_tower,
            'state_conditioned_sens': self.state_conditioned_sens,
            'detach_state_for_sens': self.detach_state_for_sens,
        }

    def _run_layers(
        self,
        layers: nn.ModuleList,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        use_vn: bool = False,
        vn_emb: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Run a set of GENConv layers, collecting outputs for JK.

        Args:
            layers: ModuleList of DeepGCN layers
            x: Input node features
            edge_index: Graph connectivity
            edge_attr: Edge features (or None)
            batch: Batch assignment (for VN)
            num_graphs: Number of graphs (for VN)
            use_vn: Whether to use virtual node in these layers
            vn_emb: Current VN embedding (updated in-place if use_vn)

        Returns:
            (layer_outputs, vn_emb): list of tensors + updated VN embedding
        """
        layer_outputs = [x]

        for layer in layers:
            # Broadcast VN to nodes
            if use_vn and self.virtual_node is not None and vn_emb is not None:
                x = x + self.virtual_node.broadcast(vn_emb, batch)

            # Message passing
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

            # Update VN from nodes
            if use_vn and self.virtual_node is not None and vn_emb is not None:
                vn_emb, _ = self.virtual_node(x, vn_emb, batch, num_graphs)

            layer_outputs.append(x)

        return layer_outputs, vn_emb

    def _finalize_repr(self, jk: JKAggregation, layer_outputs: list, ref_layer: nn.Module, x_in: torch.Tensor) -> torch.Tensor:
        """Apply JK aggregation, final norm+act, and skip connection."""
        x = jk(layer_outputs)
        x = ref_layer.act(ref_layer.norm(x))
        if self.skip_connection:
            x = torch.cat([x, x_in], dim=-1)
        return x

    def forward(self, data) -> Dict[str, torch.Tensor]:
        # Get concatenated input features
        x_in = self._get_input_features(data)

        # Input dropout
        if self.input_dropout > 0 and self.training:
            x_in_proj = F.dropout(x_in, p=self.input_dropout, training=True)
        else:
            x_in_proj = x_in

        # Input projection
        x = self.input_linear(x_in_proj)

        # Get batch info
        batch_vec = data.batch if hasattr(data, 'batch') else None
        num_graphs = self._get_num_graphs(data, batch_vec) if batch_vec is not None else 1

        # Edge features
        edge_attr = getattr(data, 'edge_attr', None) if self.use_edge_features else None

        # Initialize VN
        vn_emb = None
        if self.virtual_node is not None:
            vn_emb = self.virtual_node.init_embedding(num_graphs)

        # --- Backbone ---
        backbone_outputs, vn_emb = self._run_layers(
            self.backbone, x, data.edge_index, edge_attr,
            batch=batch_vec, num_graphs=num_graphs,
            use_vn=True, vn_emb=vn_emb,
        )
        backbone_repr = self._finalize_repr(
            self.backbone_jk, backbone_outputs, self.backbone[0], x_in,
        )
        # backbone_repr has skip: [hidden_dim + node_feature_dim] if skip_connection
        # Tower layers need hidden_dim input, so we extract just the JK part
        backbone_hidden = self.backbone_jk(backbone_outputs)
        backbone_hidden = self.backbone[0].act(self.backbone[0].norm(backbone_hidden))

        # --- State Tower ---
        state_outputs, _ = self._run_layers(
            self.state_tower, backbone_hidden, data.edge_index, edge_attr,
        )

        # Get state hidden (before skip) for sensitivity conditioning
        state_hidden = self.state_jk(state_outputs)
        state_hidden = self.state_tower[0].act(self.state_tower[0].norm(state_hidden))

        # State repr with skip connection for prediction heads
        state_repr = torch.cat([state_hidden, x_in], dim=-1) if self.skip_connection else state_hidden

        # State predictions
        result = {}
        result['node_voltages'] = self.voltage_head(state_repr).squeeze(-1)
        result['node_embeddings'] = state_repr

        if self.predict_currents and self.current_head is not None:
            result['node_currents'] = self.current_head(state_repr).squeeze(-1)
            # Auxiliary currents from intermediate state tower layer for deep KCL
            if self.aux_current_head is not None:
                result['aux_node_currents'] = self.aux_current_head(state_outputs[1]).squeeze(-1)

        # --- Sensitivity Tower ---
        if self.has_sensitivity_tower:
            # Optionally condition on state tower embeddings
            if self.state_conditioned_sens:
                state_for_sens = state_hidden.detach() if self.detach_state_for_sens else state_hidden
                sens_input = self.state_sens_proj(torch.cat([backbone_hidden, state_for_sens], dim=-1))
            else:
                sens_input = backbone_hidden

            sens_outputs, _ = self._run_layers(
                self.sensitivity_tower, sens_input, data.edge_index, edge_attr,
            )
            sens_repr = self._finalize_repr(
                self.sensitivity_jk, sens_outputs, self.sensitivity_tower[0], x_in,
            )

            result['mosfet_gm_pred'] = self.gm_head(sens_repr).squeeze(-1)
            result['mosfet_gds_pred'] = self.gds_head(sens_repr).squeeze(-1)

        return result

    def _get_num_graphs(self, data, batch: torch.Tensor) -> int:
        if hasattr(data, 'num_graphs'):
            return data.num_graphs
        if hasattr(data, 'ptr'):
            return len(data.ptr) - 1
        return int(batch.max()) + 1
