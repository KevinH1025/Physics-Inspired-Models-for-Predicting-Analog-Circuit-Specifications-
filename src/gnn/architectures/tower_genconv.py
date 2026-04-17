"""
Tower GNN architecture: shared backbone + task-specific towers.

Shared backbone (6 GENConv layers) learns general circuit representations,
then splits into:
  - State tower (2 layers): predicts voltage (V) and current (I)
  - Sensitivity tower (2 layers): predicts gm and gds
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional

from ..registry import register_model
from ..components.layers import build_mlp, create_deepgcn_layer, get_activation

_LN10 = 2.302585092994046  # math.log(10)

def log10_add(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Compute log10(10^log_a + 10^log_b) in a numerically stable way."""
    return torch.logaddexp(log_a * _LN10, log_b * _LN10) / _LN10
from ..components.virtual_node import VirtualNode
from .base import BaseGNN


# Default 3-stage opamp subcircuit groups (mosfet_info indices)
DEFAULT_SUBCIRCUIT_GROUPS = {
    'bias_pmos': [0, 1, 2, 3, 4, 7],
    'cmfb': [5, 6],
    'diff_pair': [8, 9],
    'cascode': [12, 13, 10, 11],
    'bias_nmos': [14, 16, 18, 15, 17],
    'stage2': [19, 20, 21],
    'output': [22, 23],
}

# Default DAG edges: parent → child (forward signal flow)
# Map: bias_pmos=0, cmfb=1, diff_pair=2, cascode=3, bias_nmos=4, stage2=5, output=6
DEFAULT_DAG_EDGES = [
    (0, 1),  # bias_pmos → cmfb
    (0, 2),  # bias_pmos → diff_pair
    (0, 3),  # bias_pmos → cascode
    (4, 3),  # bias_nmos → cascode
    (1, 3),  # cmfb → cascode
    (2, 3),  # diff_pair → cascode
    (0, 5),  # bias_pmos → stage2
    (3, 5),  # cascode → stage2
    (3, 6),  # cascode → output
    (5, 6),  # stage2 → output
]


class SubcircuitAttentionPool(nn.Module):
    """Multi-head attention pooling for subcircuit terminal embeddings.

    Learnable query tokens attend over variable-length terminal embeddings
    to produce a fixed-size subcircuit representation.
    """

    def __init__(self, input_dim: int = 128, output_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.output_dim = output_dim
        self.queries = nn.Parameter(torch.randn(1, num_heads, self.head_dim) * 0.02)
        self.key_proj = nn.Linear(input_dim, self.head_dim)
        self.value_proj = nn.Linear(input_dim, self.head_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self._scale = self.head_dim ** -0.5

    def forward(self, terminal_embs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Pool terminal embeddings to subcircuit embedding.

        Args:
            terminal_embs: [B, T, input_dim] padded terminal embeddings
            mask: [B, T] bool, True for valid terminals

        Returns:
            [B, output_dim] subcircuit embeddings
        """
        B, T, _ = terminal_embs.shape
        K = self.key_proj(terminal_embs)   # [B, T, head_dim]
        V = self.value_proj(terminal_embs)  # [B, T, head_dim]
        Q = self.queries.expand(B, -1, -1)  # [B, num_heads, head_dim]

        scores = torch.bmm(Q, K.transpose(1, 2)) * self._scale  # [B, num_heads, T]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        out = torch.bmm(weights, V)  # [B, num_heads, head_dim]
        out = out.reshape(B, self.output_dim)
        out = self.norm(self.output_proj(out))
        return out


class JKAggregation(nn.Module):
    """Reusable Jumping Knowledge aggregation module."""

    def __init__(
        self,
        hidden_dim: int,
        num_outputs: int,
        mode: str = 'last',
        attention: bool = False,
        learn_temperature: bool = False,
        gated_residual: bool = False,
        act_type: str = 'relu',
    ):
        super().__init__()
        self.mode = mode
        self.attention = attention
        self.per_feature = (mode == 'attention')  # per-feature softmax across layers
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs
        self.gated_residual = gated_residual
        self.act_type = act_type

        if self.per_feature:
            # Per-feature softmax: each feature independently picks its layer mix
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                get_activation(self.act_type),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
            self.linear = None
        elif gated_residual:
            # Attention over layers 0..N-2, gated addition to last layer
            n_early = max(num_outputs - 1, 1)
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                get_activation(self.act_type),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.register_buffer('temperature', torch.tensor(1.0))
            self.gate_proj = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
            self.linear = None
        elif attention:
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                get_activation(self.act_type),
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
        if self.gated_residual:
            x_last = layer_outputs[-1]  # [N, hidden_dim] — base
            if len(layer_outputs) > 1:
                early = torch.stack(layer_outputs[:-1], dim=0)  # [L-1, N, hidden_dim]
                attn_scores = self.attn(early)
                temperature = self.temperature.clamp(min=0.1)
                attn_weights = F.softmax(attn_scores / temperature, dim=0)
                x_jk = (early * attn_weights).sum(dim=0)  # [N, hidden_dim]
                gate = torch.sigmoid(self.gate_proj(x_last))  # [N, 1]
                x = x_last + gate * x_jk
            else:
                x = x_last
            return x

        if self.per_feature:
            layer_stack = torch.stack(layer_outputs, dim=0)  # [L, N, H]
            attn_scores = self.attn(layer_stack)              # [L, N, H]
            attn_weights = F.softmax(attn_scores, dim=0)      # [L, N, H] sums to 1 across layers
            return (attn_weights * layer_stack).sum(dim=0)     # [N, H]

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


class AttentionSSHead(nn.Module):
    """Terminal attention SS head: self-attention over terminal embeddings,
    then separate learned queries for gm and gds attend over the tokens."""

    def __init__(self, terminal_dim: int, ctx_dim: int = 0, hidden_dim: int = 512,
                 num_attn_layers: int = 1, num_heads: int = 4, mlp_layers: int = 2,
                 dropout: float = 0.0, act_type: str = 'relu'):
        super().__init__()
        self.act_type = act_type
        self.token_dim = terminal_dim + ctx_dim  # each terminal gets its context appended
        # Project all tokens to a common dim divisible by num_heads
        d_model = hidden_dim
        self.token_proj = nn.Linear(self.token_dim, d_model)

        # Self-attention layers over 4 terminal tokens
        attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(attn_layer, num_layers=num_attn_layers)

        # Learned query vectors for gm and gds
        self.gm_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.gds_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Final MLPs
        self.gm_mlp = self._build_mlp(d_model, hidden_dim, mlp_layers, dropout)
        self.gds_mlp = self._build_mlp(d_model, hidden_dim, mlp_layers, dropout)

    def _build_mlp(self, in_dim, hidden_dim, num_layers, dropout):
        layers = []
        curr = in_dim
        for i in range(num_layers):
            out = hidden_dim // (2 ** i) if num_layers > 1 else hidden_dim
            out = max(out, 1)
            layers.extend([nn.Linear(curr, out), nn.LayerNorm(out), get_activation(self.act_type)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr = out
        layers.append(nn.Linear(curr, 1))
        return nn.Sequential(*layers)

    def forward(self, gate, drain, source, bulk, ctx_scalars=None):
        """
        Args: gate/drain/source/bulk: [N, terminal_dim], ctx_scalars: [N, ctx_dim] or None
        Returns: gm_pred [N], gds_pred [N]
        """
        tokens = [gate, drain, source, bulk]
        if ctx_scalars is not None:
            # Append per-terminal voltage context: Vgs→gate, Vds→drain, Vbs→bulk, 0→source
            n = gate.shape[0]
            # Split ctx: [Vgs, Vds, Vbs] + optional [ID] + optional [Vov]
            ctx_dim = ctx_scalars.shape[-1]
            # Each terminal gets the full context (terminal-agnostic)
            tokens = [torch.cat([t, ctx_scalars], dim=-1) for t in tokens]

        # Stack: [N, 4, token_dim] → project → [N, 4, d_model]
        x = torch.stack(tokens, dim=1)
        x = self.token_proj(x)

        # Self-attention over terminals
        x = self.self_attn(x)  # [N, 4, d_model]

        # Cross-attention: learned query attends over terminal tokens
        n = x.shape[0]
        gm_q = self.gm_query.expand(n, -1, -1)   # [N, 1, d_model]
        gds_q = self.gds_query.expand(n, -1, -1)  # [N, 1, d_model]

        gm_repr, _ = self.cross_attn(gm_q, x, x)   # [N, 1, d_model]
        gds_repr, _ = self.cross_attn(gds_q, x, x)  # [N, 1, d_model]

        gm_pred = self.gm_mlp(gm_repr.squeeze(1)).squeeze(-1)    # [N]
        gds_pred = self.gds_mlp(gds_repr.squeeze(1)).squeeze(-1)  # [N]
        return gm_pred, gds_pred


class GatedSSHead(nn.Module):
    """Gated pairwise SS head: explicit pairwise terminal interactions
    with learned gates selecting relevant interactions for gm vs gds."""

    def __init__(self, terminal_dim: int, ctx_dim: int = 0, hidden_dim: int = 512,
                 mlp_layers: int = 2, dropout: float = 0.0, act_type: str = 'relu'):
        super().__init__()
        self.act_type = act_type
        self.terminal_dim = terminal_dim

        # Pairwise interaction projections
        pair_dim = terminal_dim
        self.gate_source_proj = nn.Linear(terminal_dim, pair_dim)
        self.drain_source_proj = nn.Linear(terminal_dim, pair_dim)
        self.gate_drain_proj = nn.Linear(terminal_dim, pair_dim)

        # Gating: separate gates for gm and gds over the 3 pairwise + 4 terminal features
        # Input: 3 pairwise interactions + 4 terminal embeddings = 7 * pair_dim
        gate_input_dim = 7 * pair_dim + ctx_dim
        self.gm_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 7),
            nn.Sigmoid(),
        )
        self.gds_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 7),
            nn.Sigmoid(),
        )

        # Final MLPs take gated features (7 * pair_dim + ctx_dim)
        mlp_in = 7 * pair_dim + ctx_dim
        self.gm_mlp = self._build_mlp(mlp_in, hidden_dim, mlp_layers, dropout)
        self.gds_mlp = self._build_mlp(mlp_in, hidden_dim, mlp_layers, dropout)

    def _build_mlp(self, in_dim, hidden_dim, num_layers, dropout):
        layers = []
        curr = in_dim
        for i in range(num_layers):
            out = hidden_dim // (2 ** i) if num_layers > 1 else hidden_dim
            out = max(out, 1)
            layers.extend([nn.Linear(curr, out), nn.LayerNorm(out), get_activation(self.act_type)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr = out
        layers.append(nn.Linear(curr, 1))
        return nn.Sequential(*layers)

    def forward(self, gate, drain, source, bulk, ctx_scalars=None):
        """
        Args: gate/drain/source/bulk: [N, terminal_dim], ctx_scalars: [N, ctx_dim] or None
        Returns: gm_pred [N], gds_pred [N]
        """
        # Pairwise interactions (element-wise product → project)
        gs = self.gate_source_proj(gate * source)  # gm-relevant
        ds = self.drain_source_proj(drain * source)  # gds-relevant
        gd = self.gate_drain_proj(gate * drain)

        # Stack all 7 features: 4 terminals + 3 pairwise
        all_features = [gate, drain, source, bulk, gs, ds, gd]  # each [N, dim]
        stacked = torch.stack(all_features, dim=1)  # [N, 7, dim]

        # Flat concat for gate computation
        flat = torch.cat(all_features, dim=-1)  # [N, 7*dim]
        if ctx_scalars is not None:
            flat_with_ctx = torch.cat([flat, ctx_scalars], dim=-1)  # [N, 7*dim + ctx]
        else:
            flat_with_ctx = flat

        # Compute gates: [N, 7] — one weight per feature group
        gm_g = self.gm_gate(flat_with_ctx).unsqueeze(-1)    # [N, 7, 1]
        gds_g = self.gds_gate(flat_with_ctx).unsqueeze(-1)   # [N, 7, 1]

        # Apply gates and flatten
        gm_gated = (stacked * gm_g).reshape(stacked.shape[0], -1)    # [N, 7*dim]
        gds_gated = (stacked * gds_g).reshape(stacked.shape[0], -1)  # [N, 7*dim]

        # Append context scalars to MLP input
        if ctx_scalars is not None:
            gm_gated = torch.cat([gm_gated, ctx_scalars], dim=-1)
            gds_gated = torch.cat([gds_gated, ctx_scalars], dim=-1)

        gm_pred = self.gm_mlp(gm_gated).squeeze(-1)
        gds_pred = self.gds_mlp(gds_gated).squeeze(-1)
        return gm_pred, gds_pred


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
        vn_gate_broadcast: bool = False,
        vn_mode: str = 'default',
        vn_num_heads: int = 4,
        vn_head_dim: int = 32,
        # JK configs per section
        backbone_jk_config: dict = None,
        state_tower_jk_config: dict = None,
        sensitivity_tower_jk_config: dict = None,
        # Prediction heads
        predict_currents: bool = False,
        voltage_head_config: dict = None,
        current_head_config: dict = None,
        ss_head_config: dict = None,
        use_device_pooling_current: bool = False,
        # Loop attention
        loop_attention_config: dict = None,
        # Accept and ignore kwargs for compatibility with create_model
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim
        self.act_type = kwargs.get('act_type', 'relu')
        self.conv_type = kwargs.get('conv_type', 'genconv')
        self.conv_num_heads = kwargs.get('conv_num_heads', 4)
        self.mlp_expansion = kwargs.get('mlp_expansion', 2)
        self.mlp_depth = kwargs.get('mlp_depth', 2)
        self.skip_connection = skip_connection
        self.predict_currents = predict_currents
        self.use_device_pooling_current = use_device_pooling_current
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
            create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim, act_type=self.act_type, conv_type=self.conv_type, num_heads=self.conv_num_heads, mlp_expansion=self.mlp_expansion, mlp_depth=self.mlp_depth)
            for _ in range(backbone_layers)
        ])

        # Virtual Node (backbone only)
        if use_virtual_node:
            self.virtual_node = VirtualNode(
                hidden_dim=hidden_dim,
                use_attention_pooling=use_attention_pooling,
                learn_temperature=vn_learn_temperature,
                gate_broadcast=vn_gate_broadcast,
                mode=vn_mode,
                num_heads=vn_num_heads,
                head_dim=vn_head_dim,
                num_layers=backbone_layers,
                act_type=self.act_type,
            )
        else:
            self.virtual_node = None

        # Loop Attention (backbone, optionally all towers)
        loop_attn_cfg = loop_attention_config or {}
        self.use_loop_attention = loop_attn_cfg.get('enabled', False)
        self.loop_attn_warmup_epochs = loop_attn_cfg.get('warmup_epochs', 0)
        self.loop_attn_warmup_duration = loop_attn_cfg.get('warmup_duration', 0)
        self.current_epoch = 0  # set by training loop
        if self.use_loop_attention:
            from src.gnn.components.loop_attention import LoopAttention
            self.loop_attn_apply_to = loop_attn_cfg.get('apply_to', 'backbone')
            n_loop_layers = backbone_layers
            if self.loop_attn_apply_to == 'all':
                n_loop_layers += state_tower_layers + sensitivity_tower_layers
            elif self.loop_attn_apply_to == 'sensitivity':
                n_loop_layers = sensitivity_tower_layers
            self._loop_attn_backbone_count = backbone_layers
            self._loop_attn_state_count = state_tower_layers
            self.loop_attn_layers = nn.ModuleList([
                LoopAttention(
                    hidden_dim=hidden_dim,
                    num_heads=loop_attn_cfg.get('num_heads', 4),
                    head_dim=loop_attn_cfg.get('head_dim', 32),
                    fusion=loop_attn_cfg.get('fusion', 'add'),
                    dropout=dropout,
                    level=loop_attn_cfg.get('level', 'device'),
                    pool_mode=loop_attn_cfg.get('pool_mode', 'mean'),
                ) for _ in range(n_loop_layers)
            ])

        # Backbone JK aggregation (over backbone_layers + 1 outputs)
        self.backbone_jk = JKAggregation(
            hidden_dim=hidden_dim,
            num_outputs=backbone_layers + 1,
            mode=backbone_jk_config.get('mode', 'cat'),
            attention=backbone_jk_config.get('attention', True),
            learn_temperature=backbone_jk_config.get('learn_temperature', False),
            gated_residual=backbone_jk_config.get('gated_residual', False),
            act_type=self.act_type,
        )

        # --- State Tower (V/I) ---
        self.state_tower = nn.ModuleList([
            create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim, act_type=self.act_type, conv_type=self.conv_type, num_heads=self.conv_num_heads, mlp_expansion=self.mlp_expansion, mlp_depth=self.mlp_depth)
            for _ in range(state_tower_layers)
        ])

        self.unified_state_jk = state_tower_jk_config.get('unified', False)
        _state_jk_num = (backbone_layers + 1 + state_tower_layers) if self.unified_state_jk else (state_tower_layers + 1)
        self.state_jk = JKAggregation(
            hidden_dim=hidden_dim,
            num_outputs=_state_jk_num,
            mode=state_tower_jk_config.get('mode', 'last'),
            attention=state_tower_jk_config.get('attention', False),
            learn_temperature=state_tower_jk_config.get('learn_temperature', False),
            gated_residual=state_tower_jk_config.get('gated_residual', False),
            act_type=self.act_type,
        )

        # Prediction heads input dim
        mlp_input_dim = hidden_dim + node_feature_dim if skip_connection else hidden_dim

        # Voltage head
        v_layers = voltage_head_config.get('num_layers', 1)
        v_hidden = voltage_head_config.get('hidden_dim', hidden_dim)
        v_dropout = voltage_head_config.get('dropout', 0.0)
        self.voltage_head = build_mlp(v_layers, mlp_input_dim, v_hidden, 1, norm_type, v_dropout, act_type=self.act_type)

        # Vgs/Vds prediction head (optional)
        vgsvds_cfg = kwargs.get('vgsvds_config', {})
        self.predict_vgsvds = vgsvds_cfg.get('enabled', False)
        if self.predict_vgsvds:
            vd_hidden = vgsvds_cfg.get('hidden_dim', 512)
            vd_layers = vgsvds_cfg.get('num_layers', 2)
            vd_dropout = vgsvds_cfg.get('dropout', 0.0)
            vd_input_dim = 3 * mlp_input_dim  # G+D+S terminal embeddings
            self.vgsvds_head = build_mlp(vd_layers, vd_input_dim, vd_hidden, 2,
                                          norm_type, vd_dropout, act_type=self.act_type)
            self.vgsvds_detach = vgsvds_cfg.get('detach', True)
        else:
            self.vgsvds_head = None

        # Current head (optional)
        self.use_autograd_ss = current_head_config.get('autograd_ss', False) and use_device_pooling_current
        if predict_currents:
            if use_device_pooling_current:
                from src.gnn.components.device_current_head import DevicePoolingCurrentHead
                c_hidden = current_head_config.get('hidden_dim', 256)
                c_layers = current_head_config.get('num_layers', 2)
                c_dropout = current_head_config.get('dropout', 0.0)
                c_all_terminals = current_head_config.get('all_terminals', False)
                self.device_current_head = DevicePoolingCurrentHead(
                    embed_dim=mlp_input_dim,
                    hidden_dim=c_hidden,
                    num_layers=c_layers,
                    dropout=c_dropout,
                    norm_type=norm_type,
                    voltage_input=self.use_autograd_ss,
                    all_terminals=c_all_terminals,
                )
                self.current_head = None
                self.aux_current_head = None
            else:
                self.device_current_head = None
                c_layers = current_head_config.get('num_layers', 2)
                c_hidden = current_head_config.get('hidden_dim', hidden_dim)
                c_dropout = current_head_config.get('dropout', 0.0)
                self.current_head = build_mlp(c_layers, mlp_input_dim, c_hidden, 1, norm_type, c_dropout, act_type=self.act_type)
                # Auxiliary current head for intermediate KCL (after state tower layer 0)
                if state_tower_layers >= 2:
                    self.aux_current_head = build_mlp(1, hidden_dim, hidden_dim, 1, norm_type, 0.0, act_type=self.act_type)
                else:
                    self.aux_current_head = None
        else:
            self.current_head = None
            self.aux_current_head = None
            self.device_current_head = None

        # --- Vov prediction head (under state tower) ---
        vov_head_config = kwargs.get('vov_head_config', {})
        self.predict_vov = vov_head_config.get('enabled', False)
        if self.predict_vov:
            vov_hidden = vov_head_config.get('hidden_dim', 128)
            # Input: gate + source + bulk from state tower = 3 * mlp_input_dim
            self.vov_head = nn.Sequential(
                nn.Linear(3 * mlp_input_dim, vov_hidden),
                nn.LayerNorm(vov_hidden),
                get_activation(self.act_type),
                nn.Linear(vov_hidden, 1),
            )
        else:
            self.vov_head = None

        # --- Vth prediction head (under state tower) ---
        vth_head_config = kwargs.get('vth_head_config', {})
        self.predict_vth = vth_head_config.get('enabled', False)
        if self.predict_vth:
            vth_hidden = vth_head_config.get('hidden_dim', 128)
            # Input: gate + drain + source + bulk from state tower = 4 * mlp_input_dim
            self.vth_head = nn.Sequential(
                nn.Linear(4 * mlp_input_dim, vth_hidden),
                nn.LayerNorm(vth_hidden),
                get_activation(self.act_type),
                nn.Linear(vth_hidden, 1),
            )
        else:
            self.vth_head = None

        # --- Autograd SS: gm/gds from current head via autograd ---
        # --- OR Differentiable I-V model (separate MLP) ---
        iv_cfg = ss_head_config.get('iv_model', {})
        self.use_iv_model = iv_cfg.get('enabled', False) and ss_head_config.get('enabled', False) and not self.use_autograd_ss
        self._needs_autograd = self.use_autograd_ss or self.use_iv_model

        if self.use_autograd_ss:
            # Register normalization stat buffers (set later via set_normalization_stats)
            self.register_buffer('vdc_mean_buf', torch.tensor(0.0))
            self.register_buffer('vdc_std_buf', torch.tensor(1.0))
            self.register_buffer('current_mean_buf', torch.tensor(0.0))
            self.register_buffer('current_std_buf', torch.tensor(1.0))
            self.register_buffer('ss_gm_mean_buf', torch.tensor(0.0))
            self.register_buffer('ss_gm_std_buf', torch.tensor(1.0))
            self.register_buffer('ss_gds_mean_buf', torch.tensor(0.0))
            self.register_buffer('ss_gds_std_buf', torch.tensor(1.0))
            # No sensitivity tower, no MLP heads, no region head
            self.has_sensitivity_tower = False
            self.sensitivity_tower = None
            self.sensitivity_jk = None
            self.gm_head = None
            self.gds_head = None
            self.ss_moe = False
            self.predict_region = False
            self.region_head = None
            self.state_conditioned_sens = False
            self.detach_state_for_sens = True
            self.ss_from_state = False
        elif self.use_iv_model:
            from ..components.differentiable_iv import DifferentiableIVModel
            iv_embed_dim = iv_cfg.get('embed_dim', 64)
            self.iv_embed_proj = nn.Linear(3 * mlp_input_dim, iv_embed_dim)
            self.iv_model = DifferentiableIVModel(
                embed_dim=iv_embed_dim,
                hidden_dim=iv_cfg.get('hidden_dim', 256),
                num_layers=iv_cfg.get('num_layers', 3),
                dropout=iv_cfg.get('dropout', 0.0),
                residual=iv_cfg.get('residual', False),
            )
            self.iv_detach_voltages = iv_cfg.get('detach_voltages', True)
            self.iv_detach_embeddings = iv_cfg.get('detach_embeddings', True)
            # Register normalization stat buffers
            self.register_buffer('vdc_mean_buf', torch.tensor(0.0))
            self.register_buffer('vdc_std_buf', torch.tensor(1.0))
            self.register_buffer('ss_gm_mean_buf', torch.tensor(0.0))
            self.register_buffer('ss_gm_std_buf', torch.tensor(1.0))
            self.register_buffer('ss_gds_mean_buf', torch.tensor(0.0))
            self.register_buffer('ss_gds_std_buf', torch.tensor(1.0))
            # No sensitivity tower
            self.has_sensitivity_tower = False
            self.sensitivity_tower = None
            self.sensitivity_jk = None
            self.gm_head = None
            self.gds_head = None
            self.ss_moe = False
            self.predict_region = False
            self.region_head = None
            self.state_conditioned_sens = False
            self.detach_state_for_sens = True
            self.ss_from_state = False

        # --- Sensitivity Tower (gm/gds) --- (skipped when autograd SS or IV model is enabled)
        _skip_sens = self.use_autograd_ss or self.use_iv_model
        if not _skip_sens:
            self.has_sensitivity_tower = ss_head_config.get('enabled', False)
        self.state_conditioned_sens = ss_head_config.get('state_conditioned', False) if not _skip_sens else False
        self.detach_state_for_sens = ss_head_config.get('detach_state', True) if not _skip_sens else True
        self.ss_from_state = (ss_head_config.get('source', 'sensitivity_tower') == 'state_tower') if not _skip_sens else False
        if self.has_sensitivity_tower:
            if not self.ss_from_state:
                # Build sensitivity tower GNN layers (skipped when reading from state tower)
                # Optional: fuse state tower embeddings into sensitivity tower input
                if self.state_conditioned_sens:
                    self.state_sens_proj = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        get_activation(self.act_type),
                    )

                self.ss_branch_layers = ss_head_config.get('sensitivity_branch_layers', 0)

                self.sensitivity_tower = nn.ModuleList([
                    create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim, act_type=self.act_type, conv_type=self.conv_type, num_heads=self.conv_num_heads, mlp_expansion=self.mlp_expansion, mlp_depth=self.mlp_depth)
                    for _ in range(sensitivity_tower_layers)
                ])

                self.unified_sensitivity_jk = sensitivity_tower_jk_config.get('unified', False)
                _sens_jk_num = (backbone_layers + 1 + sensitivity_tower_layers) if self.unified_sensitivity_jk else (sensitivity_tower_layers + 1)
                self.sensitivity_jk = JKAggregation(
                    hidden_dim=hidden_dim,
                    num_outputs=_sens_jk_num,
                    mode=sensitivity_tower_jk_config.get('mode', 'last'),
                    attention=sensitivity_tower_jk_config.get('attention', False),
                    learn_temperature=sensitivity_tower_jk_config.get('learn_temperature', False),
                    gated_residual=sensitivity_tower_jk_config.get('gated_residual', False),
                    act_type=self.act_type,
                )

                # Y-shaped branches: separate gm/gds GNN layers after shared sensitivity tower
                if self.ss_branch_layers > 0:
                    self.gm_branch = nn.ModuleList([
                        create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim, act_type=self.act_type, conv_type=self.conv_type, num_heads=self.conv_num_heads, mlp_expansion=self.mlp_expansion, mlp_depth=self.mlp_depth)
                        for _ in range(self.ss_branch_layers)
                    ])
                    self.gds_branch = nn.ModuleList([
                        create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout, edge_dim=edge_dim, act_type=self.act_type, conv_type=self.conv_type, num_heads=self.conv_num_heads, mlp_expansion=self.mlp_expansion, mlp_depth=self.mlp_depth)
                        for _ in range(self.ss_branch_layers)
                    ])
                    self.gm_branch_jk = JKAggregation(
                        hidden_dim=hidden_dim,
                        num_outputs=self.ss_branch_layers + 1,
                        mode=sensitivity_tower_jk_config.get('mode', 'last'),
                        attention=sensitivity_tower_jk_config.get('attention', False),
                        learn_temperature=sensitivity_tower_jk_config.get('learn_temperature', False),
                        act_type=self.act_type,
                    )
                    self.gds_branch_jk = JKAggregation(
                        hidden_dim=hidden_dim,
                        num_outputs=self.ss_branch_layers + 1,
                        mode=sensitivity_tower_jk_config.get('mode', 'last'),
                        attention=sensitivity_tower_jk_config.get('attention', False),
                        learn_temperature=sensitivity_tower_jk_config.get('learn_temperature', False),
                        act_type=self.act_type,
                    )
                else:
                    self.gm_branch = None
                    self.gds_branch = None
            else:
                # ss_from_state: no sensitivity tower GNN, readout heads read from state tower
                self.sensitivity_tower = None
                self.sensitivity_jk = None
                self.ss_branch_layers = 0
                self.gm_branch = None
                self.gds_branch = None

            ss_hidden = ss_head_config.get('hidden_dim', hidden_dim)
            ss_layers = ss_head_config.get('num_layers', 2)
            ss_dropout = ss_head_config.get('dropout', 0.0)
            self.ss_diff_features = ss_head_config.get('diff_features', False)
            self.ss_include_bulk = ss_head_config.get('include_bulk', False)
            self.ss_state_context = ss_head_config.get('state_context', False)
            self.ss_physics_pool = ss_head_config.get('physics_pool', False)
            self.ss_voltage_context = ss_head_config.get('voltage_context', False)
            self.ss_cross_attention = ss_head_config.get('cross_attention', False)
            self.ss_pairwise = ss_head_config.get('pairwise', False)
            self.ss_current_context = ss_head_config.get('current_context', False)
            self.ss_vov_context = ss_head_config.get('vov_context', False)
            self.ss_region_context = ss_head_config.get('region_context', False)
            self.ss_wl_context = ss_head_config.get('wl_context', False)
            self.ss_vth_context = ss_head_config.get('vth_context', False)

            if self.ss_cross_attention:
                # Find largest num_heads (up to 4) that divides mlp_input_dim
                xattn_heads = max(h for h in [1, 2, 3, 4] if mlp_input_dim % h == 0)
                self.ss_cross_attn = nn.MultiheadAttention(
                    embed_dim=mlp_input_dim, num_heads=xattn_heads, batch_first=True,
                )
                self.ss_cross_attn_norm = nn.LayerNorm(mlp_input_dim)

            if self.ss_pairwise:
                # Pairwise interaction projections: gate*drain, gate*source, drain*source
                self.pw_gd = nn.Linear(mlp_input_dim, mlp_input_dim)
                self.pw_gs = nn.Linear(mlp_input_dim, mlp_input_dim)
                self.pw_ds = nn.Linear(mlp_input_dim, mlp_input_dim)

            # Build gm/gds heads from config
            def _build_ss_head(in_dim, hidden_dim, num_layers, dropout):
                layers = []
                curr_dim = in_dim
                for i in range(num_layers):
                    out_dim = hidden_dim // (2 ** i) if num_layers > 1 else hidden_dim
                    out_dim = max(out_dim, 1)
                    layers.extend([
                        nn.Linear(curr_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        get_activation(self.act_type),
                    ])
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    curr_dim = out_dim
                layers.append(nn.Linear(curr_dim, 1))
                return nn.Sequential(*layers)

            # +3 scalars (Vgs, Vds, Vbs) if voltage_context, +1 (ID) if current_context, +1 (Vov) if vov_context
            # +3 (region probs) if region_context, +3 (w_norm, l_norm, wl_ratio_norm) if wl_context
            volt_ctx_dim = 3 if self.ss_voltage_context else 0
            curr_ctx_dim = 1 if self.ss_current_context else 0
            vov_ctx_dim = 1 if self.ss_vov_context else 0
            region_ctx_dim = 3 if self.ss_region_context else 0
            wl_ctx_dim = 3 if self.ss_wl_context else 0
            vth_ctx_dim = 1 if self.ss_vth_context else 0
            ctx_dim = volt_ctx_dim + curr_ctx_dim + vov_ctx_dim + region_ctx_dim + wl_ctx_dim + vth_ctx_dim

            self.ss_head_type = ss_head_config.get('head_type', 'mlp')
            self.ss_moe = False  # overridden in standard MLP path if configured

            if self.ss_head_type == 'attention':
                self.ss_attn_head = AttentionSSHead(
                    terminal_dim=mlp_input_dim, ctx_dim=ctx_dim,
                    hidden_dim=ss_hidden,
                    num_attn_layers=ss_head_config.get('num_attn_layers', 1),
                    num_heads=ss_head_config.get('num_heads', 4),
                    mlp_layers=ss_layers, dropout=ss_dropout,
                )
                self.gm_head = None
                self.gds_head = None
                self.state_context_mlp = None
            elif self.ss_head_type == 'gated':
                self.ss_gated_head = GatedSSHead(
                    terminal_dim=mlp_input_dim, ctx_dim=ctx_dim,
                    hidden_dim=ss_hidden,
                    mlp_layers=ss_layers, dropout=ss_dropout,
                )
                self.gm_head = None
                self.gds_head = None
                self.state_context_mlp = None
            elif self.ss_physics_pool:
                # gm: gate+source (2 terminals), gds: drain+source (2 terminals)
                gm_input_dim = 2 * mlp_input_dim + ctx_dim
                gds_input_dim = 2 * mlp_input_dim + ctx_dim
                self.gm_head = _build_ss_head(gm_input_dim, ss_hidden, ss_layers, ss_dropout)
                self.gds_head = _build_ss_head(gds_input_dim, ss_hidden, ss_layers, ss_dropout)
            else:
                # Base input: gate + drain + source [+ bulk] [+ difference] [+ 3 pairwise] from sensitivity tower
                self.ss_drain_only = ss_head_config.get('drain_only', False)
                if self.ss_drain_only:
                    n_terms = 1  # drain only
                else:
                    n_terms = 3 + (1 if self.ss_include_bulk else 0) + (1 if self.ss_diff_features else 0) + (3 if self.ss_pairwise else 0)
                ss_input_dim = n_terms * mlp_input_dim + ctx_dim

                # State context: detached summary of state tower terminal embeddings
                if self.ss_state_context:
                    state_ctx_dim = 128
                    self.state_context_mlp = nn.Sequential(
                        nn.Linear(4 * mlp_input_dim, state_ctx_dim),  # gate+drain+source+bulk from state
                        nn.LayerNorm(state_ctx_dim),
                        get_activation(self.act_type),
                    )
                    ss_input_dim += state_ctx_dim  # append state summary to sensitivity concat
                else:
                    self.state_context_mlp = None

                # gm/Id head from sensitivity tower (can be alongside gm/gds heads)
                self.ss_predict_gm_id = ss_head_config.get('predict_gm_id', False)
                if self.ss_predict_gm_id:
                    self.gm_id_ss_head = _build_ss_head(ss_input_dim, ss_hidden, ss_layers, ss_dropout)
                    self.register_buffer('ss_gm_id_mean_buf', torch.tensor(0.0))
                    self.register_buffer('ss_gm_id_std_buf', torch.tensor(1.0))
                else:
                    self.gm_id_ss_head = None

                # gm/gds heads (always created unless MoE)
                self.ss_moe = ss_head_config.get('mixture_of_experts', False)
                if self.ss_moe:
                    expert_hidden = ss_head_config.get('expert_hidden_dim', 256)
                    expert_layers = ss_head_config.get('expert_num_layers', 2)
                    self.gm_experts = nn.ModuleList([
                        _build_ss_head(ss_input_dim, expert_hidden, expert_layers, ss_dropout)
                        for _ in range(3)
                    ])
                    self.gds_experts = nn.ModuleList([
                        _build_ss_head(ss_input_dim, expert_hidden, expert_layers, ss_dropout)
                        for _ in range(3)
                    ])
                    self.gm_head = None
                    self.gds_head = None
                else:
                    self.gm_experts = None
                    self.gds_experts = None
                    # Subcircuit-conditioned heads: separate gm/gds MLP per subcircuit
                    self.use_subcircuit_heads = ss_head_config.get('subcircuit_heads', False)
                    if self.use_subcircuit_heads:
                        default_groups = {
                            'bias_pmos': [0, 1, 2, 3, 4, 7],
                            'cmfb': [5, 6],
                            'diff_pair': [8, 9],
                            'cascode': [12, 13, 10, 11],
                            'bias_nmos': [14, 16, 18, 15, 17],
                            'stage2': [19, 20, 21],
                            'output': [22, 23],
                        }
                        self._subcircuit_groups = ss_head_config.get('subcircuit_groups', None) or default_groups
                        self.subcircuit_gm_heads = nn.ModuleDict({
                            name: _build_ss_head(ss_input_dim, ss_hidden, ss_layers, ss_dropout)
                            for name in self._subcircuit_groups
                        })
                        self.subcircuit_gds_heads = nn.ModuleDict({
                            name: _build_ss_head(ss_input_dim, ss_hidden, ss_layers, ss_dropout)
                            for name in self._subcircuit_groups
                        })
                        self.gm_head = None
                        self.gds_head = None
                    else:
                        self.gm_head = _build_ss_head(ss_input_dim, ss_hidden, ss_layers, ss_dropout)
                        self.gds_head = _build_ss_head(ss_input_dim, ss_hidden, ss_layers, ss_dropout)

            # Region prediction head (CORAL ordinal, under sensitivity tower)
            region_head_config = kwargs.get('region_head_config', None)
            self.predict_region = (region_head_config is not None and
                                   region_head_config.get('enabled', False))
            if self.predict_region:
                region_hidden = region_head_config.get('hidden_dim', 128)
                region_layers = region_head_config.get('num_layers', 1)
                region_dropout = region_head_config.get('dropout', 0.0)
                n_region_terms = 4 if self.ss_include_bulk else 3
                region_input_dim = n_region_terms * mlp_input_dim
                # Shrinking pyramid MLP → 2 ordinal logits
                reg_layers = []
                curr_dim = region_input_dim
                for i in range(region_layers):
                    out_dim = region_hidden // (2 ** i) if region_layers > 1 else region_hidden
                    out_dim = max(out_dim, 1)
                    reg_layers.extend([nn.Linear(curr_dim, out_dim), nn.LayerNorm(out_dim), get_activation(self.act_type)])
                    if region_dropout > 0:
                        reg_layers.append(nn.Dropout(region_dropout))
                    curr_dim = out_dim
                reg_layers.append(nn.Linear(curr_dim, 2))  # 2 ordinal thresholds
                self.region_head = nn.Sequential(*reg_layers)
            else:
                self.region_head = None

            if self.ss_moe:
                assert self.predict_region, \
                    "mixture_of_experts requires region head to be enabled"
        else:
            self.sensitivity_tower = None
            self.sensitivity_jk = None
            self.gm_head = None
            self.gds_head = None
            self.ss_moe = False
            self.predict_region = False
            self.region_head = None

        # gm/Id auxiliary head (backbone-level, per-MOSFET)
        gm_id_head_config = kwargs.get('gm_id_head_config', {})
        self.predict_gm_id = gm_id_head_config.get('enabled', False)
        if self.predict_gm_id:
            gm_id_hidden = gm_id_head_config.get('hidden_dim', 128)
            gm_id_layers = gm_id_head_config.get('num_layers', 2)
            gm_id_input_dim = 3 * mlp_input_dim  # gate+drain+source from backbone_repr
            gm_id_mlp = []
            curr_dim = gm_id_input_dim
            for i in range(gm_id_layers):
                out_dim = max(gm_id_hidden // (2 ** i), 1)
                gm_id_mlp.extend([nn.Linear(curr_dim, out_dim), nn.LayerNorm(out_dim), get_activation(self.act_type)])
                curr_dim = out_dim
            gm_id_mlp.append(nn.Linear(curr_dim, 1))
            self.gm_id_head = nn.Sequential(*gm_id_mlp)
        else:
            self.gm_id_head = None
        self.gmid_as_feature = gm_id_head_config.get('as_feature', False) and self.predict_gm_id
        if self.gmid_as_feature:
            self.gmid_feature_proj = nn.Linear(hidden_dim + 1, hidden_dim)

        # AC head (graph-level: UGBW, PM, AM)
        ac_head_config = kwargs.get('ac_head_config', {})
        self.predict_ac = ac_head_config.get('enabled', False)
        if self.predict_ac:
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
                ac_dropout = ac_head_config.get('dropout', 0.0)
                self.ac_readout = ac_head_config.get('readout', 'vn')
                if self.ac_readout in ('cross_attn', 'mha_pool'):
                    # Layer attention: score each backbone layer's pooled MHA output
                    self.ac_layer_attn = nn.Sequential(
                        nn.Linear(hidden_dim, 64),
                        get_activation(self.act_type),
                        nn.Linear(64, 1),
                    )
                    if self.ac_readout == 'cross_attn':
                        # Cross-attention: Q=layer-attended MHA pool, K/V=state node embeddings
                        ac_num_heads = ac_head_config.get('num_heads', 8)
                        ac_head_dim = ac_head_config.get('head_dim', 48)
                        ac_attn_dim = ac_num_heads * ac_head_dim  # 384 or 512
                        self.ac_q_proj = nn.Linear(hidden_dim, ac_attn_dim)
                        self.ac_kv_proj = nn.Linear(mlp_input_dim, ac_attn_dim)
                        self.ac_cross_attn = nn.MultiheadAttention(
                            embed_dim=ac_attn_dim, num_heads=ac_num_heads,
                            batch_first=True, dropout=ac_dropout,
                        )
                        self.ac_attn_norm = nn.LayerNorm(ac_attn_dim)
                        # MLP: concat(attn_out, ac_emb) → predictions
                        ac_mlp_input = ac_attn_dim + hidden_dim  # 384+128=512 or 512+128=640
                        self.ac_head = nn.Sequential(
                            nn.Linear(ac_mlp_input, 256),
                            nn.LayerNorm(256),
                            get_activation(self.act_type),
                            nn.Linear(256, 128),
                            nn.LayerNorm(128),
                            get_activation(self.act_type),
                            nn.Linear(128, ac_output_dim),
                        )
                    else:  # mha_pool: just layer-attended MHA pool → MLP
                        self.ac_head = nn.Sequential(
                            nn.Linear(hidden_dim, 128),
                            nn.LayerNorm(128),
                            get_activation(self.act_type),
                            nn.Linear(128, ac_output_dim),
                        )
                elif self.ac_readout == 'vn':
                    # Uses vn_emb directly (128-dim, only works with default VN mode)
                    ac_hidden = ac_head_config.get('hidden_dim', 128)
                    ac_layers = ac_head_config.get('num_layers', 3)
                    ac_input_dim = hidden_dim  # vn_emb is [B, hidden_dim]
                    self.ac_head = build_mlp(ac_layers, ac_input_dim, ac_hidden, ac_output_dim, norm_type, ac_dropout, act_type=self.act_type)
                elif self.ac_readout == 'mosfet_concat':
                    # Flatten 14 key MOSFET embeddings from sensitivity tower
                    self.register_buffer('_ac_mosfet_indices', torch.tensor(
                        [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                    self.ac_detach = ac_head_config.get('detach', True)
                    concat_input_dim = 14 * mlp_input_dim  # 2086
                    self.ac_head = nn.Sequential(
                        nn.Linear(concat_input_dim, 512),
                        nn.LayerNorm(512),
                        get_activation(self.act_type),
                        nn.Linear(512, 256),
                        nn.LayerNorm(256),
                        get_activation(self.act_type),
                        nn.Linear(256, 128),
                        nn.LayerNorm(128),
                        get_activation(self.act_type),
                        nn.Linear(128, ac_output_dim),
                    )
                elif self.ac_readout == 'mosfet_physics':
                    # UGBW physics formula in log space + correction MLP
                    self.register_buffer('_ac_mosfet_indices', torch.tensor(
                        [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                    self.ac_detach = ac_head_config.get('detach', True)
                    if not hasattr(self, 'ss_gm_mean_buf'):
                        self.register_buffer('ss_gm_mean_buf', torch.tensor(0.0))
                        self.register_buffer('ss_gm_std_buf', torch.tensor(1.0))
                        self.register_buffer('ss_gds_mean_buf', torch.tensor(0.0))
                        self.register_buffer('ss_gds_std_buf', torch.tensor(1.0))
                    r_in = float(ac_head_config.get('r_in', 50000.0))
                    r_f = float(ac_head_config.get('r_f', 50000.0))
                    if not hasattr(self, '_dc_r_in'):
                        self.register_buffer('_dc_r_in', torch.tensor(r_in))
                        self.register_buffer('_dc_r_f', torch.tensor(r_f))
                    self._ac_cap_local_idx = 110  # C0 positive terminal (fixed topology)
                    self._ac_cc_log10_min = math.log10(0.5e-12)
                    self._ac_cc_log10_range = math.log10(30e-12) - math.log10(0.5e-12)
                    ac_physics_dim = 30  # 14 z_gm + 14 z_gds + Cc_norm + log10_ugbw_est
                    dc_hidden = ac_head_config.get('hidden_dim', 128)
                    self.ac_head = nn.Sequential(
                        nn.Linear(ac_physics_dim, dc_hidden),
                        nn.LayerNorm(dc_hidden),
                        get_activation(self.act_type),
                        nn.Linear(dc_hidden, dc_hidden // 2),
                        nn.LayerNorm(dc_hidden // 2),
                        get_activation(self.act_type),
                        nn.Linear(dc_hidden // 2, ac_output_dim),
                    )
                elif self.ac_readout == 'mosfet_physics_dag':
                    # UGBW physics formula + DAG subcircuit embeddings
                    self.register_buffer('_ac_mosfet_indices', torch.tensor(
                        [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                    self.ac_detach = ac_head_config.get('detach', True)
                    if not hasattr(self, 'ss_gm_mean_buf'):
                        self.register_buffer('ss_gm_mean_buf', torch.tensor(0.0))
                        self.register_buffer('ss_gm_std_buf', torch.tensor(1.0))
                        self.register_buffer('ss_gds_mean_buf', torch.tensor(0.0))
                        self.register_buffer('ss_gds_std_buf', torch.tensor(1.0))
                    r_in = float(ac_head_config.get('r_in', 50000.0))
                    r_f = float(ac_head_config.get('r_f', 50000.0))
                    if not hasattr(self, '_dc_r_in'):
                        self.register_buffer('_dc_r_in', torch.tensor(r_in))
                        self.register_buffer('_dc_r_f', torch.tensor(r_f))
                    self._ac_cap_local_idx = 110
                    self._ac_cc_log10_min = math.log10(0.5e-12)
                    self._ac_cc_log10_range = math.log10(30e-12) - math.log10(0.5e-12)
                    # Physics (30) + DAG embeddings (7 × sc_dim)
                    _sc_dag_cfg = kwargs.get('subcircuit_dag_config', {})
                    sc_dim = _sc_dag_cfg.get('dim', 128)
                    self._ac_dag_dim = 7 * sc_dim
                    ac_physics_dim = 30 + self._ac_dag_dim
                    self.ac_head = nn.Sequential(
                        nn.Linear(ac_physics_dim, 512),
                        nn.LayerNorm(512),
                        get_activation(self.act_type),
                        nn.Linear(512, 256),
                        nn.LayerNorm(256),
                        get_activation(self.act_type),
                        nn.Linear(256, ac_output_dim),
                    )
                elif self.ac_readout == 'physics_dag_residual':
                    # Physics UGBW estimate + DAG output stage residual correction
                    self.register_buffer('_ac_mosfet_indices', torch.tensor(
                        [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                    self.ac_detach = True  # always detach gm/gds for physics part
                    if not hasattr(self, 'ss_gm_mean_buf'):
                        self.register_buffer('ss_gm_mean_buf', torch.tensor(0.0))
                        self.register_buffer('ss_gm_std_buf', torch.tensor(1.0))
                        self.register_buffer('ss_gds_mean_buf', torch.tensor(0.0))
                        self.register_buffer('ss_gds_std_buf', torch.tensor(1.0))
                    r_in = float(ac_head_config.get('r_in', 50000.0))
                    r_f = float(ac_head_config.get('r_f', 50000.0))
                    if not hasattr(self, '_dc_r_in'):
                        self.register_buffer('_dc_r_in', torch.tensor(r_in))
                        self.register_buffer('_dc_r_f', torch.tensor(r_f))
                    self._ac_cap_local_idx = 110
                    self._ac_cc_log10_min = math.log10(0.5e-12)
                    self._ac_cc_log10_range = math.log10(30e-12) - math.log10(0.5e-12)
                    # Physics correction MLP (same as mosfet_physics)
                    ac_physics_dim = 30
                    dc_hidden = ac_head_config.get('hidden_dim', 128)
                    self.ac_head = nn.Sequential(
                        nn.Linear(ac_physics_dim, dc_hidden),
                        nn.LayerNorm(dc_hidden), get_activation(self.act_type),
                        nn.Linear(dc_hidden, dc_hidden // 2),
                        nn.LayerNorm(dc_hidden // 2), get_activation(self.act_type),
                        nn.Linear(dc_hidden // 2, ac_output_dim),
                    )
                    # DAG residual correction from last subcircuit embedding
                    _sc_dag_cfg = kwargs.get('subcircuit_dag_config', {})
                    sc_dim = _sc_dag_cfg.get('dim', 128)
                    self.ac_dag_correction = nn.Sequential(
                        nn.Linear(sc_dim, 64),
                        nn.LayerNorm(64), get_activation(self.act_type),
                        nn.Linear(64, ac_output_dim),
                    )
                    # Init correction near zero so physics dominates initially
                    nn.init.zeros_(self.ac_dag_correction[-1].weight)
                    nn.init.zeros_(self.ac_dag_correction[-1].bias)
                elif self.ac_readout == 'dag_only':
                    # UGBW from last DAG subcircuit embedding (output stage, has upstream context)
                    self.ac_detach = ac_head_config.get('detach', True)
                    _sc_dag_cfg = kwargs.get('subcircuit_dag_config', {})
                    sc_dim = _sc_dag_cfg.get('dim', 128)
                    self.ac_head = nn.Sequential(
                        nn.Linear(sc_dim, 256),
                        nn.LayerNorm(256),
                        get_activation(self.act_type),
                        nn.Linear(256, 128),
                        nn.LayerNorm(128),
                        get_activation(self.act_type),
                        nn.Linear(128, ac_output_dim),
                    )
                else:
                    # 'pool' fallback: mean+max pool of state_repr
                    ac_hidden = ac_head_config.get('hidden_dim', 128)
                    ac_layers = ac_head_config.get('num_layers', 3)
                    ac_input_dim = mlp_input_dim * 2  # mean+max pool
                    self.ac_head = build_mlp(ac_layers, ac_input_dim, ac_hidden, ac_output_dim, norm_type, ac_dropout, act_type=self.act_type)

        # Subcircuit DAG (between backbone and towers)
        sc_dag_config = kwargs.get('subcircuit_dag_config', {})
        self.use_subcircuit_dag = sc_dag_config.get('enabled', False)
        if self.use_subcircuit_dag:
            self.sc_dag_warmup_epochs = sc_dag_config.get('warmup_epochs', 0)
            self.sc_dag_warmup_duration = sc_dag_config.get('warmup_duration', 0)
            sc_dim = sc_dag_config.get('dim', 128)
            sc_num_heads = sc_dag_config.get('num_heads', 4)
            sc_groups = sc_dag_config.get('groups', None) or DEFAULT_SUBCIRCUIT_GROUPS
            sc_edges = sc_dag_config.get('edges', None) or DEFAULT_DAG_EDGES
            self._sc_groups = sc_groups
            self._sc_group_names = list(sc_groups.keys())
            self._sc_num_groups = len(self._sc_group_names)
            # Build assignment: mosfet index → subcircuit index
            assignment = torch.full((24,), -1, dtype=torch.long)
            for sc_idx, (sc_name, device_indices) in enumerate(sc_groups.items()):
                for dev_idx in device_indices:
                    assignment[dev_idx] = sc_idx
            self.register_buffer('_sc_assignment', assignment)
            # Build topological order from DAG edges
            parent_map = {i: [] for i in range(self._sc_num_groups)}
            for src, dst in sc_edges:
                parent_map[dst].append(src)
            # Process non-root nodes in dependency order
            self._sc_topo_order = []
            processed = set(i for i in range(self._sc_num_groups) if not parent_map[i])
            remaining = set(range(self._sc_num_groups)) - processed
            while remaining:
                for child in sorted(remaining):
                    if all(p in processed for p in parent_map[child]):
                        self._sc_topo_order.append((child, parent_map[child]))
                        processed.add(child)
                remaining -= processed
            # Attention pooler
            self.sc_attn_pool = SubcircuitAttentionPool(
                input_dim=hidden_dim, output_dim=sc_dim, num_heads=sc_num_heads)
            # DAGNN-style attention + GRU for sequential DAG processing
            self.sc_dag_W_q = nn.Linear(sc_dim, sc_dim)
            self.sc_dag_W_k = nn.Linear(sc_dim, sc_dim)
            self.sc_dag_W_v = nn.Linear(sc_dim, sc_dim)
            self.sc_dag_attn_a = nn.Linear(2 * sc_dim, 1)
            self.sc_dag_gru = nn.GRUCell(sc_dim, sc_dim)
            # Gated fusion
            self.sc_gate_proj = nn.Linear(hidden_dim + sc_dim, hidden_dim)
            nn.init.zeros_(self.sc_gate_proj.bias)

        # Flag for capturing MHA outputs in _run_layers (updated after DC gain init)
        self._need_mha_capture = (self.predict_ac and getattr(self, 'ac_readout', '') in ('cross_attn', 'mha_pool'))

        # DC gain head (graph-level)
        dc_gain_config = kwargs.get('dc_gain_config', {})
        self.predict_dc_gain = dc_gain_config.get('enabled', False)
        self.dc_gain_mode = dc_gain_config.get('mode', 'cross_attn')
        if self.predict_dc_gain:
            if self.dc_gain_mode == 'physics':
                # Physics-informed DC gain head: uses predicted gm/gds from SS head
                # + analytical circuit equations → small MLP for correction.
                #
                # 14 key signal-path MOSFETs (mosfet_info indices):
                #   Stage 1: M8(8), M9(9), M5(5), M6(6), M15(12), M16(13), M19(10), M20(11)
                #   Stage 2: M10(19), M21(20), M22(21), M7(7)
                #   Stage 3: M11(22), M23(23)
                self.register_buffer('_dc_mosfet_indices', torch.tensor(
                    [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                self.dc_gain_detach_ss = dc_gain_config.get('detach_ss', False)
                # Register SS normalization buffers if not already present (needed for denormalization)
                if not hasattr(self, 'ss_gm_mean_buf'):
                    self.register_buffer('ss_gm_mean_buf', torch.tensor(0.0))
                    self.register_buffer('ss_gm_std_buf', torch.tensor(1.0))
                    self.register_buffer('ss_gds_mean_buf', torch.tensor(0.0))
                    self.register_buffer('ss_gds_std_buf', torch.tensor(1.0))
                # Feedback network resistances (configurable for future topologies)
                r_in = float(dc_gain_config.get('r_in', 50000.0))
                r_f = float(dc_gain_config.get('r_f', 50000.0))
                self.register_buffer('_dc_r_in', torch.tensor(r_in))
                self.register_buffer('_dc_r_f', torch.tensor(r_f))
                self._dc_rout1_formula = dc_gain_config.get('rout1_formula', 'cascode')
                # Optional: augment with VN embedding for global circuit context
                self.dc_gain_use_vn = dc_gain_config.get('use_vn_context', False)
                dc_physics_dim = 35  # 14 gm + 14 gds + 7 formula intermediates
                self.dc_gain_use_volt_ctx = dc_gain_config.get('use_voltage_context', False)
                if self.dc_gain_use_volt_ctx:
                    dc_physics_dim += 28  # 14 Vgs + 14 Vds
                if self.dc_gain_use_vn:
                    dc_vn_proj_dim = dc_gain_config.get('vn_projection_dim', 0)
                    if dc_vn_proj_dim > 0:
                        self.dc_gain_vn_proj = nn.Linear(hidden_dim, dc_vn_proj_dim)
                        dc_physics_dim += dc_vn_proj_dim
                    else:
                        self.dc_gain_vn_proj = None
                        dc_physics_dim += hidden_dim
                # Optional: augment with projected device embeddings from sensitivity tower
                self.dc_gain_use_device_ctx = dc_gain_config.get('use_device_context', False)
                if self.dc_gain_use_device_ctx:
                    dc_dev_proj_dim = dc_gain_config.get('device_proj_dim', 32)
                    self.dc_gain_dev_proj = nn.Sequential(
                        nn.Linear(3 * mlp_input_dim, 128),
                        nn.LayerNorm(128),
                        get_activation(self.act_type),
                        nn.Linear(128, dc_dev_proj_dim),
                    )
                    dc_physics_dim += 14 * dc_dev_proj_dim
                dc_hidden = dc_gain_config.get('hidden_dim', 64)
                dc_num_layers = dc_gain_config.get('num_layers', 2)
                if dc_num_layers == 1:
                    self.dc_gain_head = nn.Sequential(
                        nn.Linear(dc_physics_dim, dc_hidden),
                        nn.LayerNorm(dc_hidden),
                        get_activation(self.act_type),
                        nn.Linear(dc_hidden, 1),
                    )
                else:
                    self.dc_gain_head = nn.Sequential(
                        nn.Linear(dc_physics_dim, dc_hidden),
                        nn.LayerNorm(dc_hidden),
                        get_activation(self.act_type),
                        nn.Linear(dc_hidden, dc_hidden // 2),
                        nn.LayerNorm(dc_hidden // 2),
                        get_activation(self.act_type),
                        nn.Linear(dc_hidden // 2, 1),
                    )
            elif self.dc_gain_mode == 'vn_mlp':
                # VN embedding → MLP (simplest learned DC gain)
                dc_hidden = dc_gain_config.get('hidden_dim', 128)
                self.dc_gain_head = nn.Sequential(
                    nn.Linear(hidden_dim, dc_hidden),
                    nn.LayerNorm(dc_hidden),
                    get_activation(self.act_type),
                    nn.Linear(dc_hidden, dc_hidden // 2),
                    nn.LayerNorm(dc_hidden // 2),
                    get_activation(self.act_type),
                    nn.Linear(dc_hidden // 2, 1),
                )
            elif self.dc_gain_mode == 'concat_mlp':
                # Flatten all 14 key MOSFET embeddings → MLP (no compression)
                self.register_buffer('_dc_mosfet_indices', torch.tensor(
                    [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                self.dc_gain_detach_ss = dc_gain_config.get('detach_ss', True)
                concat_input_dim = 14 * mlp_input_dim  # 14 × 149 = 2086
                self.dc_gain_head = nn.Sequential(
                    nn.Linear(concat_input_dim, 512),
                    nn.LayerNorm(512),
                    get_activation(self.act_type),
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    get_activation(self.act_type),
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    get_activation(self.act_type),
                    nn.Linear(128, 1),
                )
            elif self.dc_gain_mode == 'stage_pool':
                # Per-stage mean pooling → concat → MLP
                self.register_buffer('_dc_mosfet_indices', torch.tensor(
                    [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                self.register_buffer('_dc_bias_indices', torch.tensor(
                    [0, 1, 2, 3, 4], dtype=torch.long))
                self.dc_gain_detach_ss = dc_gain_config.get('detach_ss', True)
                stage_input_dim = 4 * mlp_input_dim  # 4 × 149 = 596
                self.dc_gain_head = nn.Sequential(
                    nn.Linear(stage_input_dim, 256),
                    nn.LayerNorm(256),
                    get_activation(self.act_type),
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    get_activation(self.act_type),
                    nn.Linear(128, 1),
                )
            elif self.dc_gain_mode == 'physics_cross_attn':
                # Physics formula + cross-attention correction from device embeddings
                # Same physics setup as 'physics' mode
                self.register_buffer('_dc_mosfet_indices', torch.tensor(
                    [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                self.dc_gain_detach_ss = dc_gain_config.get('detach_ss', False)
                if not hasattr(self, 'ss_gm_mean_buf'):
                    self.register_buffer('ss_gm_mean_buf', torch.tensor(0.0))
                    self.register_buffer('ss_gm_std_buf', torch.tensor(1.0))
                    self.register_buffer('ss_gds_mean_buf', torch.tensor(0.0))
                    self.register_buffer('ss_gds_std_buf', torch.tensor(1.0))
                r_in = float(dc_gain_config.get('r_in', 50000.0))
                r_f = float(dc_gain_config.get('r_f', 50000.0))
                self.register_buffer('_dc_r_in', torch.tensor(r_in))
                self.register_buffer('_dc_r_f', torch.tensor(r_f))
                self._dc_rout1_formula = dc_gain_config.get('rout1_formula', 'cascode')

                # Physics correction MLP (same as physics mode)
                dc_hidden = dc_gain_config.get('hidden_dim', 128)
                self.dc_gain_head = nn.Sequential(
                    nn.Linear(35, dc_hidden),
                    nn.LayerNorm(dc_hidden),
                    get_activation(self.act_type),
                    nn.Linear(dc_hidden, dc_hidden // 2),
                    nn.LayerNorm(dc_hidden // 2),
                    get_activation(self.act_type),
                    nn.Linear(dc_hidden // 2, 1),
                )

                # Cross-attention: physics features query device embeddings
                dc_attn_dim = 128  # 4 heads × 32
                n_physics_tokens = 35
                self.dc_ca_physics_proj = nn.Sequential(
                    nn.Linear(1, dc_attn_dim),
                    get_activation(self.act_type),
                )
                self.dc_ca_physics_type_embed = nn.Embedding(n_physics_tokens, dc_attn_dim)
                self.dc_ca_device_proj = nn.Linear(mlp_input_dim, dc_attn_dim)
                self.dc_ca_device_type_embed = nn.Embedding(14, dc_attn_dim)
                self.dc_ca_cross_attn = nn.MultiheadAttention(
                    embed_dim=dc_attn_dim, num_heads=4,
                    batch_first=True, dropout=0.0,
                )
                self.dc_ca_norm = nn.LayerNorm(dc_attn_dim)
                self.dc_ca_pool = nn.Sequential(
                    nn.Linear(n_physics_tokens * dc_attn_dim, 256),
                    nn.LayerNorm(256),
                    get_activation(self.act_type),
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    get_activation(self.act_type),
                    nn.Linear(128, 1),
                )
                # Learnable gate for correction strength
                self.dc_ca_gate = nn.Sequential(
                    nn.Linear(2, 1),
                    nn.Sigmoid(),
                )
            elif self.dc_gain_mode in ('physics_residual', 'physics_gated'):
                # Physics formula + learned correction (residual or gated)
                self.register_buffer('_dc_mosfet_indices', torch.tensor(
                    [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23], dtype=torch.long))
                self.dc_gain_detach_ss = dc_gain_config.get('detach_ss', False)
                if not hasattr(self, 'ss_gm_mean_buf'):
                    self.register_buffer('ss_gm_mean_buf', torch.tensor(0.0))
                    self.register_buffer('ss_gm_std_buf', torch.tensor(1.0))
                    self.register_buffer('ss_gds_mean_buf', torch.tensor(0.0))
                    self.register_buffer('ss_gds_std_buf', torch.tensor(1.0))
                r_in = float(dc_gain_config.get('r_in', 50000.0))
                r_f = float(dc_gain_config.get('r_f', 50000.0))
                self.register_buffer('_dc_r_in', torch.tensor(r_in))
                self.register_buffer('_dc_r_f', torch.tensor(r_f))
                self._dc_rout1_formula = dc_gain_config.get('rout1_formula', 'cascode')
                # Physics correction MLP
                dc_hidden = dc_gain_config.get('hidden_dim', 128)
                self.dc_gain_head = nn.Sequential(
                    nn.Linear(35, dc_hidden),
                    nn.LayerNorm(dc_hidden),
                    get_activation(self.act_type),
                    nn.Linear(dc_hidden, dc_hidden // 2),
                    nn.LayerNorm(dc_hidden // 2),
                    get_activation(self.act_type),
                    nn.Linear(dc_hidden // 2, 1),
                )
                # Learned MLP from 14 MOSFET embeddings
                concat_input_dim = 14 * mlp_input_dim  # 2086
                self.dc_gain_learned_head = nn.Sequential(
                    nn.Linear(concat_input_dim, 512),
                    nn.LayerNorm(512),
                    get_activation(self.act_type),
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    get_activation(self.act_type),
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),
                    get_activation(self.act_type),
                    nn.Linear(128, 1),
                )
                if self.dc_gain_mode == 'physics_gated':
                    self.dc_gain_gate = nn.Sequential(
                        nn.Linear(2, 1),
                        nn.Sigmoid(),
                    )
            elif self.dc_gain_mode == 'pool':
                # Simple mean+max pool of all nodes → MLP
                dc_hidden = dc_gain_config.get('hidden_dim', 128)
                self.dc_gain_head = nn.Sequential(
                    nn.Linear(2 * hidden_dim, dc_hidden),
                    nn.LayerNorm(dc_hidden),
                    get_activation(self.act_type),
                    nn.Linear(dc_hidden, 1),
                )
            else:
                # Cross-attention mode (original)
                dc_num_heads = dc_gain_config.get('num_heads', 4)
                dc_head_dim = dc_gain_config.get('head_dim', 32)
                dc_attn_dim = dc_num_heads * dc_head_dim  # default 128
                dc_dropout = dc_gain_config.get('dropout', 0.0)
                self.dc_gain_query = nn.Parameter(torch.randn(1, 1, dc_attn_dim) * 0.02)
                self.dc_gain_kv_proj = nn.Linear(mlp_input_dim, dc_attn_dim)
                self.dc_gain_cross_attn = nn.MultiheadAttention(
                    embed_dim=dc_attn_dim, num_heads=dc_num_heads,
                    batch_first=True, dropout=dc_dropout,
                )
                self.dc_gain_attn_norm = nn.LayerNorm(dc_attn_dim)
                self.dc_gain_head = nn.Sequential(
                    nn.Linear(dc_attn_dim, 128), nn.LayerNorm(128), get_activation(self.act_type),
                    nn.Linear(128, 1),
                )

        # Update MHA capture flag now that dc_gain_use_vn is known
        if getattr(self, 'dc_gain_use_vn', False) or self.dc_gain_mode == 'vn_mlp':
            self._need_mha_capture = True

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
            'ss_head_type': getattr(self, 'ss_head_type', 'mlp'),
            'ss_branch_layers': getattr(self, 'ss_branch_layers', 0),
            'use_device_pooling_current': self.use_device_pooling_current,
            'ss_from_state': self.ss_from_state,
            'predict_region': self.predict_region,
            'use_autograd_ss': self.use_autograd_ss,
            'use_iv_model': self.use_iv_model,
        }

    def set_normalization_stats(self, vdc_mean, vdc_std, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std,
                               current_mean=0.0, current_std=1.0):
        """Set normalization stats (stored as buffers for checkpoint compat)."""
        if hasattr(self, 'vdc_mean_buf'):
            self.vdc_mean_buf.fill_(vdc_mean)
            self.vdc_std_buf.fill_(vdc_std)
        if hasattr(self, 'ss_gm_mean_buf'):
            self.ss_gm_mean_buf.fill_(ss_gm_mean)
            self.ss_gm_std_buf.fill_(ss_gm_std)
            self.ss_gds_mean_buf.fill_(ss_gds_mean)
            self.ss_gds_std_buf.fill_(ss_gds_std)
        if hasattr(self, 'current_mean_buf'):
            self.current_mean_buf.fill_(current_mean)
            self.current_std_buf.fill_(current_std)

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
        # Loop attention (optional)
        loop_attn_layers: Optional[nn.ModuleList] = None,
        loop_attn_offset: int = 0,
        device_terminal_map: Optional[torch.Tensor] = None,
        loop_edge_index: Optional[torch.Tensor] = None,
        node_loop_edge_index: Optional[torch.Tensor] = None,
        capture_mha: bool = False,
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
            loop_attn_layers: Optional ModuleList of LoopAttention modules
            loop_attn_offset: Index offset into loop_attn_layers for this tower
            device_terminal_map: [D_total, 4] terminal indices per device
            loop_edge_index: [2, E_loop] device-level loop edges
            node_loop_edge_index: [2, E_node_loop] terminal-level loop edges (for level='node')
            capture_mha: If True, capture raw MHA outputs for AC head

        Returns:
            (layer_outputs, vn_emb, mha_outputs): list of tensors + updated VN embedding + MHA outputs
        """
        layer_outputs = [x]
        mha_outputs = []

        vn_is_mha = use_vn and self.virtual_node is not None and self.virtual_node.mode == 'mha'
        vn_is_default = use_vn and self.virtual_node is not None and self.virtual_node.mode == 'default' and vn_emb is not None

        for layer_idx, layer in enumerate(layers):
            # Global context: MHA or default VN broadcast
            if vn_is_mha:
                mha_out = self.virtual_node.global_mha(x, batch, layer_idx=layer_idx)
                x = x + mha_out
                if capture_mha:
                    mha_outputs.append(mha_out)
            elif vn_is_default:
                x = x + self.virtual_node.broadcast(vn_emb, batch, layer_idx=layer_idx)

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

            # Loop attention (between GENConv and VN update)
            if loop_attn_layers is not None and self.current_epoch >= self.loop_attn_warmup_epochs:
                la_layer = loop_attn_layers[loop_attn_offset + layer_idx]
                if la_layer.level == 'node' and node_loop_edge_index is not None:
                    x_loop = la_layer(x, device_terminal_map, node_loop_edge_index)
                elif device_terminal_map is not None and loop_edge_index is not None:
                    x_loop = la_layer(x, device_terminal_map, loop_edge_index)
                else:
                    x_loop = None
                if x_loop is not None:
                    # Gradual warmup: ramp alpha from 0→1 over warmup_duration
                    if self.loop_attn_warmup_duration > 0:
                        warmup_progress = self.current_epoch - self.loop_attn_warmup_epochs
                        alpha = min(1.0, warmup_progress / self.loop_attn_warmup_duration)
                        x = x + alpha * x_loop
                    else:
                        x = x + x_loop

            # Update VN from nodes (default mode only — MHA is stateless)
            if vn_is_default:
                vn_emb, _ = self.virtual_node(x, vn_emb, batch, num_graphs)

            layer_outputs.append(x)

        return layer_outputs, vn_emb, mha_outputs

    def _finalize_repr(self, jk: JKAggregation, layer_outputs: list, ref_layer: nn.Module, x_in: torch.Tensor) -> torch.Tensor:
        """Apply JK aggregation, final norm+act, and skip connection."""
        x = jk(layer_outputs)
        x = ref_layer.act(ref_layer.norm(x))
        if self.skip_connection:
            x = torch.cat([x, x_in], dim=-1)
        return x

    def _get_loop_data(self, data):
        """Compute loop_edge_index and device_terminal_map on-the-fly for batched data.

        Computed once and cached as buffers for subsequent calls.
        Returns: (device_loop_ei, dtm, node_loop_ei)
        """
        if hasattr(self, '_cached_loop_ei') and self._cached_loop_ei is not None:
            dev = data.mosfet_info.device
            # Replicate cached per-graph data for this batch
            num_graphs = data.ptr.shape[0] - 1
            N_per = (data.ptr[1] - data.ptr[0]).item()
            D_per = self._cached_dtm.shape[0]
            E_loop = self._cached_loop_ei.shape[1]
            # Debug: validate loop edge indices
            if E_loop > 0:
                D_total = D_per * num_graphs
                max_idx = self._cached_loop_ei.max().item()
                if max_idx >= D_per:
                    print(f"[LoopAttention WARNING] cached loop_ei max={max_idx} >= D_per={D_per}, clamping")
                    self._cached_loop_ei = self._cached_loop_ei.clamp(max=D_per - 1)

            if E_loop == 0:
                node_ei = self._cached_node_loop_ei.to(dev) if self._cached_node_loop_ei is not None else None
                return self._cached_loop_ei.to(dev), self._cached_dtm.repeat(num_graphs, 1).to(dev), node_ei

            dev_offsets = torch.arange(num_graphs, device=dev) * D_per
            loop_ei = self._cached_loop_ei.to(dev).repeat(1, num_graphs) + dev_offsets.repeat_interleave(E_loop).unsqueeze(0)

            dtm = self._cached_dtm.to(dev).repeat(num_graphs, 1)
            node_offsets = data.ptr[:-1].repeat_interleave(D_per)
            pad_mask = dtm >= 0
            dtm[pad_mask] += node_offsets.unsqueeze(1).expand_as(dtm)[pad_mask]

            # Replicate node-level loop edges for batch
            node_ei = None
            if self._cached_node_loop_ei is not None:
                E_node = self._cached_node_loop_ei.shape[1]
                if E_node > 0:
                    node_offsets_flat = data.ptr[:-1]  # [num_graphs]
                    node_ei = self._cached_node_loop_ei.to(dev).repeat(1, num_graphs) + node_offsets_flat.repeat_interleave(E_node).unsqueeze(0)
                else:
                    node_ei = self._cached_node_loop_ei.to(dev)

            return loop_ei, dtm, node_ei

        # First call: compute from single-graph topology
        from collections import defaultdict
        import networkx as nx

        mi = data.mosfet_info
        ptr = data.ptr
        # Extract single-graph mosfet_info
        mosfet_ptr = getattr(data, 'mosfet_ptr', None)
        if mosfet_ptr is not None:
            M_per = (mosfet_ptr[1] - mosfet_ptr[0]).item()
            mi_single = mi[:M_per]
        else:
            num_graphs = ptr.shape[0] - 1
            M_per = mi.shape[0] // num_graphs
            mi_single = mi[:M_per]

        N_per = (ptr[1] - ptr[0]).item()
        num_terminals = getattr(data, 'num_terminals', N_per)
        if isinstance(num_terminals, torch.Tensor):
            num_terminals = num_terminals[0].item() if num_terminals.dim() > 0 else num_terminals.item()

        device_terms = []
        device_nets = []
        # Per-terminal net mapping: term_to_net[terminal_idx] = net_idx
        term_to_net = {}

        for row in mi_single:
            gate_idx, drain_idx, source_idx = row[0].item(), row[1].item(), row[2].item()
            gate_net, drain_net, source_net = row[3].item(), row[4].item(), row[5].item()
            bulk_idx = drain_idx + 3
            terms = [gate_idx, drain_idx, source_idx]
            term_to_net[gate_idx] = gate_net
            term_to_net[drain_idx] = drain_net
            term_to_net[source_idx] = source_net
            if bulk_idx < num_terminals:
                terms.append(bulk_idx)
                term_to_net[bulk_idx] = source_net  # bulk shares net with source
            device_terms.append(terms)
            device_nets.append({gate_net, drain_net, source_net})

        # Also add resistors/capacitors if available
        ri = getattr(data, 'resistor_info', None)
        ri_ptr = getattr(data, 'resistor_ptr', None)
        if ri is not None and ri.shape[0] > 0:
            R_per = (ri_ptr[1] - ri_ptr[0]).item() if ri_ptr is not None else ri.shape[0] // (ptr.shape[0] - 1)
            for row in ri[:R_per]:
                t0, t1 = row[0].item(), row[1].item()
                n0, n1 = row[2].item(), row[3].item()
                device_terms.append([t0, t1])
                device_nets.append({n0, n1})
                term_to_net[t0] = n0
                term_to_net[t1] = n1

        ci = getattr(data, 'capacitor_info', None)
        ci_ptr = getattr(data, 'capacitor_ptr', None)
        if ci is not None and ci.shape[0] > 0:
            C_per = (ci_ptr[1] - ci_ptr[0]).item() if ci_ptr is not None else ci.shape[0] // (ptr.shape[0] - 1)
            for row in ci[:C_per]:
                t0, t1 = row[0].item(), row[1].item()
                n0, n1 = row[2].item(), row[3].item()
                device_terms.append([t0, t1])
                device_nets.append({n0, n1})
                term_to_net[t0] = n0
                term_to_net[t1] = n1

        num_devices = len(device_terms)
        if num_devices == 0:
            self._cached_loop_ei = torch.zeros((2, 0), dtype=torch.long)
            self._cached_dtm = torch.zeros((0, 4), dtype=torch.long)
            self._cached_node_loop_ei = None
            return self._cached_loop_ei.to(data.ptr.device), self._cached_dtm.to(data.ptr.device), None

        # Build device-level graph
        net_to_devs = defaultdict(set)
        for d, nets in enumerate(device_nets):
            for net in nets:
                net_to_devs[net].add(d)

        G = nx.Graph()
        G.add_nodes_from(range(num_devices))
        for net, devs in net_to_devs.items():
            devs = list(devs)
            for i in range(len(devs)):
                for j in range(i + 1, len(devs)):
                    G.add_edge(devs[i], devs[j])

        cycles = nx.cycle_basis(G)
        loop_edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                for j in range(i + 1, len(cycle)):
                    a, b = min(cycle[i], cycle[j]), max(cycle[i], cycle[j])
                    loop_edges.add((a, b))

        if loop_edges:
            src, dst = [], []
            for a, b in loop_edges:
                src.extend([a, b])
                dst.extend([b, a])
            loop_ei = torch.tensor([src, dst], dtype=torch.long)
        else:
            loop_ei = torch.zeros((2, 0), dtype=torch.long)

        max_terms = 4
        dtm = torch.full((num_devices, max_terms), -1, dtype=torch.long)
        for d, terms in enumerate(device_terms):
            for t, idx in enumerate(terms[:max_terms]):
                dtm[d, t] = idx

        # --- Node-level loop edges: expand device cycles to terminal pairs ---
        # Collect all loop-participating terminals per cycle, then full attention
        node_loop_edges = set()
        for cycle in cycles:  # reuse device-level cycles
            # Find terminals on shared nets for each adjacent device pair
            loop_terminals = set()
            for idx in range(len(cycle)):
                dev_a = cycle[idx]
                dev_b = cycle[(idx + 1) % len(cycle)]
                shared_nets = device_nets[dev_a] & device_nets[dev_b]
                for ta in device_terms[dev_a]:
                    if term_to_net.get(ta) in shared_nets:
                        loop_terminals.add(ta)
                for tb in device_terms[dev_b]:
                    if term_to_net.get(tb) in shared_nets:
                        loop_terminals.add(tb)
            # Full attention between all loop-participating terminals
            loop_terminals = list(loop_terminals)
            for i in range(len(loop_terminals)):
                for j in range(i + 1, len(loop_terminals)):
                    node_loop_edges.add((loop_terminals[i], loop_terminals[j]))
                    node_loop_edges.add((loop_terminals[j], loop_terminals[i]))

        if node_loop_edges:
            ns, nd = zip(*node_loop_edges)
            node_loop_ei = torch.tensor([list(ns), list(nd)], dtype=torch.long)
        else:
            node_loop_ei = torch.zeros((2, 0), dtype=torch.long)

        # Cache for reuse (topology is fixed)
        self._cached_loop_ei = loop_ei
        self._cached_dtm = dtm
        self._cached_node_loop_ei = node_loop_ei

        if not getattr(self, '_loop_edges_printed', False):
            print(f"[LoopAttention] Computed: {num_devices} devices, {len(cycles)} device cycles, {loop_ei.shape[1]} device loop edges")
            print(f"[LoopAttention] Node-level: {len(cycles)} device cycles expanded to {node_loop_ei.shape[1]} terminal loop edges")
            self._loop_edges_printed = True

        # Now replicate for this batch
        return self._get_loop_data(data)

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

        # Loop attention data
        _loop_attn = None
        _loop_ei = None
        _loop_dtm = None
        _node_loop_ei = None
        if self.use_loop_attention:
            _loop_attn = self.loop_attn_layers
            _loop_ei = getattr(data, 'loop_edge_index', None)
            _loop_dtm = getattr(data, 'device_terminal_map', None)
            # Compute on-the-fly if not precomputed (e.g., prebatched data)
            if _loop_ei is None or _loop_dtm is None:
                _loop_ei, _loop_dtm, _node_loop_ei = self._get_loop_data(data)

        # --- Backbone ---
        backbone_outputs, vn_emb, backbone_mha_outputs = self._run_layers(
            self.backbone, x, data.edge_index, edge_attr,
            batch=batch_vec, num_graphs=num_graphs,
            use_vn=True, vn_emb=vn_emb,
            loop_attn_layers=_loop_attn if (self.use_loop_attention and self.loop_attn_apply_to in ('backbone', 'all')) else None,
            loop_attn_offset=0,
            device_terminal_map=_loop_dtm,
            loop_edge_index=_loop_ei,
            node_loop_edge_index=_node_loop_ei,
            capture_mha=self._need_mha_capture,
        )
        backbone_repr = self._finalize_repr(
            self.backbone_jk, backbone_outputs, self.backbone[0], x_in,
        )
        # backbone_repr has skip: [hidden_dim + node_feature_dim] if skip_connection
        # Tower layers need hidden_dim input, so we extract just the JK part
        backbone_hidden = self.backbone_jk(backbone_outputs)
        backbone_hidden = self.backbone[0].act(self.backbone[0].norm(backbone_hidden))

        # --- Subcircuit DAG (enrich MOSFET terminal embeddings) ---
        _sc_embs = None  # stored for UGBW physics_dag readout
        _sc_embs_live = None  # non-detached version for physics_dag_residual
        if self.use_subcircuit_dag and self.current_epoch >= self.sc_dag_warmup_epochs:
            mi = data.mosfet_info.long()
            n_mosfets = mi.shape[0]
            n_graphs = data.ptr.shape[0] - 1
            m_per_graph = n_mosfets // n_graphs  # 24 for fixed topology
            m_ptr = getattr(data, 'mosfet_ptr', None)
            if m_ptr is not None:
                m_graph_idx = torch.bucketize(
                    torch.arange(n_mosfets, device=data.ptr.device),
                    m_ptr[1:].to(data.ptr.device), right=True)
            else:
                m_graph_idx = torch.arange(n_mosfets, device=data.ptr.device) // m_per_graph
            m_offsets = data.ptr[m_graph_idx]

            # Gather all 4 terminal embeddings per MOSFET
            gate_h = backbone_hidden[mi[:, 0] + m_offsets]    # [B*24, 128]
            drain_h = backbone_hidden[mi[:, 1] + m_offsets]   # [B*24, 128]
            source_h = backbone_hidden[mi[:, 2] + m_offsets]  # [B*24, 128]
            bulk_h = backbone_hidden[mi[:, 2] + 1 + m_offsets]  # [B*24, 128]
            all_terms = torch.stack([gate_h, drain_h, source_h, bulk_h], dim=1)  # [B*24, 4, 128]
            all_terms = all_terms.view(n_graphs, m_per_graph, 4, -1)  # [B, 24, 4, 128]

            # Attention pool per subcircuit
            sc_embs = []
            for sc_name in self._sc_group_names:
                dev_indices = self._sc_groups[sc_name]
                # Gather terminals for this subcircuit: [B, num_devs*4, 128]
                sc_terms = all_terms[:, dev_indices].reshape(n_graphs, -1, backbone_hidden.shape[-1])
                sc_emb = self.sc_attn_pool(sc_terms)  # [B, sc_dim]
                sc_embs.append(sc_emb)
            sc_embs = torch.stack(sc_embs, dim=1)  # [B, 7, sc_dim]

            # Sequential topological DAG with attention + GRU (DAGNN-style)
            sc_list = [sc_embs[:, i] for i in range(self._sc_num_groups)]

            for sc_idx, parent_indices in self._sc_topo_order:
                child_emb = sc_list[sc_idx]  # [B, sc_dim]
                parent_embs = torch.stack([sc_list[p] for p in parent_indices], dim=1)  # [B, P, sc_dim]

                # Attention over parents
                q = self.sc_dag_W_q(child_emb).unsqueeze(1).expand_as(parent_embs)  # [B, P, sc_dim]
                k = self.sc_dag_W_k(parent_embs)  # [B, P, sc_dim]
                v = self.sc_dag_W_v(parent_embs)  # [B, P, sc_dim]
                scores = self.sc_dag_attn_a(F.leaky_relu(torch.cat([q, k], dim=-1), 0.2)).squeeze(-1)  # [B, P]
                alpha = torch.softmax(scores, dim=-1)  # [B, P]
                message = (alpha.unsqueeze(-1) * v).sum(dim=1)  # [B, sc_dim]

                # GRU update
                sc_list[sc_idx] = self.sc_dag_gru(message, child_emb)  # [B, sc_dim]

            sc_embs = torch.stack(sc_list, dim=1)  # [B, 7, sc_dim]
            _sc_embs = sc_embs.detach() if getattr(self, 'ac_detach', True) else sc_embs  # detach unless AC needs gradients
            _sc_embs_live = sc_embs  # always keep live version for physics_dag_residual correction

            # Compute warmup alpha for gradual DAG introduction
            if self.sc_dag_warmup_duration > 0:
                warmup_progress = self.current_epoch - self.sc_dag_warmup_epochs
                sc_alpha = min(1.0, warmup_progress / self.sc_dag_warmup_duration)
            else:
                sc_alpha = 1.0

            # Scatter back to MOSFET terminals with gated fusion
            # assignment: [24] → subcircuit index per device
            for dev_idx in range(m_per_graph):
                sc_idx = self._sc_assignment[dev_idx].item()
                if sc_idx < 0:
                    continue
                sc_emb_for_dev = sc_embs[:, sc_idx]  # [B, sc_dim]
                # Apply to all 4 terminals (G, D, S, B) of this device
                for term_col in [0, 1, 2]:
                    term_global = mi[:, term_col] + m_offsets  # [B*24] but we want this device only
                    # Select only this device across all graphs
                    dev_mask = torch.arange(n_mosfets, device=mi.device) % m_per_graph == dev_idx
                    term_idx = mi[dev_mask, term_col] + m_offsets[dev_mask]
                    sc_for_term = sc_emb_for_dev  # [B, sc_dim]
                    bh_at_term = backbone_hidden[term_idx]  # [B, 128]
                    gate = torch.sigmoid(self.sc_gate_proj(
                        torch.cat([bh_at_term, sc_for_term], dim=-1)))  # [B, 128]
                    backbone_hidden[term_idx] = sc_alpha * gate * sc_for_term + (1 - sc_alpha * gate) * bh_at_term
                # Bulk terminal (source + 1)
                dev_mask = torch.arange(n_mosfets, device=mi.device) % m_per_graph == dev_idx
                bulk_idx = mi[dev_mask, 2] + 1 + m_offsets[dev_mask]
                bh_at_bulk = backbone_hidden[bulk_idx]
                gate = torch.sigmoid(self.sc_gate_proj(
                    torch.cat([bh_at_bulk, sc_emb_for_dev], dim=-1)))
                backbone_hidden[bulk_idx] = sc_alpha * gate * sc_emb_for_dev + (1 - sc_alpha * gate) * bh_at_bulk

        # --- gm/Id auxiliary prediction (backbone-level) ---
        _gm_id_pred = None
        if self.predict_gm_id and self.gm_id_head is not None:
            mosfet_info_l = data.mosfet_info.long()
            num_mosfets = mosfet_info_l.shape[0]
            num_graphs = data.ptr.shape[0] - 1
            mosfet_ptr = getattr(data, 'mosfet_ptr', None)
            if mosfet_ptr is not None:
                mg_idx = torch.bucketize(
                    torch.arange(num_mosfets, device=data.ptr.device),
                    mosfet_ptr[1:].to(data.ptr.device), right=True)
            else:
                mg_idx = torch.arange(num_mosfets, device=data.ptr.device) // (num_mosfets // num_graphs)
            bb_offsets = data.ptr[mg_idx]
            gate_bb = backbone_repr[mosfet_info_l[:, 0] + bb_offsets]
            drain_bb = backbone_repr[mosfet_info_l[:, 1] + bb_offsets]
            source_bb = backbone_repr[mosfet_info_l[:, 2] + bb_offsets]
            gm_id_input = torch.cat([gate_bb, drain_bb, source_bb], dim=-1)
            _gm_id_pred = self.gm_id_head(gm_id_input).squeeze(-1)

        # --- Scatter detached gm/Id as feature for towers ---
        tower_input = backbone_hidden
        if self.gmid_as_feature and _gm_id_pred is not None:
            gmid_detached = _gm_id_pred.detach()
            gmid_node_feat = torch.zeros(backbone_hidden.shape[0], 1, device=backbone_hidden.device)
            for term_col in [0, 1, 2]:  # G, D, S
                term_idx = mosfet_info_l[:, term_col] + bb_offsets
                gmid_node_feat[term_idx, 0] = gmid_detached
            bulk_idx = mosfet_info_l[:, 2] + 1 + bb_offsets  # B = S + 1
            gmid_node_feat[bulk_idx, 0] = gmid_detached
            tower_input = self.gmid_feature_proj(
                torch.cat([backbone_hidden, gmid_node_feat], dim=-1)
            )

        # --- State Tower ---
        _state_loop = _loop_attn if (self.use_loop_attention and getattr(self, 'loop_attn_apply_to', 'backbone') == 'all') else None
        _state_offset = getattr(self, '_loop_attn_backbone_count', 0)
        if self.state_tower_num_layers > 0:
            state_outputs, _, _ = self._run_layers(
                self.state_tower, tower_input, data.edge_index, edge_attr,
                loop_attn_layers=_state_loop,
                loop_attn_offset=_state_offset,
                device_terminal_map=_loop_dtm,
                loop_edge_index=_loop_ei,
                node_loop_edge_index=_node_loop_ei,
            )
            if self.unified_state_jk:
                unified_state = backbone_outputs + state_outputs[1:]  # [x_in, bb0..bb5, tw0, tw1]
                state_hidden = self.state_jk(unified_state)
            else:
                state_hidden = self.state_jk(state_outputs)
            state_hidden = self.state_tower[0].act(self.state_tower[0].norm(state_hidden))
        else:
            state_hidden = tower_input

        # State repr with skip connection for prediction heads
        state_repr = torch.cat([state_hidden, x_in], dim=-1) if self.skip_connection else state_hidden

        # State predictions
        result = {}
        if _gm_id_pred is not None:
            result['mosfet_gm_id_pred'] = _gm_id_pred
        result['node_voltages'] = self.voltage_head(state_repr).squeeze(-1)
        result['node_embeddings'] = state_repr

        # Vgs/Vds prediction from terminal embeddings
        if self.predict_vgsvds and self.vgsvds_head is not None:
            mi = data.mosfet_info.long()
            num_mosfets = mi.shape[0]
            n_graphs = data.ptr.shape[0] - 1
            m_ptr = getattr(data, 'mosfet_ptr', None)
            if m_ptr is not None:
                mg_idx = torch.bucketize(torch.arange(num_mosfets, device=data.ptr.device),
                                          m_ptr[1:].to(data.ptr.device), right=True)
            else:
                mg_idx = torch.arange(num_mosfets, device=data.ptr.device) // (num_mosfets // n_graphs)
            vd_offsets = data.ptr[mg_idx]
            g_emb = state_repr[mi[:, 0] + vd_offsets]
            d_emb = state_repr[mi[:, 1] + vd_offsets]
            s_emb = state_repr[mi[:, 2] + vd_offsets]
            vgsvds_input = torch.cat([g_emb, d_emb, s_emb], dim=-1)
            result['vgsvds_pred'] = self.vgsvds_head(vgsvds_input)  # [M, 2]

        if self.predict_currents:
            if self.use_device_pooling_current and self.device_current_head is not None:
                # Compute voltage inputs for autograd SS if enabled
                mosfet_voltages = None
                autograd_vgs = None
                autograd_vds = None
                if self.use_autograd_ss:
                    mosfet_info_l = data.mosfet_info.long()
                    mosfet_ptr_l = getattr(data, 'mosfet_ptr', None)
                    num_mosfets_l = mosfet_info_l.shape[0]
                    num_graphs_l = data.ptr.shape[0] - 1
                    if mosfet_ptr_l is not None:
                        mg_idx_l = torch.bucketize(
                            torch.arange(num_mosfets_l, device=data.ptr.device),
                            mosfet_ptr_l[1:].to(data.ptr.device), right=True)
                    else:
                        mg_idx_l = torch.arange(num_mosfets_l, device=data.ptr.device) // (num_mosfets_l // num_graphs_l)
                    offsets_l = data.ptr[mg_idx_l]

                    v_pred = result['node_voltages'].detach()
                    v_gate = v_pred[mosfet_info_l[:, 0] + offsets_l]
                    v_drain = v_pred[mosfet_info_l[:, 1] + offsets_l]
                    v_source = v_pred[mosfet_info_l[:, 2] + offsets_l]
                    v_bulk = v_pred[mosfet_info_l[:, 2] + 1 + offsets_l]

                    # Z-score normalize voltage differences
                    vgs = (v_gate - v_source) / self.vdc_std_buf.clamp(min=1e-6)
                    vds = (v_drain - v_source) / self.vdc_std_buf.clamp(min=1e-6)
                    vbs = (v_bulk - v_source) / self.vdc_std_buf.clamp(min=1e-6)

                    # Detach and enable grad for autograd differentiation
                    autograd_vgs = vgs.detach().requires_grad_(True)
                    autograd_vds = vds.detach().requires_grad_(True)
                    vbs_detached = vbs.detach()

                    mosfet_voltages = torch.stack([autograd_vgs, autograd_vds, vbs_detached], dim=-1)

                node_currents, device_mask, mosfet_current = self.device_current_head(
                    x=state_repr,
                    mosfet_info=getattr(data, 'mosfet_info', None),
                    resistor_info=getattr(data, 'resistor_info', None),
                    capacitor_info=getattr(data, 'capacitor_info', None),
                    vsource_info=getattr(data, 'vsource_info', None),
                    isource_info=getattr(data, 'isource_info', None),
                    num_nodes=state_repr.size(0),
                    ptr=getattr(data, 'ptr', None),
                    mosfet_ptr=getattr(data, 'mosfet_ptr', None),
                    resistor_ptr=getattr(data, 'resistor_ptr', None),
                    capacitor_ptr=getattr(data, 'capacitor_ptr', None),
                    isource_ptr=getattr(data, 'isource_ptr', None),
                    mosfet_voltages=mosfet_voltages,
                )
                result['node_currents'] = node_currents
                result['device_current_mask'] = device_mask

                # Autograd gm/gds from current head output
                if self.use_autograd_ss and mosfet_current is not None and autograd_vgs is not None:
                    # Denormalize from z-scored log10 to raw log10(|ID|)
                    log_abs_id = mosfet_current * self.current_std_buf + self.current_mean_buf

                    # Differentiate log10(|ID|) w.r.t. voltages in log-space
                    d_log_id_dvgs = torch.autograd.grad(
                        log_abs_id.sum(), autograd_vgs,
                        create_graph=self.training, retain_graph=True,
                    )[0]
                    d_log_id_dvds = torch.autograd.grad(
                        log_abs_id.sum(), autograd_vds,
                        create_graph=self.training,
                    )[0]

                    # Reconstruct log10(gm) and log10(gds) in log-space
                    # gm = |ID| * ln(10) * |d(log10|ID|)/dVGS|
                    # log10(gm) = log_abs_id + log10(ln(10)) + log10(|d_log_id_dvgs|) - log10(vdc_std)
                    LOG10_LN10 = 0.36221568869946325
                    log10_vdc_std = torch.log10(self.vdc_std_buf.clamp(min=1e-6))
                    log10_gm = log_abs_id + LOG10_LN10 + torch.log10(d_log_id_dvgs.abs().clamp(min=1e-20)) - log10_vdc_std
                    log10_gds = log_abs_id + LOG10_LN10 + torch.log10(d_log_id_dvds.abs().clamp(min=1e-20)) - log10_vdc_std

                    # Z-score for loss compatibility
                    result['mosfet_gm_pred'] = (log10_gm - self.ss_gm_mean_buf) / self.ss_gm_std_buf
                    result['mosfet_gds_pred'] = (log10_gds - self.ss_gds_mean_buf) / self.ss_gds_std_buf
            elif self.current_head is not None:
                result['node_currents'] = self.current_head(state_repr).squeeze(-1)
                # Auxiliary currents from intermediate state tower layer for deep KCL
                if self.aux_current_head is not None:
                    result['aux_node_currents'] = self.aux_current_head(state_outputs[1]).squeeze(-1)

        # --- Per-MOSFET indexing (shared by Vov head and sensitivity tower) ---
        mosfet_info = data.mosfet_info.long()
        mosfet_ptr = getattr(data, 'mosfet_ptr', None)
        num_mosfets = mosfet_info.shape[0]
        num_graphs = data.ptr.shape[0] - 1
        if mosfet_ptr is not None:
            mg_idx = torch.bucketize(
                torch.arange(num_mosfets, device=data.ptr.device),
                mosfet_ptr[1:].to(data.ptr.device), right=True)
        else:
            mg_idx = torch.arange(num_mosfets, device=data.ptr.device) // (num_mosfets // num_graphs)
        offsets = data.ptr[mg_idx]

        # --- Vov prediction head (from state tower, detached) ---
        if self.predict_vov:
            state_repr_d = state_repr.detach()
            gate_st = state_repr_d[mosfet_info[:, 0] + offsets]
            source_st = state_repr_d[mosfet_info[:, 2] + offsets]
            bulk_st = state_repr_d[mosfet_info[:, 2] + 1 + offsets]
            vov_input = torch.cat([gate_st, source_st, bulk_st], dim=-1)
            result['mosfet_vov_pred'] = self.vov_head(vov_input).squeeze(-1)

        # --- Vth prediction head (from state tower, detached) ---
        if self.predict_vth:
            state_repr_d = state_repr.detach() if not self.predict_vov else state_repr_d
            gate_st = state_repr_d[mosfet_info[:, 0] + offsets]
            drain_st = state_repr_d[mosfet_info[:, 1] + offsets]
            source_st = state_repr_d[mosfet_info[:, 2] + offsets]
            bulk_st = state_repr_d[mosfet_info[:, 2] + 1 + offsets]
            vth_input = torch.cat([gate_st, drain_st, source_st, bulk_st], dim=-1)
            result['mosfet_vth_pred'] = self.vth_head(vth_input).squeeze(-1)

        # --- Sensitivity Tower ---
        if self.has_sensitivity_tower:
            if self.ss_from_state:
                # Read gm/gds from state tower output directly (no sensitivity tower GNN)
                sens_repr = state_repr
            else:
                # Optionally condition on state tower embeddings
                if self.state_conditioned_sens:
                    state_for_sens = state_hidden.detach() if self.detach_state_for_sens else state_hidden
                    sens_input = self.state_sens_proj(torch.cat([tower_input, state_for_sens], dim=-1))
                else:
                    sens_input = tower_input

                if self.sensitivity_tower_num_layers > 0:
                    _sens_loop = _loop_attn if (self.use_loop_attention and self.loop_attn_apply_to in ('all', 'sensitivity')) else None
                    _sens_offset = 0 if (self.use_loop_attention and self.loop_attn_apply_to == 'sensitivity') else getattr(self, '_loop_attn_backbone_count', 0) + getattr(self, '_loop_attn_state_count', 0)
                    sens_outputs, _, _ = self._run_layers(
                        self.sensitivity_tower, sens_input, data.edge_index, edge_attr,
                        loop_attn_layers=_sens_loop,
                        loop_attn_offset=_sens_offset,
                        device_terminal_map=_loop_dtm,
                        loop_edge_index=_loop_ei,
                        node_loop_edge_index=_node_loop_ei,
                    )
                else:
                    sens_outputs = [sens_input]

            # Compute voltage context scalars (Vgs, Vds, Vbs) if enabled
            if self.ss_voltage_context:
                if self.predict_vgsvds and 'vgsvds_pred' in result:
                    # Use predicted Vgs/Vds from dedicated head, Vbs from net voltages
                    vgsvds = result['vgsvds_pred'].detach() if self.vgsvds_detach else result['vgsvds_pred']
                    vgs_ctx = vgsvds[:, 0]
                    vds_ctx = vgsvds[:, 1]
                    v_pred = result['node_voltages'].detach()
                    v_source = v_pred[mosfet_info[:, 2] + offsets]
                    v_bulk = v_pred[mosfet_info[:, 2] + 1 + offsets]
                    vbs_ctx = v_bulk - v_source
                    volt_ctx = torch.stack([vgs_ctx, vds_ctx, vbs_ctx], dim=-1)
                else:
                    v_pred = result['node_voltages'] if not self.detach_state_for_sens else result['node_voltages'].detach()
                    v_gate = v_pred[mosfet_info[:, 0] + offsets]
                    v_drain = v_pred[mosfet_info[:, 1] + offsets]
                    v_source = v_pred[mosfet_info[:, 2] + offsets]
                    v_bulk = v_pred[mosfet_info[:, 2] + 1 + offsets]
                    volt_ctx = torch.stack([v_gate - v_source, v_drain - v_source, v_bulk - v_source], dim=-1)

            # Compute current context (ID at drain) if enabled
            if self.ss_current_context:
                i_pred = result['node_currents'] if not self.detach_state_for_sens else result['node_currents'].detach()
                i_drain = i_pred[mosfet_info[:, 1] + offsets].unsqueeze(-1)

            # Extract W/L context from raw node features at gate terminal
            if self.ss_wl_context:
                raw_gate_feats = data.x[mosfet_info[:, 0] + offsets]
                wl_ctx = raw_gate_feats[:, :3]  # [w_norm, l_norm, wl_ratio_norm]

            # Vth context from predicted Vth (detached)
            if self.ss_vth_context and 'mosfet_vth_pred' in result:
                vth_ctx = result['mosfet_vth_pred'].detach().unsqueeze(-1)  # [M, 1]

            # Region prediction (CORAL ordinal) — computed before SS heads for conditioning
            region_probs = None
            region_probs_detached = None
            if self.predict_region and self.region_head is not None:
                # Finalize sens_repr first if needed (for region head only, before SS branching)
                if not self.ss_from_state and self.gm_branch is None:
                    # Standard path: need to finalize sens_repr before using it
                    if self.sensitivity_tower_num_layers > 0:
                        _sens_jk_inputs = (backbone_outputs + sens_outputs[1:]) if getattr(self, 'unified_sensitivity_jk', False) else sens_outputs
                        sens_repr = self._finalize_repr(
                            self.sensitivity_jk, _sens_jk_inputs, self.sensitivity_tower[0], x_in,
                        )
                    else:
                        sens_repr = torch.cat([sens_outputs[0], x_in], dim=-1) if self.skip_connection else sens_outputs[0]
                    _sens_repr_finalized = True
                else:
                    _sens_repr_finalized = False

                r_gate = sens_repr[mosfet_info[:, 0] + offsets]
                r_drain = sens_repr[mosfet_info[:, 1] + offsets]
                r_source = sens_repr[mosfet_info[:, 2] + offsets]
                r_parts = [r_gate, r_drain, r_source]
                if self.ss_include_bulk:
                    r_parts.append(sens_repr[mosfet_info[:, 2] + 1 + offsets])
                region_input = torch.cat(r_parts, dim=-1)
                region_logits = self.region_head(region_input)  # [num_mosfets, 2]
                result['mosfet_region_logits'] = region_logits

                # Derive soft probabilities: P(cutoff), P(triode), P(saturation)
                sigma = torch.sigmoid(region_logits)
                region_probs = torch.stack([
                    1 - sigma[:, 0],              # P(cutoff)
                    sigma[:, 0] - sigma[:, 1],    # P(triode)
                    sigma[:, 1],                  # P(saturation)
                ], dim=-1).clamp(min=0)  # clamp for numerical stability
                result['mosfet_region_probs'] = region_probs.detach()
                region_probs_detached = region_probs.detach()
            else:
                _sens_repr_finalized = False

            if self.gm_branch is not None and not self.ss_from_state:
                # Y-shaped: run separate gm/gds GNN branches from shared tower output
                branch_input = sens_outputs[-1]  # last hidden state from shared tower

                gm_outputs, _, _ = self._run_layers(self.gm_branch, branch_input, data.edge_index, edge_attr)
                gm_repr = self._finalize_repr(self.gm_branch_jk, gm_outputs, self.gm_branch[0], x_in)

                gds_outputs, _, _ = self._run_layers(self.gds_branch, branch_input, data.edge_index, edge_attr)
                gds_repr = self._finalize_repr(self.gds_branch_jk, gds_outputs, self.gds_branch[0], x_in)

                # Build gm input from gm branch
                gm_parts = [gm_repr[mosfet_info[:, 0] + offsets],
                             gm_repr[mosfet_info[:, 1] + offsets],
                             gm_repr[mosfet_info[:, 2] + offsets]]
                if self.ss_include_bulk:
                    gm_parts.append(gm_repr[mosfet_info[:, 2] + 1 + offsets])
                if self.ss_voltage_context:
                    gm_parts.append(volt_ctx)
                if self.ss_current_context:
                    gm_parts.append(i_drain)
                if self.ss_vov_context:
                    gm_parts.append(result['mosfet_vov_pred'].detach().unsqueeze(-1))
                if self.ss_region_context and region_probs_detached is not None:
                    gm_parts.append(region_probs_detached)
                if self.ss_wl_context:
                    gm_parts.append(wl_ctx)
                if self.ss_vth_context and 'mosfet_vth_pred' in result:
                    gm_parts.append(vth_ctx)
                gm_input = torch.cat(gm_parts, dim=-1)
                result['mosfet_gm_pred'] = self.gm_head(gm_input).squeeze(-1)

                # Build gds input from gds branch
                gds_parts = [gds_repr[mosfet_info[:, 0] + offsets],
                              gds_repr[mosfet_info[:, 1] + offsets],
                              gds_repr[mosfet_info[:, 2] + offsets]]
                if self.ss_include_bulk:
                    gds_parts.append(gds_repr[mosfet_info[:, 2] + 1 + offsets])
                if self.ss_voltage_context:
                    gds_parts.append(volt_ctx)
                if self.ss_current_context:
                    gds_parts.append(i_drain)
                if self.ss_vov_context:
                    gds_parts.append(result['mosfet_vov_pred'].detach().unsqueeze(-1))
                if self.ss_region_context and region_probs_detached is not None:
                    gds_parts.append(region_probs_detached)
                if self.ss_wl_context:
                    gds_parts.append(wl_ctx)
                if self.ss_vth_context and 'mosfet_vth_pred' in result:
                    gds_parts.append(vth_ctx)
                gds_input = torch.cat(gds_parts, dim=-1)
                result['mosfet_gds_pred'] = self.gds_head(gds_input).squeeze(-1)
            else:
                # Standard path: shared sensitivity tower (or state tower) → separate MLP heads
                if not self.ss_from_state and not _sens_repr_finalized:
                    if self.sensitivity_tower_num_layers > 0:
                        _sens_jk_inputs = (backbone_outputs + sens_outputs[1:]) if getattr(self, 'unified_sensitivity_jk', False) else sens_outputs
                        sens_repr = self._finalize_repr(
                            self.sensitivity_jk, _sens_jk_inputs, self.sensitivity_tower[0], x_in,
                        )
                    else:
                        sens_repr = torch.cat([sens_outputs[0], x_in], dim=-1) if self.skip_connection else sens_outputs[0]

                gate_emb = sens_repr[mosfet_info[:, 0] + offsets]
                drain_emb = sens_repr[mosfet_info[:, 1] + offsets]
                source_emb = sens_repr[mosfet_info[:, 2] + offsets]

                # Cross-attention between terminals before readout
                if self.ss_cross_attention:
                    terminal_stack = torch.stack([gate_emb, drain_emb, source_emb], dim=1)
                    attn_out, _ = self.ss_cross_attn(terminal_stack, terminal_stack, terminal_stack)
                    attn_out = self.ss_cross_attn_norm(attn_out + terminal_stack)
                    gate_emb = attn_out[:, 0]
                    drain_emb = attn_out[:, 1]
                    source_emb = attn_out[:, 2]

                if self.ss_head_type in ('attention', 'gated'):
                    bulk_emb = sens_repr[mosfet_info[:, 2] + 1 + offsets]
                    ctx_parts = []
                    if self.ss_voltage_context:
                        ctx_parts.append(volt_ctx)
                    if self.ss_current_context:
                        ctx_parts.append(i_drain)
                    if self.ss_vov_context:
                        ctx_parts.append(result['mosfet_vov_pred'].detach().unsqueeze(-1))
                    if self.ss_wl_context:
                        ctx_parts.append(wl_ctx)
                    if self.ss_vth_context and 'mosfet_vth_pred' in result:
                        ctx_parts.append(vth_ctx)
                    ctx_scalars = torch.cat(ctx_parts, dim=-1) if ctx_parts else None
                    head = self.ss_attn_head if self.ss_head_type == 'attention' else self.ss_gated_head
                    gm_pred, gds_pred = head(gate_emb, drain_emb, source_emb, bulk_emb, ctx_scalars)
                    result['mosfet_gm_pred'] = gm_pred
                    result['mosfet_gds_pred'] = gds_pred
                elif self.ss_physics_pool:
                    gm_parts = [gate_emb, source_emb]
                    gds_parts = [drain_emb, source_emb]
                    if self.ss_voltage_context:
                        gm_parts.append(volt_ctx)
                        gds_parts.append(volt_ctx)
                    if self.ss_current_context:
                        gm_parts.append(i_drain)
                        gds_parts.append(i_drain)
                    if self.ss_vov_context:
                        vov_pred = result['mosfet_vov_pred'].detach().unsqueeze(-1)
                        gm_parts.append(vov_pred)
                        gds_parts.append(vov_pred)
                    if self.ss_region_context and region_probs_detached is not None:
                        gm_parts.append(region_probs_detached)
                        gds_parts.append(region_probs_detached)
                    if self.ss_wl_context:
                        gm_parts.append(wl_ctx)
                        gds_parts.append(wl_ctx)
                    if self.ss_vth_context and 'mosfet_vth_pred' in result:
                        gm_parts.append(vth_ctx)
                        gds_parts.append(vth_ctx)
                    gm_repr = torch.cat(gm_parts, dim=-1)
                    gds_repr = torch.cat(gds_parts, dim=-1)
                    result['mosfet_gm_pred'] = self.gm_head(gm_repr).squeeze(-1)
                    result['mosfet_gds_pred'] = self.gds_head(gds_repr).squeeze(-1)
                else:
                    if getattr(self, 'ss_drain_only', False):
                        parts = [drain_emb]
                    else:
                        parts = [gate_emb, drain_emb, source_emb]
                    if not getattr(self, 'ss_drain_only', False) and self.ss_include_bulk:
                        bulk_emb = sens_repr[mosfet_info[:, 2] + 1 + offsets]
                        parts.append(bulk_emb)
                    if self.ss_diff_features:
                        parts.append(gate_emb - source_emb)
                    if self.ss_pairwise:
                        parts.append(self.pw_gd(gate_emb * drain_emb))
                        parts.append(self.pw_gs(gate_emb * source_emb))
                        parts.append(self.pw_ds(drain_emb * source_emb))
                    if self.ss_voltage_context:
                        parts.append(volt_ctx)
                    if self.ss_current_context:
                        parts.append(i_drain)
                    if self.ss_vov_context:
                        vov_pred = result['mosfet_vov_pred'].detach().unsqueeze(-1)
                        parts.append(vov_pred)
                    if self.ss_region_context and region_probs_detached is not None:
                        parts.append(region_probs_detached)
                    if self.ss_wl_context:
                        parts.append(wl_ctx)
                    if self.ss_vth_context and 'mosfet_vth_pred' in result:
                        parts.append(vth_ctx)
                    sens_concat = torch.cat(parts, dim=-1)

                    if self.state_context_mlp is not None:
                        bulk_term_idx = mosfet_info[:, 2] + 1
                        gate_state = state_repr[mosfet_info[:, 0] + offsets]
                        drain_state = state_repr[mosfet_info[:, 1] + offsets]
                        source_state = state_repr[mosfet_info[:, 2] + offsets]
                        bulk_state = state_repr[bulk_term_idx + offsets]
                        state_concat = torch.cat([gate_state, drain_state, source_state, bulk_state], dim=-1)
                        state_summary = self.state_context_mlp(state_concat).detach()
                        ss_input = torch.cat([sens_concat, state_summary], dim=-1)
                    else:
                        ss_input = sens_concat

                    if getattr(self, 'ss_predict_gm_id', False) and self.gm_id_ss_head is not None:
                        result['mosfet_gm_id_pred'] = self.gm_id_ss_head(ss_input).squeeze(-1)
                    if self.ss_moe and region_probs is not None:
                        # MoE: 3 expert heads per quantity, soft-routed by region probs
                        gm_expert_out = torch.stack(
                            [exp(ss_input).squeeze(-1) for exp in self.gm_experts], dim=-1
                        )  # [M, 3]
                        gds_expert_out = torch.stack(
                            [exp(ss_input).squeeze(-1) for exp in self.gds_experts], dim=-1
                        )  # [M, 3]
                        # Non-detached routing: SS loss flows through region probs
                        result['mosfet_gm_pred'] = (gm_expert_out * region_probs).sum(dim=-1)
                        result['mosfet_gds_pred'] = (gds_expert_out * region_probs).sum(dim=-1)
                    elif getattr(self, 'use_subcircuit_heads', False):
                        # Subcircuit-conditioned: route each MOSFET through its subcircuit's head
                        gm_preds = torch.zeros(num_mosfets, device=ss_input.device)
                        gds_preds = torch.zeros(num_mosfets, device=ss_input.device)
                        for sc_name, sc_indices in self._subcircuit_groups.items():
                            local_mask = torch.zeros(24, dtype=torch.bool, device=ss_input.device)
                            for idx in sc_indices:
                                local_mask[idx] = True
                            batch_mask = local_mask.repeat(num_graphs)
                            if batch_mask.any():
                                sc_input = ss_input[batch_mask]
                                gm_preds[batch_mask] = self.subcircuit_gm_heads[sc_name](sc_input).squeeze(-1)
                                gds_preds[batch_mask] = self.subcircuit_gds_heads[sc_name](sc_input).squeeze(-1)
                        result['mosfet_gm_pred'] = gm_preds
                        result['mosfet_gds_pred'] = gds_preds
                    else:
                        result['mosfet_gm_pred'] = self.gm_head(ss_input).squeeze(-1)
                        result['mosfet_gds_pred'] = self.gds_head(ss_input).squeeze(-1)

        # --- Differentiable I-V model (autograd gm/gds via separate MLP) ---
        if self.use_iv_model:
            # 1. Denormalize predicted voltages to raw volts
            v_pred = result['node_voltages']  # z-scored
            v_raw = v_pred * self.vdc_std_buf + self.vdc_mean_buf

            # 2. Per-MOSFET terminal voltages
            v_gate = v_raw[mosfet_info[:, 0] + offsets]
            v_drain = v_raw[mosfet_info[:, 1] + offsets]
            v_source = v_raw[mosfet_info[:, 2] + offsets]
            v_bulk = v_raw[mosfet_info[:, 2] + 1 + offsets]

            # 3. Compute VGS, VDS, VBS — z-score normalize, preserve sign
            vgs = (v_gate - v_source) / self.vdc_std_buf.clamp(min=1e-6)
            vds = (v_drain - v_source) / self.vdc_std_buf.clamp(min=1e-6)
            vbs = (v_bulk - v_source) / self.vdc_std_buf.clamp(min=1e-6)

            # 4. Detach voltages (break gradient to voltage tower) and enable grad for autograd
            if self.iv_detach_voltages:
                vgs = vgs.detach().requires_grad_(True)
                vds = vds.detach().requires_grad_(True)
                vbs = vbs.detach()
            else:
                vgs = vgs.requires_grad_(True)
                vds = vds.requires_grad_(True)

            # 5. Device embedding from state tower (optionally non-detached)
            gate_emb = state_repr[mosfet_info[:, 0] + offsets]
            drain_emb = state_repr[mosfet_info[:, 1] + offsets]
            source_emb = state_repr[mosfet_info[:, 2] + offsets]
            if self.iv_detach_embeddings:
                gate_emb = gate_emb.detach()
                drain_emb = drain_emb.detach()
                source_emb = source_emb.detach()
            device_emb = self.iv_embed_proj(torch.cat([gate_emb, drain_emb, source_emb], dim=-1))

            # 6. Forward through I-V model
            iv_out = self.iv_model(vgs, vds, vbs, device_emb, training=self.training)

            # 7. Convert log10(gm/gds) to z-scored space for loss compatibility
            log10_vdc_std = torch.log10(self.vdc_std_buf.clamp(min=1e-6))
            log10_gm = iv_out['log10_gm'] - log10_vdc_std
            log10_gds = iv_out['log10_gds'] - log10_vdc_std
            result['mosfet_gm_pred'] = (log10_gm - self.ss_gm_mean_buf) / self.ss_gm_std_buf
            result['mosfet_gds_pred'] = (log10_gds - self.ss_gds_mean_buf) / self.ss_gds_std_buf

            # 8. Store IV outputs for ID supervision and diagnostics
            result['mosfet_iv_log_abs_id'] = iv_out['log_abs_id']

        # DC gain head
        if self.predict_dc_gain:
            if self.dc_gain_mode == 'physics':
                # Physics-informed DC gain from predicted gm/gds
                gm_pred = result['mosfet_gm_pred']   # [total_mosfets] z-scored log10
                gds_pred = result['mosfet_gds_pred']  # [total_mosfets] z-scored log10
                if self.dc_gain_detach_ss:
                    gm_pred = gm_pred.detach()
                    gds_pred = gds_pred.detach()

                # Denormalize to log10 domain
                log10_gm = gm_pred * self.ss_gm_std_buf + self.ss_gm_mean_buf
                log10_gds = gds_pred * self.ss_gds_std_buf + self.ss_gds_mean_buf

                # Reshape flat [total_mosfets] → [B, 24] (fixed topology: 24 MOSFETs/graph)
                B = num_graphs
                M = 24  # fixed topology
                log10_gm_b = log10_gm.view(B, M)    # [B, 24]
                log10_gds_b = log10_gds.view(B, M)   # [B, 24]

                # Extract 14 key MOSFETs: [B, 14]
                idx = self._dc_mosfet_indices
                gm_key = log10_gm_b[:, idx]     # [B, 14] log10 domain
                gds_key = log10_gds_b[:, idx]    # [B, 14] log10 domain

                # All computation in log10 space — no pow(10,...) needed
                # Index map within the 14 key MOSFETs:
                # 0=M8, 1=M9, 2=M5, 3=M6, 4=M15, 5=M16, 6=M19, 7=M20
                # 8=M7, 9=M10, 10=M21, 11=M22, 12=M11, 13=M23

                # Stage 1: log10(Rout1) = -log10(gds6 + gds16) or cascode variant
                if self._dc_rout1_formula == 'simple':
                    log_rout1 = -log10_add(gds_key[:, 3], gds_key[:, 5])
                else:  # cascode: second term = gds16*gds20/gm16
                    log_cascode_term = gds_key[:, 5] + gds_key[:, 7] - gm_key[:, 5]
                    log_rout1 = -log10_add(gds_key[:, 3], log_cascode_term)
                log_A1 = gm_key[:, 0] + log_rout1  # log10(gm8 * Rout1)

                # Stage 2: A2 = [gm10/(gm21+gds21+gds10)] * [gm22/(gds7+gds22)]
                log_denom1 = log10_add(log10_add(gm_key[:, 10], gds_key[:, 10]), gds_key[:, 9])
                log_A2_p1 = gm_key[:, 9] - log_denom1
                log_denom2 = log10_add(gds_key[:, 8], gds_key[:, 11])
                log_A2 = log_A2_p1 + gm_key[:, 11] - log_denom2

                # Stage 3: Rout3 = 1/(gds11+gds23), loaded with R_load
                log_rout3 = -log10_add(gds_key[:, 12], gds_key[:, 13])
                R_load = self._dc_r_in + self._dc_r_f
                log_inv_rload = torch.tensor(-math.log10(float(R_load)), device=gm_key.device)
                log_rout3_loaded = -log10_add(-log_rout3, log_inv_rload)

                # Feedback factor
                beta = self._dc_r_in / (self._dc_r_in + self._dc_r_f + 1e-15)

                # T = A1 * Rout3_loaded * (gm11 + A2*gm23) * beta
                log_sum_gm = log10_add(gm_key[:, 12], log_A2 + gm_key[:, 13])
                log_T = log_A1 + log_rout3_loaded + log_sum_gm + math.log10(float(beta))
                dc_gain_est_dB = 20.0 * log_T  # [B], already in log10

                # Assemble: 14 gm + 14 gds + 7 formula intermediates = 35 features (all log10)
                formula_feats = torch.stack([
                    log_rout1, log_A1, log_A2_p1, log_A2,
                    log_rout3, log_rout3_loaded, dc_gain_est_dB,
                ], dim=-1)  # [B, 7]
                parts = [gm_key, gds_key, formula_feats]
                # Optionally augment with voltage operating point context (Vgs, Vds)
                if getattr(self, 'dc_gain_use_volt_ctx', False):
                    v_pred = result['node_voltages'].detach()
                    v_gate = v_pred[mosfet_info[:, 0] + offsets].view(B, M)[:, idx]
                    v_drain = v_pred[mosfet_info[:, 1] + offsets].view(B, M)[:, idx]
                    v_source = v_pred[mosfet_info[:, 2] + offsets].view(B, M)[:, idx]
                    parts.append(v_gate - v_source)  # Vgs [B, 14]
                    parts.append(v_drain - v_source)  # Vds [B, 14]
                # Optionally augment with VN context for global circuit info
                if self.dc_gain_use_vn:
                    if len(backbone_mha_outputs) > 0:
                        # MHA mode: pool last MHA layer output to graph-level
                        from torch_geometric.nn import global_mean_pool
                        vn_graph = global_mean_pool(backbone_mha_outputs[-1].detach(), batch_vec)
                    elif vn_emb is not None:
                        # Default VN mode: use graph-level VN embedding directly
                        vn_graph = vn_emb.detach()
                    else:
                        vn_graph = None
                    if vn_graph is not None:
                        vn_feat = self.dc_gain_vn_proj(vn_graph) if self.dc_gain_vn_proj is not None else vn_graph
                        parts.append(vn_feat)
                # Optionally augment with projected device embeddings from sensitivity tower
                if getattr(self, 'dc_gain_use_device_ctx', False):
                    idx = self._dc_mosfet_indices
                    gate_sens = sens_repr[mosfet_info[:, 0] + offsets].view(B, M, -1)[:, idx]
                    drain_sens = sens_repr[mosfet_info[:, 1] + offsets].view(B, M, -1)[:, idx]
                    source_sens = sens_repr[mosfet_info[:, 2] + offsets].view(B, M, -1)[:, idx]
                    dev_concat = torch.cat([gate_sens, drain_sens, source_sens], dim=-1)  # [B, 14, 447]
                    if self.dc_gain_detach_ss:
                        dev_concat = dev_concat.detach()
                    dev_proj = self.dc_gain_dev_proj(dev_concat)  # [B, 14, 32]
                    parts.append(dev_proj.reshape(B, -1))  # [B, 448]

                physics_input = torch.cat(parts, dim=-1)  # [B, 35+448]

                result['dc_gain_pred'] = self.dc_gain_head(physics_input).squeeze(-1)  # [B]
                result['dc_gain_physics_est_dB'] = dc_gain_est_dB.detach()  # for diagnostics
            elif self.dc_gain_mode == 'vn_mlp':
                # VN embedding → MLP (simplest learned DC gain)
                from torch_geometric.nn import global_mean_pool
                if len(backbone_mha_outputs) > 0:
                    vn_input = global_mean_pool(backbone_mha_outputs[-1].detach(), batch_vec)
                elif vn_emb is not None:
                    vn_input = vn_emb.detach()
                else:
                    vn_input = global_mean_pool(backbone_repr.detach(), batch_vec)
                result['dc_gain_pred'] = self.dc_gain_head(vn_input).squeeze(-1)
            elif self.dc_gain_mode == 'concat_mlp':
                # Flatten 14 key MOSFET embeddings → MLP
                B, M = num_graphs, 24
                idx = self._dc_mosfet_indices
                key_emb = sens_repr[mosfet_info[:, 1] + offsets].view(B, M, -1)[:, idx]  # [B, 14, 149]
                if self.dc_gain_detach_ss:
                    key_emb = key_emb.detach()
                result['dc_gain_pred'] = self.dc_gain_head(key_emb.reshape(B, -1)).squeeze(-1)
            elif self.dc_gain_mode == 'stage_pool':
                # Per-stage mean pool → concat → MLP
                B, M = num_graphs, 24
                idx = self._dc_mosfet_indices
                bias_idx = self._dc_bias_indices
                all_emb = sens_repr[mosfet_info[:, 1] + offsets].view(B, M, -1)
                if self.dc_gain_detach_ss:
                    all_emb = all_emb.detach()
                key_emb = all_emb[:, idx]
                stage1 = key_emb[:, 0:8].mean(dim=1)
                stage2 = key_emb[:, 8:12].mean(dim=1)
                stage3 = key_emb[:, 12:14].mean(dim=1)
                bias = all_emb[:, bias_idx].mean(dim=1)
                result['dc_gain_pred'] = self.dc_gain_head(
                    torch.cat([stage1, stage2, stage3, bias], dim=-1)
                ).squeeze(-1)
            elif self.dc_gain_mode == 'pool':
                # Simple mean+max pool of all nodes → MLP
                from torch_geometric.nn import global_mean_pool, global_max_pool
                pool_mean = global_mean_pool(backbone_hidden.detach(), batch_vec)
                pool_max = global_max_pool(backbone_hidden.detach(), batch_vec)
                pool_input = torch.cat([pool_mean, pool_max], dim=-1)
                result['dc_gain_pred'] = self.dc_gain_head(pool_input).squeeze(-1)
            elif self.dc_gain_mode == 'physics_cross_attn':
                # Physics formula + cross-attention correction
                # Step 1: Same physics formula as 'physics' mode
                gm_pred = result['mosfet_gm_pred']
                gds_pred = result['mosfet_gds_pred']
                if self.dc_gain_detach_ss:
                    gm_pred = gm_pred.detach()
                    gds_pred = gds_pred.detach()
                log10_gm = gm_pred * self.ss_gm_std_buf + self.ss_gm_mean_buf
                log10_gds = gds_pred * self.ss_gds_std_buf + self.ss_gds_mean_buf
                B, M = num_graphs, 24
                idx = self._dc_mosfet_indices
                gm_key = log10_gm.view(B, M)[:, idx]
                gds_key = log10_gds.view(B, M)[:, idx]
                # Formula in log10 space (same as physics mode)
                if self._dc_rout1_formula == 'simple':
                    log_rout1 = -log10_add(gds_key[:, 3], gds_key[:, 5])
                else:
                    log_cascode_term = gds_key[:, 5] + gds_key[:, 7] - gm_key[:, 5]
                    log_rout1 = -log10_add(gds_key[:, 3], log_cascode_term)
                log_A1 = gm_key[:, 0] + log_rout1
                log_denom1 = log10_add(log10_add(gm_key[:, 10], gds_key[:, 10]), gds_key[:, 9])
                log_A2_p1 = gm_key[:, 9] - log_denom1
                log_denom2 = log10_add(gds_key[:, 8], gds_key[:, 11])
                log_A2 = log_A2_p1 + gm_key[:, 11] - log_denom2
                log_rout3 = -log10_add(gds_key[:, 12], gds_key[:, 13])
                R_load = self._dc_r_in + self._dc_r_f
                log_inv_rload = torch.tensor(-math.log10(float(R_load)), device=gm_key.device)
                log_rout3_loaded = -log10_add(-log_rout3, log_inv_rload)
                beta = self._dc_r_in / (self._dc_r_in + self._dc_r_f + 1e-15)
                log_sum_gm = log10_add(gm_key[:, 12], log_A2 + gm_key[:, 13])
                log_T = log_A1 + log_rout3_loaded + log_sum_gm + math.log10(float(beta))
                dc_gain_est_dB = 20.0 * log_T
                formula_feats = torch.stack([
                    log_rout1, log_A1, log_A2_p1, log_A2,
                    log_rout3, log_rout3_loaded, dc_gain_est_dB,
                ], dim=-1)
                physics_input = torch.cat([gm_key, gds_key, formula_feats], dim=-1)
                physics_pred = self.dc_gain_head(physics_input).squeeze(-1)

                # Step 2: Cross-attention correction
                # Physics query tokens: [B, 35, 1] → [B, 35, 128]
                Q = self.dc_ca_physics_proj(physics_input.unsqueeze(-1))
                type_ids = torch.arange(35, device=Q.device)
                Q = Q + self.dc_ca_physics_type_embed(type_ids)

                # Device key/value tokens: [B, 14, 149] → [B, 14, 128]
                key_emb = sens_repr[mosfet_info[:, 1] + offsets].view(B, M, -1)[:, idx]
                if self.dc_gain_detach_ss:
                    key_emb = key_emb.detach()
                KV = self.dc_ca_device_proj(key_emb)
                dev_ids = torch.arange(14, device=KV.device)
                KV = KV + self.dc_ca_device_type_embed(dev_ids)

                # Cross-attention + residual + norm
                attended, _ = self.dc_ca_cross_attn(Q, KV, KV)
                attended = self.dc_ca_norm(Q + attended)

                # Pool → correction scalar
                correction = self.dc_ca_pool(attended.reshape(B, -1)).squeeze(-1)

                # Gated residual: physics + gate * correction
                gate = self.dc_ca_gate(torch.stack([physics_pred.detach(), correction], dim=-1))
                result['dc_gain_pred'] = physics_pred + gate.squeeze(-1) * correction
                result['dc_gain_physics_est_dB'] = dc_gain_est_dB.detach()
            elif self.dc_gain_mode in ('physics_residual', 'physics_gated'):
                # Physics formula + learned correction (residual or gated)
                # Step 1: Same physics formula
                gm_pred = result['mosfet_gm_pred']
                gds_pred = result['mosfet_gds_pred']
                if self.dc_gain_detach_ss:
                    gm_pred = gm_pred.detach()
                    gds_pred = gds_pred.detach()
                log10_gm = gm_pred * self.ss_gm_std_buf + self.ss_gm_mean_buf
                log10_gds = gds_pred * self.ss_gds_std_buf + self.ss_gds_mean_buf
                B, M = num_graphs, 24
                idx = self._dc_mosfet_indices
                gm_key = log10_gm.view(B, M)[:, idx]
                gds_key = log10_gds.view(B, M)[:, idx]
                # Formula in log10 space (same as physics mode)
                if self._dc_rout1_formula == 'simple':
                    log_rout1 = -log10_add(gds_key[:, 3], gds_key[:, 5])
                else:
                    log_cascode_term = gds_key[:, 5] + gds_key[:, 7] - gm_key[:, 5]
                    log_rout1 = -log10_add(gds_key[:, 3], log_cascode_term)
                log_A1 = gm_key[:, 0] + log_rout1
                log_denom1 = log10_add(log10_add(gm_key[:, 10], gds_key[:, 10]), gds_key[:, 9])
                log_A2_p1 = gm_key[:, 9] - log_denom1
                log_denom2 = log10_add(gds_key[:, 8], gds_key[:, 11])
                log_A2 = log_A2_p1 + gm_key[:, 11] - log_denom2
                log_rout3 = -log10_add(gds_key[:, 12], gds_key[:, 13])
                R_load = self._dc_r_in + self._dc_r_f
                log_inv_rload = torch.tensor(-math.log10(float(R_load)), device=gm_key.device)
                log_rout3_loaded = -log10_add(-log_rout3, log_inv_rload)
                beta = self._dc_r_in / (self._dc_r_in + self._dc_r_f + 1e-15)
                log_sum_gm = log10_add(gm_key[:, 12], log_A2 + gm_key[:, 13])
                log_T = log_A1 + log_rout3_loaded + log_sum_gm + math.log10(float(beta))
                dc_gain_est_dB = 20.0 * log_T
                formula_feats = torch.stack([
                    log_rout1, log_A1, log_A2_p1, log_A2,
                    log_rout3, log_rout3_loaded, dc_gain_est_dB,
                ], dim=-1)
                physics_input = torch.cat([gm_key, gds_key, formula_feats], dim=-1)
                physics_pred = self.dc_gain_head(physics_input).squeeze(-1)

                # Step 2: Learned path from 14 MOSFET embeddings
                key_emb = sens_repr[mosfet_info[:, 1] + offsets].view(B, M, -1)[:, idx]  # [B, 14, 149]
                if self.dc_gain_detach_ss:
                    key_emb = key_emb.detach()
                learned_pred = self.dc_gain_learned_head(key_emb.reshape(B, -1)).squeeze(-1)  # [B]

                if self.dc_gain_mode == 'physics_residual':
                    result['dc_gain_pred'] = physics_pred + learned_pred
                else:  # physics_gated
                    gate = self.dc_gain_gate(torch.stack([physics_pred.detach(), learned_pred], dim=-1)).squeeze(-1)
                    result['dc_gain_pred'] = gate * physics_pred + (1 - gate) * learned_pred
                result['dc_gain_physics_est_dB'] = dc_gain_est_dB.detach()
            else:
                # Cross-attention mode (original)
                B = num_graphs
                N = (data.ptr[1] - data.ptr[0]).item()  # 134
                sens_batched = sens_repr.view(B, N, -1)  # [B, 134, mlp_input_dim]
                kv = self.dc_gain_kv_proj(sens_batched)  # [B, 134, attn_dim]
                q = self.dc_gain_query.expand(B, -1, -1)  # [B, 1, attn_dim]
                attn_out, _ = self.dc_gain_cross_attn(q, kv, kv)  # [B, 1, attn_dim]
                attn_out = self.dc_gain_attn_norm(attn_out.squeeze(1))  # [B, attn_dim]
                result['dc_gain_pred'] = self.dc_gain_head(attn_out).squeeze(-1)  # [B]

        # AC head: graph-level prediction (detached from backbone)
        if self.predict_ac:
            if self.ac_readout in ('cross_attn', 'mha_pool') and len(backbone_mha_outputs) > 0:
                from torch_geometric.nn import global_mean_pool
                # Step 1: Pool each backbone MHA output to graph-level (detached)
                layer_pools = [global_mean_pool(mha.detach(), batch_vec) for mha in backbone_mha_outputs]
                layer_stack = torch.stack(layer_pools, dim=1)  # [B, num_layers, hidden_dim]
                # Step 2: Layer attention — learned softmax weighting
                layer_scores = self.ac_layer_attn(layer_stack).squeeze(-1)  # [B, num_layers]
                layer_weights = torch.softmax(layer_scores, dim=1)          # [B, num_layers]
                ac_emb = (layer_stack * layer_weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]
                if self.ac_readout == 'cross_attn':
                    # Step 3: Cross-attention over state nodes
                    B = num_graphs
                    N = (data.ptr[1] - data.ptr[0]).item()
                    state_batched = state_repr.detach().view(B, N, -1)     # [B, 134, mlp_input_dim]
                    q = self.ac_q_proj(ac_emb).unsqueeze(1)       # [B, 1, attn_dim]
                    kv = self.ac_kv_proj(state_batched)            # [B, 134, attn_dim]
                    attn_out, _ = self.ac_cross_attn(q, kv, kv)   # [B, 1, attn_dim]
                    attn_out = self.ac_attn_norm(attn_out.squeeze(1))  # [B, attn_dim]
                    # Step 4: Concat with ac_emb (residual) and predict
                    ac_input = torch.cat([attn_out, ac_emb], dim=-1)  # [B, attn_dim + hidden_dim]
                    result['ac_pred'] = self.ac_head(ac_input)
                else:
                    # mha_pool: just layer-attended MHA pool → MLP
                    result['ac_pred'] = self.ac_head(ac_emb)
            elif self.ac_readout == 'vn' and vn_emb is not None:
                # Use actual VN embedding (accumulated across all backbone layers)
                result['ac_pred'] = self.ac_head(vn_emb.detach())
            elif self.ac_readout == 'mosfet_concat':
                # Flatten 14 key MOSFET embeddings from sensitivity tower
                B, M = num_graphs, 24
                idx = self._ac_mosfet_indices
                key_emb = sens_repr[mosfet_info[:, 1] + offsets].view(B, M, -1)[:, idx]  # [B, 14, 149]
                if self.ac_detach:
                    key_emb = key_emb.detach()
                result['ac_pred'] = self.ac_head(key_emb.reshape(B, -1))  # [B, ac_output_dim]
            elif self.ac_readout == 'mosfet_physics':
                # UGBW physics formula in log space + correction MLP
                B, M = num_graphs, 24
                idx = self._ac_mosfet_indices
                gm_pred = result['mosfet_gm_pred']
                gds_pred = result['mosfet_gds_pred']
                if self.ac_detach:
                    gm_pred = gm_pred.detach()
                    gds_pred = gds_pred.detach()
                z_gm_key = gm_pred.view(B, M)[:, idx]   # [B, 14] z-scored
                z_gds_key = gds_pred.view(B, M)[:, idx]  # [B, 14] z-scored

                # gm_M8 in log10 space (for formula estimate)
                log10_gm_M8 = z_gm_key[:, 0] * self.ss_gm_std_buf + self.ss_gm_mean_buf

                # Cc from capacitor terminal node feature (vectorized)
                cap_global_idx = data.ptr[:-1] + self._ac_cap_local_idx
                Cc_norm = data.x[cap_global_idx, 0]
                log10_Cc = Cc_norm * self._ac_cc_log10_range + self._ac_cc_log10_min

                # Formula in log10 space (no pow(10) needed!)
                beta = self._dc_r_in / (self._dc_r_in + self._dc_r_f + 1e-15)
                log10_ugbw_est = log10_gm_M8 + math.log10(float(beta) / (2 * math.pi)) - log10_Cc

                # Assemble: 14 z_gm + 14 z_gds + Cc_norm + log10_ugbw_est = 30
                parts = [z_gm_key, z_gds_key, Cc_norm.unsqueeze(-1), log10_ugbw_est.unsqueeze(-1)]
                physics_input = torch.cat(parts, dim=-1)
                result['ac_pred'] = self.ac_head(physics_input)
            elif self.ac_readout == 'mosfet_physics_dag':
                # UGBW physics formula + DAG subcircuit embeddings
                B, M = num_graphs, 24
                idx = self._ac_mosfet_indices
                gm_pred = result['mosfet_gm_pred']
                gds_pred = result['mosfet_gds_pred']
                if self.ac_detach:
                    gm_pred = gm_pred.detach()
                    gds_pred = gds_pred.detach()
                z_gm_key = gm_pred.view(B, M)[:, idx]
                z_gds_key = gds_pred.view(B, M)[:, idx]
                log10_gm_M8 = z_gm_key[:, 0] * self.ss_gm_std_buf + self.ss_gm_mean_buf
                cap_global_idx = data.ptr[:-1] + self._ac_cap_local_idx
                Cc_norm = data.x[cap_global_idx, 0]
                log10_Cc = Cc_norm * self._ac_cc_log10_range + self._ac_cc_log10_min
                beta = self._dc_r_in / (self._dc_r_in + self._dc_r_f + 1e-15)
                log10_ugbw_est = log10_gm_M8 + math.log10(float(beta) / (2 * math.pi)) - log10_Cc
                parts = [z_gm_key, z_gds_key, Cc_norm.unsqueeze(-1), log10_ugbw_est.unsqueeze(-1)]
                # Append DAG subcircuit embeddings (zeros if DAG not active yet)
                if _sc_embs is not None:
                    parts.append(_sc_embs.reshape(B, -1))
                else:
                    parts.append(torch.zeros(B, self._ac_dag_dim, device=gm_pred.device))
                physics_input = torch.cat(parts, dim=-1)
                result['ac_pred'] = self.ac_head(physics_input)
            elif self.ac_readout == 'physics_dag_residual':
                # Physics UGBW estimate + DAG output stage residual correction
                B, M = num_graphs, 24
                idx = self._ac_mosfet_indices
                gm_pred = result['mosfet_gm_pred'].detach()
                gds_pred = result['mosfet_gds_pred'].detach()
                z_gm_key = gm_pred.view(B, M)[:, idx]
                z_gds_key = gds_pred.view(B, M)[:, idx]
                log10_gm_M8 = z_gm_key[:, 0] * self.ss_gm_std_buf + self.ss_gm_mean_buf
                cap_global_idx = data.ptr[:-1] + self._ac_cap_local_idx
                Cc_norm = data.x[cap_global_idx, 0]
                log10_Cc = Cc_norm * self._ac_cc_log10_range + self._ac_cc_log10_min
                beta = self._dc_r_in / (self._dc_r_in + self._dc_r_f + 1e-15)
                log10_ugbw_est = log10_gm_M8 + math.log10(float(beta) / (2 * math.pi)) - log10_Cc
                physics_input = torch.cat([z_gm_key, z_gds_key, Cc_norm.unsqueeze(-1), log10_ugbw_est.unsqueeze(-1)], dim=-1)
                physics_pred = self.ac_head(physics_input)  # [B, 1]
                # DAG correction from output stage (NOT detached)
                if _sc_embs_live is not None:
                    dag_output = _sc_embs_live[:, -1, :]  # [B, 128]
                else:
                    dag_output = torch.zeros(B, 128, device=backbone_hidden.device)
                correction = self.ac_dag_correction(dag_output)
                result['ac_pred'] = physics_pred + correction
            elif self.ac_readout == 'dag_only':
                # UGBW from last DAG subcircuit embedding (output stage)
                B = num_graphs
                if _sc_embs is not None:
                    dag_input = _sc_embs[:, -1, :]  # [B, sc_dim] — last subcircuit (output)
                else:
                    dag_input = torch.zeros(B, 128, device=backbone_hidden.device)
                result['ac_pred'] = self.ac_head(dag_input)
            else:
                # Fallback: mean+max pool (for 'pool' mode or MHA mode where vn_emb is None)
                from torch_geometric.nn import global_mean_pool, global_max_pool
                x_mean = global_mean_pool(state_repr.detach(), batch_vec)
                x_max = global_max_pool(state_repr.detach(), batch_vec)
                ac_input = torch.cat([x_mean, x_max], dim=-1)
                result['ac_pred'] = self.ac_head(ac_input)
            result['ac_components'] = self.ac_components

        return result

    def _get_num_graphs(self, data, batch: torch.Tensor) -> int:
        if hasattr(data, 'num_graphs'):
            return data.num_graphs
        if hasattr(data, 'ptr'):
            return len(data.ptr) - 1
        return int(batch.max()) + 1
