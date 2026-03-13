"""
Device-aware aggregation layer for GNN backbone.

Enriches terminal embeddings with device-level context by gathering
terminal embeddings per device, pooling them, and injecting back.
Similar to a virtual node but at device scope instead of global scope.
"""

import torch
import torch.nn as nn

from src.gnn.components.device_current_head import _compute_batch_offsets


class DeviceAggregationLayer(nn.Module):
    """Inject device-level context into terminal embeddings.

    For each device, gathers terminal embeddings, computes a device context
    vector via pooling + MLP, and adds it back to each terminal.
    Supports mean pooling or attention-weighted pooling.
    """

    def __init__(self, hidden_dim: int, norm_type: str = 'layer', attention: bool = False,
                 separate_mlps: bool = False):
        super().__init__()
        self.separate_mlps = separate_mlps

        def _make_mlp():
            norm = nn.BatchNorm1d(hidden_dim) if norm_type == 'batch' else nn.LayerNorm(hidden_dim)
            return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), norm, nn.ReLU())

        if separate_mlps:
            # MOSFET: separate MLP per terminal
            self.mlp_gate = _make_mlp()
            self.mlp_drain = _make_mlp()
            self.mlp_source = _make_mlp()
            # 2-terminal: separate MLP per terminal
            self.mlp_p = _make_mlp()
            self.mlp_n = _make_mlp()
        else:
            self.mlp = _make_mlp()

        self.attention = attention
        if attention:
            self.attn_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mosfet_info: torch.Tensor = None,
        resistor_info: torch.Tensor = None,
        capacitor_info: torch.Tensor = None,
        vsource_info: torch.Tensor = None,
        isource_info: torch.Tensor = None,
        ptr: torch.Tensor = None,
        mosfet_ptr: torch.Tensor = None,
        resistor_ptr: torch.Tensor = None,
        capacitor_ptr: torch.Tensor = None,
        isource_ptr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Enrich terminal embeddings with device-level context.

        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            *_info: Device info tensors with terminal indices
            ptr: Node boundaries for batching
            *_ptr: Device ptr tensors for batching

        Returns:
            Updated node embeddings [num_nodes, hidden_dim]
        """
        is_batched = ptr is not None and len(ptr) > 2
        out = x.clone()

        # MOSFETs: pool gate + drain + source
        if mosfet_info is not None and mosfet_info.numel() > 0:
            g_idx = mosfet_info[:, 0].long()
            d_idx = mosfet_info[:, 1].long()
            s_idx = mosfet_info[:, 2].long()
            if is_batched:
                offsets = _compute_batch_offsets(len(mosfet_info), mosfet_ptr, ptr, x.device)
                g_idx = g_idx + offsets
                d_idx = d_idx + offsets
                s_idx = s_idx + offsets
            if self.attention:
                # Stack terminals: [N_devices, 3, hidden_dim]
                stacked = torch.stack([x[g_idx], x[d_idx], x[s_idx]], dim=1)
                # Attention scores: [N_devices, 3, 1]
                scores = self.attn_proj(stacked)
                weights = torch.softmax(scores, dim=1)  # [N_devices, 3, 1]
                device_pool = (stacked * weights).sum(dim=1)  # [N_devices, hidden_dim]
            else:
                device_pool = (x[g_idx] + x[d_idx] + x[s_idx]) / 3.0
            if self.separate_mlps:
                out[g_idx] = out[g_idx] + self.mlp_gate(device_pool)
                out[d_idx] = out[d_idx] + self.mlp_drain(device_pool)
                out[s_idx] = out[s_idx] + self.mlp_source(device_pool)
            else:
                device_ctx = self.mlp(device_pool)
                out[g_idx] = out[g_idx] + device_ctx
                out[d_idx] = out[d_idx] + device_ctx
                out[s_idx] = out[s_idx] + device_ctx

        # 2-terminal devices: pool p + n
        two_term_infos = [
            (resistor_info, resistor_ptr),
            (capacitor_info, capacitor_ptr),
            (vsource_info, None),
            (isource_info, isource_ptr),
        ]
        for info, dev_ptr in two_term_infos:
            if info is not None and info.numel() > 0:
                p_idx = info[:, 0].long()
                n_idx = info[:, 1].long()
                if is_batched:
                    offsets = _compute_batch_offsets(len(info), dev_ptr, ptr, x.device)
                    p_idx = p_idx + offsets
                    n_idx = n_idx + offsets
                if self.attention:
                    stacked = torch.stack([x[p_idx], x[n_idx]], dim=1)
                    scores = self.attn_proj(stacked)
                    weights = torch.softmax(scores, dim=1)
                    device_pool = (stacked * weights).sum(dim=1)
                else:
                    device_pool = (x[p_idx] + x[n_idx]) / 2.0
                if self.separate_mlps:
                    out[p_idx] = out[p_idx] + self.mlp_p(device_pool)
                    out[n_idx] = out[n_idx] + self.mlp_n(device_pool)
                else:
                    device_ctx = self.mlp(device_pool)
                    out[p_idx] = out[p_idx] + device_ctx
                    out[n_idx] = out[n_idx] + device_ctx

        return out
