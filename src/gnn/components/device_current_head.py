"""
Device-level current prediction head.

Predicts one current per device by pooling terminal embeddings,
then scatters back to terminal nodes. This guarantees that
drain and source of the same MOSFET get identical current predictions.
"""

import torch
import torch.nn as nn


def _make_norm(dim, norm_type):
    return nn.BatchNorm1d(dim) if norm_type == 'batch' else nn.LayerNorm(dim)


def _compute_batch_offsets(num_devices, device_ptr, ptr, device):
    """Compute node offsets for each device in a batched graph.

    Args:
        num_devices: Total number of devices across all graphs
        device_ptr: Cumulative device counts [num_graphs + 1], or None
        ptr: Node boundaries [num_graphs + 1]
        device: torch device

    Returns:
        Node offset for each device [num_devices]
    """
    if device_ptr is not None:
        graph_idx = torch.bucketize(
            torch.arange(num_devices, device=device),
            device_ptr[1:].to(device), right=True)
    else:
        num_graphs = len(ptr) - 1
        devices_per_graph = num_devices // num_graphs
        graph_idx = torch.arange(num_devices, device=device) // devices_per_graph
    return ptr[graph_idx]


class DevicePoolingCurrentHead(nn.Module):
    """Predict one current per device by pooling terminal embeddings.

    For MOSFETs: concat(drain, source, gate) -> 256 -> 128 -> 1
    For 2-terminal devices (R, V, I): concat(p, n) -> 256 -> 128 -> 1
    Capacitors get 0 current (DC analysis).

    The output is a [num_nodes] tensor with the device-level prediction
    scattered to all relevant terminal indices.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        norm_type: str = 'layer',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        mid_dim = hidden_dim // 2  # 128

        # MOSFET head: concat(drain_emb, source_emb, gate_emb) -> 256 -> 128 -> 1
        self.mosfet_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            _make_norm(hidden_dim, norm_type), nn.ReLU(),
            nn.Linear(hidden_dim, mid_dim),
            _make_norm(mid_dim, norm_type), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

        # 2-terminal head: concat(p_emb, n_emb) -> 256 -> 128 -> 1
        self.two_term_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            _make_norm(hidden_dim, norm_type), nn.ReLU(),
            nn.Linear(hidden_dim, mid_dim),
            _make_norm(mid_dim, norm_type), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mosfet_info: torch.Tensor = None,
        resistor_info: torch.Tensor = None,
        capacitor_info: torch.Tensor = None,
        vsource_info: torch.Tensor = None,
        isource_info: torch.Tensor = None,
        num_nodes: int = None,
        ptr: torch.Tensor = None,
        mosfet_ptr: torch.Tensor = None,
        resistor_ptr: torch.Tensor = None,
        capacitor_ptr: torch.Tensor = None,
        isource_ptr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict one current per device, scatter to terminals.

        Args:
            x: Node embeddings [num_nodes, embed_dim]
            mosfet_info: [M, 7] with columns [gate, drain, source, gate_net, drain_net, source_net, is_nmos]
            resistor_info: [R, 5] with columns [p, n, net_p, net_n, r_value]
            capacitor_info: [C, 4] with columns [p, n, net_p, net_n]
            vsource_info: [V, 4] with columns [p, n, net_p, net_n]
            isource_info: [I, 5] with columns [p, n, net_p, net_n, i_ref]
            num_nodes: Total number of nodes (for output tensor size)
            ptr: Node boundaries per graph [num_graphs + 1] for batch offset adjustment
            mosfet_ptr: Cumulative MOSFET counts [num_graphs + 1]
            resistor_ptr: Cumulative resistor counts [num_graphs + 1]
            capacitor_ptr: Cumulative capacitor counts [num_graphs + 1]
            isource_ptr: Cumulative isource counts [num_graphs + 1]

        Returns:
            Tensor [num_nodes] with current predictions at terminal positions
        """
        if num_nodes is None:
            num_nodes = x.size(0)

        is_batched = ptr is not None and len(ptr) > 2

        out = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)

        # MOSFETs: gather D,S,G embeddings -> predict one current -> scatter to D,S
        if mosfet_info is not None and mosfet_info.numel() > 0:
            g_idx = mosfet_info[:, 0].long()
            d_idx = mosfet_info[:, 1].long()
            s_idx = mosfet_info[:, 2].long()
            if is_batched:
                offsets = _compute_batch_offsets(len(mosfet_info), mosfet_ptr, ptr, x.device)
                g_idx = g_idx + offsets
                d_idx = d_idx + offsets
                s_idx = s_idx + offsets
            device_emb = torch.cat([x[d_idx], x[s_idx], x[g_idx]], dim=-1)
            device_current = self.mosfet_mlp(device_emb).squeeze(-1).to(out.dtype)
            out[d_idx] = device_current
            out[s_idx] = device_current

        # 2-terminal devices: gather p,n embeddings -> predict -> scatter
        two_term_infos = [
            (resistor_info, resistor_ptr),
            (vsource_info, None),  # no vsource_ptr in batching.py
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
                device_emb = torch.cat([x[p_idx], x[n_idx]], dim=-1)
                device_current = self.two_term_mlp(device_emb).squeeze(-1).to(out.dtype)
                out[p_idx] = device_current
                out[n_idx] = device_current

        # Capacitors: DC current = 0 (already initialized to 0, skip)

        return out
