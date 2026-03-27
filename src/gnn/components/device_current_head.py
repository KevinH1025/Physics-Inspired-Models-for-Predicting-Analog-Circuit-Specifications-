"""
Device-level current prediction head.

Predicts one current per device by pooling terminal embeddings,
then scatters back to terminal nodes. This guarantees that
drain and source of the same MOSFET get identical current predictions.

When voltage_input=True, the MOSFET MLP also receives (Vgs, Vds, Vbs) as
explicit scalar inputs, enabling autograd-based gm/gds derivation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


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

    For MOSFETs: concat(drain, source[, Vgs, Vds, Vbs]) -> MLP -> 1 current
    For 2-terminal devices (R, V, I): concat(p, n) -> MLP -> 1 current
    Capacitors get 0 current (DC analysis).

    The output is a [num_nodes] tensor with the device-level prediction
    scattered to all relevant terminal indices, plus a device_mask that
    marks one terminal per device for loss computation (avoids double-counting).

    When voltage_input=True, the MOSFET MLP uses SiLU activations (smooth,
    twice differentiable) to enable autograd-based gm/gds computation.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        norm_type: str = 'layer',
        voltage_input: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.voltage_input = voltage_input

        def _build_mlp(in_dim, use_silu=False):
            layers = []
            curr = in_dim
            act_fn = nn.SiLU if use_silu else nn.ReLU
            for i in range(num_layers - 1):
                out = hidden_dim // (2 ** i) if num_layers > 2 else hidden_dim
                out = max(out, 1)
                layers.extend([nn.Linear(curr, out), _make_norm(out, norm_type), act_fn()])
                curr = out
            layers.append(nn.Linear(curr, 1))
            return nn.Sequential(*layers)

        # MOSFET head: concat(drain_emb, source_emb[, Vgs, Vds, Vbs]) -> MLP -> 1
        mosfet_in_dim = 2 * embed_dim + (3 if voltage_input else 0)
        self.mosfet_mlp = _build_mlp(mosfet_in_dim, use_silu=voltage_input)

        # 2-terminal head: concat(p_emb, n_emb) -> MLP -> 1
        self.two_term_mlp = _build_mlp(2 * embed_dim)

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
        mosfet_voltages: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Predict one current per device, scatter to terminals.

        Args:
            x: Node embeddings [num_nodes, embed_dim]
            mosfet_info: [M, 7] with columns [gate, drain, source, ...]
            resistor_info: [R, 5] with columns [p, n, ...]
            capacitor_info: [C, 4] with columns [p, n, ...]
            vsource_info: [V, 4] with columns [p, n, ...]
            isource_info: [I, 5] with columns [p, n, ...]
            num_nodes: Total number of nodes (for output tensor size)
            ptr: Node boundaries per graph [num_graphs + 1]
            mosfet_ptr, resistor_ptr, capacitor_ptr, isource_ptr: device offsets
            mosfet_voltages: [M, 3] with (Vgs, Vds, Vbs) z-scored, for autograd SS

        Returns:
            (node_currents, device_mask, mosfet_current):
                node_currents: [num_nodes] with current at terminal positions
                device_mask: [num_nodes] bool, True at one terminal per device
                    (drain for MOSFETs, p for 2-terminal) for loss without double-counting
                mosfet_current: [M] per-MOSFET current (before scatter), or None
                    Returned when voltage_input=True for autograd gm/gds computation
        """
        if num_nodes is None:
            num_nodes = x.size(0)

        is_batched = ptr is not None and len(ptr) > 2

        out = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        device_mask = torch.zeros(num_nodes, device=x.device, dtype=torch.bool)
        mosfet_current = None

        # MOSFETs: concat(drain, source[, voltages]) -> predict one current -> scatter to D,S
        if mosfet_info is not None and mosfet_info.numel() > 0:
            d_idx = mosfet_info[:, 1].long()
            s_idx = mosfet_info[:, 2].long()
            if is_batched:
                offsets = _compute_batch_offsets(len(mosfet_info), mosfet_ptr, ptr, x.device)
                d_idx = d_idx + offsets
                s_idx = s_idx + offsets
            parts = [x[d_idx], x[s_idx]]
            if self.voltage_input and mosfet_voltages is not None:
                parts.append(mosfet_voltages)
            device_emb = torch.cat(parts, dim=-1)
            device_current = self.mosfet_mlp(device_emb).squeeze(-1)
            if self.voltage_input:
                mosfet_current = device_current  # keep for autograd before scatter
            device_current_cast = device_current.to(out.dtype)
            out[d_idx] = device_current_cast
            out[s_idx] = device_current_cast
            device_mask[d_idx] = True  # only drain for loss

        # 2-terminal devices: concat(p, n) -> predict -> scatter
        two_term_infos = [
            (resistor_info, resistor_ptr),
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
                device_emb = torch.cat([x[p_idx], x[n_idx]], dim=-1)
                device_current = self.two_term_mlp(device_emb).squeeze(-1).to(out.dtype)
                out[p_idx] = device_current
                out[n_idx] = device_current
                device_mask[p_idx] = True  # only p-terminal for loss

        # Capacitors: DC current = 0 (already initialized to 0, skip)

        return out, device_mask, mosfet_current
