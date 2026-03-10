"""
Voltage-derived current prediction components.

Computes device currents from predicted node voltages using physics-informed
relationships (e.g., I_ds = f(Vgs, Vds) for MOSFETs).
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import build_mlp


class MOSFETCurrentMLP(nn.Module):
    """
    Predict MOSFET drain-source current from terminal voltages and device parameters.

    Uses a small MLP to learn the I-V relationship:
        I_ds = f(Vgs, Vds, W/L, is_nmos)

    The MLP learns to approximate transistor equations across operating regions
    (cutoff, triode, saturation) without explicitly modeling them.

    Args:
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers
        dropout: Dropout probability
        use_wl_ratio: Include W/L ratio as input feature
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_wl_ratio: bool = True,
    ):
        super().__init__()
        self.use_wl_ratio = use_wl_ratio
        # Input: [Vgs, Vds, is_nmos] or [Vgs, Vds, wl_ratio, is_nmos]
        input_dim = 4 if use_wl_ratio else 3
        # Output: I_ds (log-scale normalized, same as current targets)
        self.mlp = build_mlp(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            norm_type='layer',
            dropout=dropout,
        )
        # Initialize output layer bias to shift predictions closer to target range
        # Target currents are normalized with mean ~-5 (log10 scale), std ~1
        # Normalized target values are roughly centered around 0
        # Initialize bias to 0 (neutral) and small output layer weights
        self._init_output_layer()

    def forward(
        self,
        Vgs: torch.Tensor,
        Vds: torch.Tensor,
        is_nmos: torch.Tensor,
        wl_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict drain-source current from voltages and device parameters.

        Args:
            Vgs: Gate-source voltage [num_mosfets]
            Vds: Drain-source voltage [num_mosfets]
            is_nmos: 1 for NMOS, 0 for PMOS [num_mosfets]
            wl_ratio: W/L ratio [num_mosfets] (optional, used if use_wl_ratio=True)

        Returns:
            I_ds: Predicted drain-source current [num_mosfets]
        """
        if self.use_wl_ratio and wl_ratio is not None:
            x = torch.stack([Vgs, Vds, wl_ratio, is_nmos.float()], dim=-1)
        else:
            x = torch.stack([Vgs, Vds, is_nmos.float()], dim=-1)
        out = self.mlp(x).squeeze(-1)
        return out

    def _init_output_layer(self):
        """Initialize output layer with small weights to prevent large initial predictions."""
        # Find the last linear layer in the MLP
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear):
                # Small weights to reduce initial output magnitude
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    # Initialize bias to 0 - outputs will be close to zero initially
                    nn.init.zeros_(module.bias)
                break


def compute_mosfet_currents(
    pred_voltages: torch.Tensor,
    mosfet_info: torch.Tensor,
    num_terminals,
    current_mlp: nn.Module,
    ptr: Optional[torch.Tensor] = None,
    mosfet_ptr: Optional[torch.Tensor] = None,
    terminal_features: Optional[torch.Tensor] = None,
    wl_feature_idx: int = 2,
    num_nodes: Optional[int] = None,
    use_terminal_indices: bool = False,
    voltage_mean: float = 0.0,
    voltage_std: float = 1.0,
) -> Optional[torch.Tensor]:
    """
    Compute MOSFET currents from predicted voltages.

    Uses mosfet_info tensor to gather voltages at gate/drain/source nets,
    computes Vgs and Vds, then predicts I_ds using the current MLP.

    Handles both single graphs and batched graphs. For batched data, uses ptr
    (node offsets per graph) and mosfet_ptr (MOSFET offsets per graph) to
    compute proper indices.

    Args:
        pred_voltages: Predicted voltages for all nodes [num_nodes]
        mosfet_info: MOSFET terminal/net indices [num_mosfets, 7]
            Columns: [gate_term, drain_term, source_term, gate_net, drain_net, source_net, is_nmos]
        num_terminals: Number of terminal nodes per graph (int or Tensor)
        current_mlp: MLP module for I-V prediction
        ptr: Node offset per graph [num_graphs + 1], for batched data
        mosfet_ptr: MOSFET offset per graph [num_graphs + 1], for batched data
        terminal_features: Node features tensor [num_nodes, feature_dim] for gathering W/L
        wl_feature_idx: Index of W/L ratio in terminal features (default: 2)
        num_nodes: Total number of nodes for output size (for KCL compatibility).
            If provided, output will be [num_nodes] with zeros at net positions.
            If None, output will be [total_terminals].

    Returns:
        node_currents: Current at each node [num_nodes] or [total_terminals], or None if no MOSFETs
    """
    if mosfet_info is None or len(mosfet_info) == 0:
        return None

    device = pred_voltages.device
    num_mosfets = len(mosfet_info)

    # Compute total terminals and determine if batched
    if isinstance(num_terminals, torch.Tensor):
        if num_terminals.numel() == 1:
            total_terminals = int(num_terminals.item())
            is_batched = False
        else:
            total_terminals = int(num_terminals.sum().item())
            is_batched = True
    else:
        total_terminals = int(num_terminals)
        is_batched = False

    # Extract base indices (per-graph, before offsetting)
    gate_term_idx = mosfet_info[:, 0].clone()
    drain_term_idx = mosfet_info[:, 1].clone()
    source_term_idx = mosfet_info[:, 2].clone()
    gate_net_idx = mosfet_info[:, 3].clone()
    drain_net_idx = mosfet_info[:, 4].clone()
    source_net_idx = mosfet_info[:, 5].clone()
    is_nmos = mosfet_info[:, 6]

    # Handle batched data: add node offsets per graph
    if is_batched and ptr is not None:
        num_graphs = len(ptr) - 1

        # Compute which graph each MOSFET belongs to
        if mosfet_ptr is not None:
            # Use provided mosfet_ptr
            mosfet_graph_idx = torch.bucketize(
                torch.arange(num_mosfets, device=device),
                mosfet_ptr[1:],
                right=True
            )
        else:
            # Assume uniform: same number of MOSFETs per graph
            mosfets_per_graph = num_mosfets // num_graphs
            mosfet_graph_idx = torch.arange(num_mosfets, device=device) // mosfets_per_graph

        # Get node offsets for each MOSFET's graph
        node_offsets = ptr[mosfet_graph_idx]

        # Add offsets to all index columns
        gate_term_idx = gate_term_idx + node_offsets
        drain_term_idx = drain_term_idx + node_offsets
        source_term_idx = source_term_idx + node_offsets
        gate_net_idx = gate_net_idx + node_offsets
        drain_net_idx = drain_net_idx + node_offsets
        source_net_idx = source_net_idx + node_offsets

        # Output indices: use node offsets (ptr) when output is num_nodes,
        # otherwise use terminal offsets (term_ptr) when output is total_terminals
        if num_nodes is not None:
            # Output is [num_nodes], so use node offsets
            drain_term_idx_out = mosfet_info[:, 1].long() + node_offsets
            source_term_idx_out = mosfet_info[:, 2].long() + node_offsets
        elif isinstance(num_terminals, torch.Tensor):
            # Output is [total_terminals], so use terminal offsets
            term_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
            term_ptr[1:] = num_terminals.cumsum(0)
            term_offsets = term_ptr[mosfet_graph_idx]
            drain_term_idx_out = mosfet_info[:, 1].long() + term_offsets
            source_term_idx_out = mosfet_info[:, 2].long() + term_offsets
        else:
            drain_term_idx_out = drain_term_idx
            source_term_idx_out = source_term_idx
    else:
        drain_term_idx_out = drain_term_idx
        source_term_idx_out = source_term_idx

    # Gather voltages - use terminal indices for ground truth, net indices for predictions
    if use_terminal_indices:
        V_g = pred_voltages[gate_term_idx]
        V_d = pred_voltages[drain_term_idx]
        V_s = pred_voltages[source_term_idx]
    else:
        V_g = pred_voltages[gate_net_idx]
        V_d = pred_voltages[drain_net_idx]
        V_s = pred_voltages[source_net_idx]

    # Compute voltage differences in NORMALIZED space
    # For differences, the mean cancels out anyway: (V1*std+mean) - (V2*std+mean) = (V1-V2)*std
    # By keeping voltages normalized, MLP inputs have consistent scale regardless of voltage_mean/std
    # The MLP can learn the I-V relationship equally well on normalized or actual voltage differences
    Vgs = V_g - V_s
    Vds = V_d - V_s

    # Gather W/L ratio from terminal features (if available)
    wl_ratio = None
    if terminal_features is not None and hasattr(current_mlp, 'use_wl_ratio') and current_mlp.use_wl_ratio:
        # Use drain terminal to gather W/L (all terminals of same device have same props)
        # drain_term_idx already has correct offsets for batched data
        wl_ratio = terminal_features[drain_term_idx, wl_feature_idx]

    # Predict current magnitude via MLP
    I_ds = current_mlp(Vgs, Vds, is_nmos, wl_ratio=wl_ratio)

    # Initialize output tensor - use num_nodes if provided (for KCL compatibility)
    output_size = num_nodes if num_nodes is not None else total_terminals

    # Use scatter_add for efficient and differentiable sparse placement
    # scatter_add is out-of-place (returns new tensor) and preserves gradients to I_ds
    # Add I_ds to both drain and source terminals (same magnitude flows through both)
    node_currents = torch.zeros(output_size, device=device, dtype=I_ds.dtype)
    node_currents = node_currents.scatter_add(0, drain_term_idx_out, I_ds)
    node_currents = node_currents.scatter_add(0, source_term_idx_out, I_ds)

    return node_currents


def compute_resistor_currents(
    pred_voltages: torch.Tensor,
    resistor_info: torch.Tensor,
    num_nodes: int,
    ptr: Optional[torch.Tensor] = None,
    resistor_ptr: Optional[torch.Tensor] = None,
    voltage_mean: float = 0.0,
    voltage_std: float = 1.0,
    current_mean: float = 0.0,
    current_std: float = 1.0,
) -> torch.Tensor:
    """
    Compute resistor currents from Ohm's law: I = (V_p - V_n) / R

    Args:
        pred_voltages: Predicted voltages for all nodes [num_nodes] (z-score normalized)
        resistor_info: [num_resistors, 5] - [term_p, term_n, net_p, net_n, R_norm]
            R_norm is log10(R) normalized
        num_nodes: Total number of nodes
        ptr: Node offsets for batched graphs
        resistor_ptr: Resistor offsets for batched graphs
        voltage_mean/std: For denormalizing voltages
        current_mean/std: For normalizing output to log scale

    Returns:
        Currents at resistor terminals [num_nodes] (normalized log scale)
    """
    if resistor_info is None or len(resistor_info) == 0:
        return torch.zeros(num_nodes, device=pred_voltages.device, dtype=pred_voltages.dtype)

    device = pred_voltages.device
    num_resistors = len(resistor_info)

    # Extract indices
    term_p_idx = resistor_info[:, 0].long()
    term_n_idx = resistor_info[:, 1].long()
    net_p_idx = resistor_info[:, 2].long()
    net_n_idx = resistor_info[:, 3].long()
    r_value_norm = resistor_info[:, 4]  # log10(R) normalized

    # Handle batched data: add node offsets per graph
    if ptr is not None and len(ptr) > 2:
        num_graphs = len(ptr) - 1

        # Compute which graph each resistor belongs to
        if resistor_ptr is not None:
            resistor_graph_idx = torch.bucketize(
                torch.arange(num_resistors, device=device),
                resistor_ptr[1:],
                right=True
            )
        else:
            resistors_per_graph = num_resistors // num_graphs
            resistor_graph_idx = torch.arange(num_resistors, device=device) // resistors_per_graph

        node_offsets = ptr[resistor_graph_idx]
        term_p_idx = term_p_idx + node_offsets
        term_n_idx = term_n_idx + node_offsets
        net_p_idx = net_p_idx + node_offsets
        net_n_idx = net_n_idx + node_offsets

    # Gather voltages at net nodes (normalized)
    V_p_norm = pred_voltages[net_p_idx]
    V_n_norm = pred_voltages[net_n_idx]

    # Denormalize voltages to actual values
    V_p = V_p_norm * voltage_std + voltage_mean
    V_n = V_n_norm * voltage_std + voltage_mean

    # Denormalize R from log10 scale
    R = 10 ** r_value_norm

    # Compute current: I = (V_p - V_n) / R
    delta_V = V_p - V_n
    I_raw = delta_V / R

    # Normalize to log scale (same as current targets)
    # Handle small/zero currents
    I_abs = torch.abs(I_raw).clamp(min=1e-15)
    I_log = torch.log10(I_abs)
    I_norm = (I_log - current_mean) / current_std

    # Place at terminal positions using scatter_add (efficient and differentiable)
    currents = torch.zeros(num_nodes, device=device, dtype=I_norm.dtype)
    currents = currents.scatter_add(0, term_p_idx, I_norm)
    currents = currents.scatter_add(0, term_n_idx, I_norm)
    return currents


def compute_capacitor_currents(
    capacitor_info: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return zeros for DC analysis (capacitors are open circuits).

    For DC analysis, capacitor current is always zero.
    We still place zeros at the terminal positions for consistency.

    Args:
        capacitor_info: [num_capacitors, 4] - [term_p, term_n, net_p, net_n]
        num_nodes: Total number of nodes
        device: Torch device
        dtype: Tensor dtype

    Returns:
        Zero currents [num_nodes]
    """
    # For DC analysis, capacitors are open circuits, I = 0
    return torch.zeros(num_nodes, device=device, dtype=dtype)


def compute_isource_currents(
    isource_info: torch.Tensor,
    num_nodes: int,
    ptr: Optional[torch.Tensor] = None,
    isource_ptr: Optional[torch.Tensor] = None,
    current_mean: float = 0.0,
    current_std: float = 1.0,
) -> torch.Tensor:
    """
    Return i_ref values at current source terminals.

    Current sources have known, constant current values.

    Args:
        isource_info: [num_isources, 5] - [term_p, term_n, net_p, net_n, i_ref_log]
            i_ref_log is log10(i_ref), NOT normalized yet
        num_nodes: Total number of nodes
        ptr: Node offsets for batched graphs
        isource_ptr: I-source offsets for batched graphs
        current_mean/std: For normalizing output to match target scale

    Returns:
        Currents at I-source terminals [num_nodes] (normalized log scale)
    """
    if isource_info is None or len(isource_info) == 0:
        return torch.zeros(num_nodes, dtype=torch.float32)

    device = isource_info.device
    num_isources = len(isource_info)

    # Extract indices
    term_p_idx = isource_info[:, 0].long()
    term_n_idx = isource_info[:, 1].long()
    i_ref_log = isource_info[:, 4]  # log10(i_ref)

    # Handle batched data
    if ptr is not None and len(ptr) > 2:
        num_graphs = len(ptr) - 1

        if isource_ptr is not None:
            isource_graph_idx = torch.bucketize(
                torch.arange(num_isources, device=device),
                isource_ptr[1:],
                right=True
            )
        else:
            isources_per_graph = num_isources // num_graphs
            isource_graph_idx = torch.arange(num_isources, device=device) // isources_per_graph

        node_offsets = ptr[isource_graph_idx]
        term_p_idx = term_p_idx + node_offsets
        term_n_idx = term_n_idx + node_offsets

    # Normalize i_ref to match target scale
    I_norm = (i_ref_log - current_mean) / current_std

    # Place at terminal positions using scatter_add (efficient and differentiable)
    currents = torch.zeros(num_nodes, device=device, dtype=I_norm.dtype)
    currents = currents.scatter_add(0, term_p_idx, I_norm)
    currents = currents.scatter_add(0, term_n_idx, I_norm)
    return currents
