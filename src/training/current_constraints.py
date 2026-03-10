"""
Physics-based current constraints for circuit simulation.

Provides loss functions that enforce circuit relationships:
- Differential pair current balance: I_M1 + I_M2 = I_tail
- Current mirror ratios: I_mirror/I_ref = WL_mirror/WL_ref
"""

import torch
from typing import Dict, List, Optional, Tuple

# Default 2-stage opamp constraints (empty — individual constraints proved
# redundant with KCL loss or too noisy for reliable training).
OPAMP_2STAGE_CONSTRAINTS = {}


def create_constraint_tensors(
    mosfet_device_names: List[str],
    constraint_config: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create tensors for current constraint specification from device names.

    Args:
        mosfet_device_names: List of MOSFET device names (e.g., ['M1', 'M2', ...])
        constraint_config: Dict specifying constraints. If None, uses OPAMP_2STAGE_CONSTRAINTS

    Returns:
        diff_pair_constraints: [N, 3] tensor with [idx_m1, idx_m2, idx_tail] per constraint
        mirror_constraints: [M, 2] tensor with [idx_ref, idx_mirror] per pair
        output_stage_constraints: [K, 2] tensor with [idx_pmos, idx_nmos] per constraint
        lambda_mirror_constraints: [L, 2] tensor with [idx_ref, idx_mirror] per pair
    """
    if constraint_config is None:
        constraint_config = OPAMP_2STAGE_CONSTRAINTS

    # Build name-to-index map (case insensitive, strip 'X' prefix from subcircuit names)
    # Handles both 'M1' and 'Xm1' naming conventions
    name_to_idx = {}
    for i, n in enumerate(mosfet_device_names):
        n_upper = n.upper()
        name_to_idx[n_upper] = i
        # Also map without 'X' prefix (Xm1 -> M1)
        if n_upper.startswith('X'):
            name_to_idx[n_upper[1:]] = i

    # Build diff pair constraint
    diff_pair_list = []
    dp_cfg = constraint_config.get('diff_pair', {})
    devices = dp_cfg.get('devices', [])
    if len(devices) == 3:
        indices = [name_to_idx.get(d.upper(), -1) for d in devices]
        if -1 not in indices:
            diff_pair_list.append(indices)

    # Build mirror constraints
    mirror_list = []
    for key in ['pmos_mirror', 'nmos_bias_mirror']:
        mirror_cfg = constraint_config.get(key, {})
        ref_name = mirror_cfg.get('reference', '')
        ref_idx = name_to_idx.get(ref_name.upper(), -1)
        if ref_idx == -1:
            continue
        for mirror_name in mirror_cfg.get('mirrors', []):
            mirror_idx = name_to_idx.get(mirror_name.upper(), -1)
            if mirror_idx != -1:
                mirror_list.append([ref_idx, mirror_idx])

    # Build output stage constraint (I_M6 = I_M7)
    output_stage_list = []
    os_cfg = constraint_config.get('output_stage', {})
    os_devices = os_cfg.get('devices', [])
    if len(os_devices) == 2:
        indices = [name_to_idx.get(d.upper(), -1) for d in os_devices]
        if -1 not in indices:
            output_stage_list.append(indices)

    # Build lambda-corrected mirror constraint (I_M7/I_M8 with Vds correction)
    lambda_mirror_list = []
    lm_cfg = constraint_config.get('lambda_mirror', {})
    ref_name = lm_cfg.get('reference', '')
    mirror_name = lm_cfg.get('mirror', '')
    if ref_name and mirror_name:
        ref_idx = name_to_idx.get(ref_name.upper(), -1)
        mirror_idx = name_to_idx.get(mirror_name.upper(), -1)
        if ref_idx != -1 and mirror_idx != -1:
            lambda_mirror_list.append([ref_idx, mirror_idx])

    diff_pair_tensor = (
        torch.tensor(diff_pair_list, dtype=torch.long)
        if diff_pair_list else torch.zeros((0, 3), dtype=torch.long)
    )
    mirror_tensor = (
        torch.tensor(mirror_list, dtype=torch.long)
        if mirror_list else torch.zeros((0, 2), dtype=torch.long)
    )
    output_stage_tensor = (
        torch.tensor(output_stage_list, dtype=torch.long)
        if output_stage_list else torch.zeros((0, 2), dtype=torch.long)
    )
    lambda_mirror_tensor = (
        torch.tensor(lambda_mirror_list, dtype=torch.long)
        if lambda_mirror_list else torch.zeros((0, 2), dtype=torch.long)
    )

    return diff_pair_tensor, mirror_tensor, output_stage_tensor, lambda_mirror_tensor


def _gather_mosfet_drain_currents(
    pred_currents: torch.Tensor,
    mosfet_info: torch.Tensor,
    mosfet_indices: torch.Tensor,
    ptr: torch.Tensor,
    current_mean: float,
    current_std: float,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """
    Gather and denormalize drain currents for specified MOSFETs.

    Args:
        pred_currents: Predicted currents [num_nodes] (normalized log scale)
        mosfet_info: MOSFET info tensor [num_mosfets, 7]
        mosfet_indices: Indices into mosfet_info [num_constraints, k]
        ptr: Node boundaries per graph [batch_size + 1]
        current_mean: Mean for denormalization (log10 scale)
        current_std: Std for denormalization (log10 scale)

    Returns:
        currents: [num_constraints, k] denormalized currents in Amps
    """
    device = pred_currents.device
    num_constraints, k = mosfet_indices.shape
    num_graphs = len(ptr) - 1

    if num_constraints == 0:
        return torch.zeros((0, k), device=device)

    # Compute per-graph offsets
    # Constraints are replicated per graph (same local indices)
    # and mosfet_info is concatenated across graphs
    num_mosfets_total = len(mosfet_info)

    # Determine which graph each constraint belongs to
    constraints_per_graph = num_constraints // num_graphs if num_graphs > 0 else num_constraints
    if constraints_per_graph == 0:
        constraints_per_graph = 1

    constraint_graph_idx = torch.arange(num_constraints, device=device) // constraints_per_graph
    constraint_graph_idx = constraint_graph_idx.clamp(max=num_graphs - 1)

    # Compute global MOSFET indices using mosfet_ptr for variable-topology support
    if mosfet_ptr is not None:
        mosfet_offsets = mosfet_ptr[constraint_graph_idx].to(device)
    else:
        mosfets_per_graph = num_mosfets_total // num_graphs
        mosfet_offsets = constraint_graph_idx * mosfets_per_graph
    global_mosfet_indices = mosfet_indices + mosfet_offsets.unsqueeze(-1)

    # Clamp to valid range
    global_mosfet_indices = global_mosfet_indices.clamp(0, num_mosfets_total - 1)

    # Get drain terminal indices (column 1 of mosfet_info)
    drain_term_indices = mosfet_info[global_mosfet_indices.flatten(), 1]

    # Add node offsets for batched graphs
    node_offsets = ptr[constraint_graph_idx].repeat_interleave(k)
    global_drain_indices = drain_term_indices + node_offsets

    # Clamp to valid node range
    global_drain_indices = global_drain_indices.clamp(0, len(pred_currents) - 1)

    # Gather currents
    currents_norm = pred_currents[global_drain_indices]
    currents_norm = currents_norm.view(num_constraints, k)

    # Denormalize: I = 10^(norm * std + mean)
    currents = torch.pow(10, currents_norm * current_std + current_mean)

    return currents


def _gather_mosfet_wl_ratios(
    terminal_features: torch.Tensor,
    mosfet_info: torch.Tensor,
    mosfet_indices: torch.Tensor,
    ptr: torch.Tensor,
    wl_feature_idx: int = 2,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """
    Gather W/L ratios for specified MOSFETs.

    Args:
        terminal_features: Node features [num_nodes, num_features]
        mosfet_info: MOSFET info tensor [num_mosfets, 7]
        mosfet_indices: Indices into mosfet_info [num_constraints, k]
        ptr: Node boundaries per graph [batch_size + 1]
        wl_feature_idx: Index of W/L ratio in feature vector

    Returns:
        wl_ratios: [num_constraints, k] W/L ratios
    """
    device = terminal_features.device
    num_constraints, k = mosfet_indices.shape
    num_graphs = len(ptr) - 1

    if num_constraints == 0:
        return torch.zeros((0, k), device=device)

    num_mosfets_total = len(mosfet_info)

    constraints_per_graph = num_constraints // num_graphs if num_graphs > 0 else num_constraints
    if constraints_per_graph == 0:
        constraints_per_graph = 1

    constraint_graph_idx = torch.arange(num_constraints, device=device) // constraints_per_graph
    constraint_graph_idx = constraint_graph_idx.clamp(max=num_graphs - 1)

    # Compute global MOSFET indices using mosfet_ptr for variable-topology support
    if mosfet_ptr is not None:
        mosfet_offsets = mosfet_ptr[constraint_graph_idx].to(device)
    else:
        mosfets_per_graph = num_mosfets_total // num_graphs
        mosfet_offsets = constraint_graph_idx * mosfets_per_graph
    global_mosfet_indices = mosfet_indices + mosfet_offsets.unsqueeze(-1)
    global_mosfet_indices = global_mosfet_indices.clamp(0, num_mosfets_total - 1)

    # Get drain terminal indices
    drain_term_indices = mosfet_info[global_mosfet_indices.flatten(), 1]

    # Add node offsets
    node_offsets = ptr[constraint_graph_idx].repeat_interleave(k)
    global_drain_indices = drain_term_indices + node_offsets
    global_drain_indices = global_drain_indices.clamp(0, len(terminal_features) - 1)

    # Gather W/L ratios
    wl_ratios = terminal_features[global_drain_indices, wl_feature_idx]
    wl_ratios = wl_ratios.view(num_constraints, k)

    return wl_ratios


def compute_diff_pair_loss(
    pred_currents: torch.Tensor,
    mosfet_info: torch.Tensor,
    diff_pair_constraints: torch.Tensor,
    ptr: torch.Tensor,
    current_mean: float = 0.0,
    current_std: float = 1.0,
) -> torch.Tensor:
    """
    Compute differential pair constraint loss: I_M1 + I_M2 = I_tail.

    Loss = mean(((I_M1 + I_M2) - I_tail)² / I_tail²)

    This is relative squared error, making it scale-invariant.

    Args:
        pred_currents: Predicted currents [num_nodes] (normalized log scale)
        mosfet_info: MOSFET info tensor [num_mosfets, 7]
        diff_pair_constraints: [num_constraints, 3] with [idx_m1, idx_m2, idx_tail]
        ptr: Node boundaries per graph [batch_size + 1]
        current_mean: Mean for denormalization (log10 scale)
        current_std: Std for denormalization (log10 scale)

    Returns:
        Scalar loss tensor
    """
    if diff_pair_constraints is None or len(diff_pair_constraints) == 0:
        return torch.tensor(0.0, device=pred_currents.device)

    # Gather currents for all three devices
    currents = _gather_mosfet_drain_currents(
        pred_currents, mosfet_info, diff_pair_constraints,
        ptr, current_mean, current_std
    )  # [num_constraints, 3]

    I_m1 = currents[:, 0]
    I_m2 = currents[:, 1]
    I_tail = currents[:, 2]

    # Compute relative error: (I_m1 + I_m2 - I_tail) / I_tail
    sum_current = I_m1 + I_m2
    eps = 1e-12
    relative_error = (sum_current - I_tail) / (I_tail + eps)

    return (relative_error ** 2).mean()


def compute_mirror_loss(
    pred_currents: torch.Tensor,
    mosfet_info: torch.Tensor,
    terminal_features: torch.Tensor,
    mirror_constraints: torch.Tensor,
    ptr: torch.Tensor,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    wl_feature_idx: int = 2,
) -> torch.Tensor:
    """
    Compute current mirror constraint loss: I_mirror/I_ref = WL_mirror/WL_ref.

    Loss = mean((actual_ratio - expected_ratio)² / expected_ratio²)

    This is relative squared error, making it scale-invariant.

    Args:
        pred_currents: Predicted currents [num_nodes] (normalized log scale)
        mosfet_info: MOSFET info tensor [num_mosfets, 7]
        terminal_features: Node features [num_nodes, num_features]
        mirror_constraints: [num_pairs, 2] with [idx_ref, idx_mirror]
        ptr: Node boundaries per graph [batch_size + 1]
        current_mean: Mean for denormalization (log10 scale)
        current_std: Std for denormalization (log10 scale)
        wl_feature_idx: Index of W/L ratio in feature vector

    Returns:
        Scalar loss tensor
    """
    if mirror_constraints is None or len(mirror_constraints) == 0:
        return torch.tensor(0.0, device=pred_currents.device)

    # Gather currents for ref and mirror devices
    currents = _gather_mosfet_drain_currents(
        pred_currents, mosfet_info, mirror_constraints,
        ptr, current_mean, current_std
    )  # [num_pairs, 2]

    I_ref = currents[:, 0]
    I_mirror = currents[:, 1]

    # Gather W/L ratios (normalized as log10(W/L) / 3.0)
    wl_ratios_norm = _gather_mosfet_wl_ratios(
        terminal_features, mosfet_info, mirror_constraints,
        ptr, wl_feature_idx
    )  # [num_pairs, 2]

    wl_ref_norm = wl_ratios_norm[:, 0]
    wl_mirror_norm = wl_ratios_norm[:, 1]

    # Denormalize W/L ratios: wl_norm = log10(W/L) / 3.0
    # So W/L = 10^(wl_norm * 3.0)
    # Expected ratio: WL_mirror / WL_ref = 10^(3 * (wl_mirror_norm - wl_ref_norm))
    expected_ratio = torch.pow(10, 3.0 * (wl_mirror_norm - wl_ref_norm))

    # Actual ratio: I_mirror / I_ref
    eps = 1e-12
    actual_ratio = I_mirror / (I_ref + eps)

    # Relative error
    relative_error = (actual_ratio - expected_ratio) / (expected_ratio + eps)

    return (relative_error ** 2).mean()


def compute_output_stage_loss(
    pred_currents: torch.Tensor,
    mosfet_info: torch.Tensor,
    output_stage_constraints: torch.Tensor,
    ptr: torch.Tensor,
    current_mean: float = 0.0,
    current_std: float = 1.0,
) -> torch.Tensor:
    """
    Compute output stage constraint loss: I_M6 = I_M7 (KCL at output node).

    Loss = mean(((I_M6 - I_M7) / (I_M6 + I_M7))²)

    This is relative squared error, making it scale-invariant.

    Args:
        pred_currents: Predicted currents [num_nodes] (normalized log scale)
        mosfet_info: MOSFET info tensor [num_mosfets, 7]
        output_stage_constraints: [num_constraints, 2] with [idx_pmos, idx_nmos]
        ptr: Node boundaries per graph [batch_size + 1]
        current_mean: Mean for denormalization (log10 scale)
        current_std: Std for denormalization (log10 scale)

    Returns:
        Scalar loss tensor
    """
    if output_stage_constraints is None or len(output_stage_constraints) == 0:
        return torch.tensor(0.0, device=pred_currents.device)

    # Gather currents for both devices
    currents = _gather_mosfet_drain_currents(
        pred_currents, mosfet_info, output_stage_constraints,
        ptr, current_mean, current_std
    )  # [num_constraints, 2]

    I_pmos = currents[:, 0]  # M6
    I_nmos = currents[:, 1]  # M7

    # Compute relative error: (I_pmos - I_nmos) / (I_pmos + I_nmos)
    eps = 1e-12
    relative_error = (I_pmos - I_nmos) / (I_pmos + I_nmos + eps)

    return (relative_error ** 2).mean()


def compute_lambda_mirror_loss(
    pred_currents: torch.Tensor,
    pred_voltages: torch.Tensor,
    mosfet_info: torch.Tensor,
    terminal_features: torch.Tensor,
    lambda_mirror_constraints: torch.Tensor,
    node_names: List[List[str]],
    ptr: torch.Tensor,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    wl_feature_idx: int = 2,
    lambda_n: float = 0.05,
    ref_vds_node: str = 'nbias',
    mir_vds_node: str = 'vout',
) -> torch.Tensor:
    """
    Compute lambda-corrected mirror constraint loss.

    I_mir/I_ref = (WL_mir/WL_ref) * (1 + λ*Vds_mir) / (1 + λ*Vds_ref)

    This accounts for channel length modulation effect where Vds differs
    between mirror devices.

    Args:
        pred_currents: Predicted currents [num_nodes] (normalized log scale)
        pred_voltages: Predicted voltages [num_nodes] (normalized)
        mosfet_info: MOSFET info tensor [num_mosfets, 7]
        terminal_features: Node features [num_nodes, num_features]
        lambda_mirror_constraints: [num_constraints, 2] with [idx_ref, idx_mir]
        node_names: List of node names per graph
        ptr: Node boundaries per graph [batch_size + 1]
        current_mean: Mean for current denormalization (log10 scale)
        current_std: Std for current denormalization (log10 scale)
        vdc_mean: Mean for voltage denormalization
        vdc_std: Std for voltage denormalization
        wl_feature_idx: Index of W/L ratio in feature vector
        lambda_n: Channel length modulation parameter
        ref_vds_node: Node name for reference device Vds (e.g., 'nbias')
        mir_vds_node: Node name for mirror device Vds (e.g., 'vout')

    Returns:
        Scalar loss tensor
    """
    if lambda_mirror_constraints is None or len(lambda_mirror_constraints) == 0:
        return torch.tensor(0.0, device=pred_currents.device)

    if pred_voltages is None:
        return torch.tensor(0.0, device=pred_currents.device)

    device = pred_currents.device
    num_graphs = len(ptr) - 1
    eps = 1e-12

    # Gather currents for ref and mirror devices
    currents = _gather_mosfet_drain_currents(
        pred_currents, mosfet_info, lambda_mirror_constraints,
        ptr, current_mean, current_std
    )  # [num_constraints, 2]

    I_ref = currents[:, 0]   # M8
    I_mir = currents[:, 1]   # M7

    # Gather W/L ratios
    wl_ratios_norm = _gather_mosfet_wl_ratios(
        terminal_features, mosfet_info, lambda_mirror_constraints,
        ptr, wl_feature_idx
    )  # [num_constraints, 2]

    wl_ref_norm = wl_ratios_norm[:, 0]
    wl_mir_norm = wl_ratios_norm[:, 1]

    # Denormalize W/L: W/L = 10^(wl_norm * 3.0)
    WL_ref = torch.pow(10, 3.0 * wl_ref_norm)
    WL_mir = torch.pow(10, 3.0 * wl_mir_norm)

    # Find voltage node indices and gather Vds values
    # Build per-graph voltage tensors
    Vds_ref_list = []
    Vds_mir_list = []

    for g in range(num_graphs):
        start = ptr[g].item()
        end = ptr[g + 1].item()
        names = node_names[g] if isinstance(node_names[0], list) else node_names[start:end]

        # Find node indices (case-insensitive)
        ref_idx = None
        mir_idx = None
        for i, name in enumerate(names):
            name_lower = name.lower()
            if name_lower == ref_vds_node.lower():
                ref_idx = i
            if name_lower == mir_vds_node.lower():
                mir_idx = i

        if ref_idx is not None and mir_idx is not None:
            # Get normalized voltages and denormalize
            V_ref_norm = pred_voltages[start + ref_idx]
            V_mir_norm = pred_voltages[start + mir_idx]

            # Denormalize: V = V_norm * std + mean (in Volts)
            Vds_ref = V_ref_norm * vdc_std + vdc_mean
            Vds_mir = V_mir_norm * vdc_std + vdc_mean

            Vds_ref_list.append(Vds_ref)
            Vds_mir_list.append(Vds_mir)
        else:
            # Fallback: assume equal Vds (no correction)
            Vds_ref_list.append(torch.tensor(1.0, device=device))
            Vds_mir_list.append(torch.tensor(1.0, device=device))

    Vds_ref = torch.stack(Vds_ref_list)
    Vds_mir = torch.stack(Vds_mir_list)

    # Lambda correction factor: (1 + λ*Vds_mir) / (1 + λ*Vds_ref)
    lambda_corr = (1 + lambda_n * Vds_mir) / (1 + lambda_n * Vds_ref + eps)

    # Expected ratio with lambda correction
    ratio_expected = (WL_mir / (WL_ref + eps)) * lambda_corr

    # Predicted ratio
    ratio_pred = I_mir / (I_ref + eps)

    # Relative error: (ratio_pred / ratio_expected) - 1
    relative_error = (ratio_pred / (ratio_expected + eps)) - 1

    return (relative_error ** 2).mean()


def compute_current_constraint_loss(
    pred_currents: torch.Tensor,
    mosfet_info: torch.Tensor,
    terminal_features: torch.Tensor,
    diff_pair_constraints: Optional[torch.Tensor],
    mirror_constraints: Optional[torch.Tensor],
    output_stage_constraints: Optional[torch.Tensor],
    lambda_mirror_constraints: Optional[torch.Tensor],
    ptr: torch.Tensor,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    wl_feature_idx: int = 2,
    # Lambda mirror parameters
    pred_voltages: Optional[torch.Tensor] = None,
    node_names: Optional[List[List[str]]] = None,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    lambda_n: float = 0.05,
    ref_vds_node: str = 'nbias',
    mir_vds_node: str = 'vout',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified entry point for all current constraints.

    Args:
        pred_currents: Predicted currents [num_nodes] (normalized log scale)
        mosfet_info: MOSFET info tensor [num_mosfets, 7]
        terminal_features: Node features [num_nodes, num_features]
        diff_pair_constraints: [N, 3] or None
        mirror_constraints: [M, 2] or None
        output_stage_constraints: [K, 2] or None
        lambda_mirror_constraints: [L, 2] or None
        ptr: Node boundaries per graph [batch_size + 1]
        current_mean: Mean for denormalization
        current_std: Std for denormalization
        wl_feature_idx: Index of W/L ratio in feature vector
        pred_voltages: Predicted voltages [num_nodes] (normalized)
        node_names: List of node names per graph
        vdc_mean: Mean for voltage denormalization
        vdc_std: Std for voltage denormalization
        lambda_n: Channel length modulation parameter
        ref_vds_node: Node name for reference Vds
        mir_vds_node: Node name for mirror Vds

    Returns:
        Tuple of (total_constraint_loss, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss)
    """
    device = pred_currents.device

    diff_pair_loss = compute_diff_pair_loss(
        pred_currents, mosfet_info, diff_pair_constraints,
        ptr, current_mean, current_std
    )

    mirror_loss = compute_mirror_loss(
        pred_currents, mosfet_info, terminal_features, mirror_constraints,
        ptr, current_mean, current_std, wl_feature_idx
    )

    output_stage_loss = compute_output_stage_loss(
        pred_currents, mosfet_info, output_stage_constraints,
        ptr, current_mean, current_std
    )

    lambda_mirror_loss = compute_lambda_mirror_loss(
        pred_currents, pred_voltages, mosfet_info, terminal_features,
        lambda_mirror_constraints, node_names, ptr,
        current_mean, current_std, vdc_mean, vdc_std, wl_feature_idx,
        lambda_n, ref_vds_node, mir_vds_node
    )

    total_loss = diff_pair_loss + mirror_loss + output_stage_loss + lambda_mirror_loss

    return total_loss, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss
