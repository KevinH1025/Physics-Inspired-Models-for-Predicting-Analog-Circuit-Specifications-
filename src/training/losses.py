"""
Loss computation utilities for GNN training.

This module provides loss functions for voltage and current prediction.
To add a new loss type:
  1. Create a function: def <name>_loss(pred, target) -> Tensor
  2. Add it to LOSS_FUNCTIONS dict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .current_constraints import compute_current_constraint_loss


def get_device_graph_idx(num_devices: int, num_graphs: int,
                         device_ptr: torch.Tensor = None,
                         device: torch.device = None) -> torch.Tensor:
    """Get graph assignment index for each device, supporting variable counts per graph.

    Args:
        num_devices: Total number of devices across all graphs in the batch.
        num_graphs: Number of graphs in the batch.
        device_ptr: Optional [num_graphs + 1] cumulative size tensor.
            If provided, uses bucketize for variable-size support.
            If None, falls back to uniform integer division (legacy behavior).
        device: Torch device for output tensor.

    Returns:
        Tensor [num_devices] mapping each device to its graph index.
    """
    if device_ptr is not None:
        return torch.bucketize(
            torch.arange(num_devices, device=device),
            device_ptr[1:].to(device),
            right=True,
        )
    # Fallback: uniform (backward compat for single-topology batches)
    devices_per_graph = num_devices // num_graphs
    return torch.arange(num_devices, device=device) // devices_per_graph


# Individual Loss Functions
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return F.mse_loss(pred, target)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss (smooth L1). Less sensitive to outliers than MSE."""
    return F.huber_loss(pred, target, delta=delta)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error loss (L1)."""
    return F.l1_loss(pred, target)


# Loss Function Registry
LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'huber': huber_loss,
    'mae': mae_loss,
}


# Base Loss Computation
def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'mse',
    huber_delta: float = 1.0
) -> torch.Tensor:
    """
    Compute loss using the specified loss function.

    Args:
        pred: Predicted values
        target: Target values
        loss_type: Loss function name ('mse', 'huber', 'mae')
        huber_delta: Delta parameter for Huber loss

    Returns:
        Computed loss tensor

    Raises:
        ValueError: If loss_type is not recognized
    """
    if loss_type not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: {list(LOSS_FUNCTIONS.keys())}")

    loss_fn = LOSS_FUNCTIONS[loss_type]
    if loss_type == 'huber':
        return loss_fn(pred, target, delta=huber_delta)
    return loss_fn(pred, target)


# Component Losses
def compute_voltage_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'mse',
    huber_delta: float = 1.0,
    node_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute voltage prediction loss with optional per-node weighting.

    Args:
        pred: Predicted voltage values (normalized)
        target: Target voltage values (normalized)
        loss_type: Loss function name
        huber_delta: Delta for Huber loss
        node_weights: Optional weights per node (e.g., 2.0 for stage2 nodes)

    Returns:
        Voltage loss tensor (weighted mean if node_weights provided)
    """
    if node_weights is None:
        return compute_loss(pred, target, loss_type, huber_delta)

    # Compute per-element loss
    if loss_type == 'mse':
        element_loss = (pred - target) ** 2
    elif loss_type == 'mae':
        element_loss = (pred - target).abs()
    elif loss_type == 'huber':
        element_loss = F.huber_loss(pred, target, delta=huber_delta, reduction='none')
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'")

    # Weighted mean
    return (element_loss * node_weights).sum() / node_weights.sum()


def compute_current_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = 'mse',
    huber_delta: float = 1.0
) -> torch.Tensor:
    """
    Compute current prediction loss (masked).

    Args:
        pred: Predicted current values (normalized, all nodes)
        target: Target current values (normalized, all nodes)
        mask: Boolean mask for nodes with current targets
        loss_type: Loss function name
        huber_delta: Delta for Huber loss

    Returns:
        Current loss tensor (0 if no valid targets)
    """
    if pred is None or mask is None or not mask.any():
        return torch.tensor(0.0, device=target.device if target is not None else 'cpu')

    pred_masked = pred[mask]
    target_masked = target[mask]
    return compute_loss(pred_masked, target_masked, loss_type, huber_delta)


def apply_kcl_conservation_projection(
    node_currents: torch.Tensor,
    edge_index: torch.Tensor,
    num_terminals: torch.Tensor,
    train_mask: torch.Tensor,
    ptr: torch.Tensor,
    terminal_current_sign: torch.Tensor,
    current_mean: float,
    current_std: float,
    kcl_include_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Project predicted currents onto the KCL-satisfying subspace.

    For each valid KCL net, adjusts terminal currents so that
    Σ sign_i * I_i = 0 (KCL exactly satisfied). Differentiable.

    Works uniformly for 2-term and 3+ term nets in denormalized space.
    """
    device = node_currents.device
    num_nodes = train_mask.size(0)

    # Build batch index for each node
    graph_sizes = ptr[1:] - ptr[:-1]
    batch_idx = torch.repeat_interleave(torch.arange(len(graph_sizes), device=device), graph_sizes)

    # Terminal mask
    local_idx = torch.arange(num_nodes, device=device) - ptr[batch_idx]
    if isinstance(num_terminals, int):
        terminal_mask = local_idx < num_terminals
    else:
        terminal_mask = local_idx < num_terminals[batch_idx]

    internal_net_mask = train_mask & ~terminal_mask

    # Valid edges: terminal -> internal net
    src, dst = edge_index
    valid_edges = terminal_mask[src] & internal_net_mask[dst]
    if not valid_edges.any():
        return node_currents

    valid_src = src[valid_edges]
    valid_dst = dst[valid_edges]

    # Filter by kcl_include_mask (exclude gate, bulk terminals with 0 DC current)
    if kcl_include_mask is not None:
        kcl_eligible = kcl_include_mask[valid_src]
        valid_src = valid_src[kcl_eligible]
        valid_dst = valid_dst[kcl_eligible]

    if not valid_src.numel():
        return node_currents

    # Filter: ≥2 terminals per net
    terms_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
    terms_per_net.scatter_add_(0, valid_dst, torch.ones_like(valid_dst))
    has_enough = terms_per_net >= 2

    # Filter: both positive and negative sign terminals
    if terminal_current_sign is not None:
        sign_at_src = terminal_current_sign[valid_src]
        pos_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
        neg_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
        pos_per_net.scatter_add_(0, valid_dst, (sign_at_src > 0).long())
        neg_per_net.scatter_add_(0, valid_dst, (sign_at_src < 0).long())
        has_both_signs = (pos_per_net > 0) & (neg_per_net > 0)
    else:
        return node_currents  # Need signs for conservation

    valid_net_mask = internal_net_mask & has_enough & has_both_signs
    if not valid_net_mask.any():
        return node_currents

    # Keep only edges to valid nets
    net_valid = valid_net_mask[valid_dst]
    v_src = valid_src[net_valid]
    v_dst = valid_dst[net_valid]
    v_sign = sign_at_src[net_valid]

    # Denormalize to amps
    z = node_currents[v_src]
    amps = torch.pow(10, z * current_std + current_mean)

    # Per-net signed sum (violation) and total magnitude
    signed_sum = torch.zeros(num_nodes, device=device)
    signed_sum.scatter_add_(0, v_dst, v_sign * amps)
    total_abs = torch.zeros(num_nodes, device=device)
    total_abs.scatter_add_(0, v_dst, amps)

    # Per-terminal relative violation and adjustment factor
    rel_viol = signed_sum[v_dst] / (total_abs[v_dst] + 1e-12)
    factor = (1.0 - v_sign * rel_viol).clamp(min=1e-6)

    # Z-space correction: z_adj = z + log10(factor) / std
    delta_z = torch.log10(factor) / current_std

    # Apply correction to affected terminals only
    result = node_currents.clone()
    result[v_src] = z + delta_z

    return result


def compute_kcl_loss(
    node_currents: torch.Tensor,
    edge_index: torch.Tensor,
    num_terminals: torch.Tensor,
    train_mask: torch.Tensor,
    ptr: torch.Tensor,
    terminal_current_sign: torch.Tensor = None,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    gt_currents: torch.Tensor = None,
    kcl_threshold: float = 1e-6,  # 1µA threshold for ngspice numerical precision
    kcl_include_mask: torch.Tensor = None,
    kcl_min_current: float = 1e-9,  # 1nA min total current to include net in KCL
    return_stats: bool = False,
    kcl_mode: str = 'logsumexp',  # 'logsumexp', 'z_diff', or 'denorm'
    kcl_violation_threshold: float = 0.0,  # min relative violation to penalize (0 = penalize all)
    kcl_skip_two_term: bool = False,  # skip 2-term KCL loss (useful when 2-term is enforced in architecture)
    kcl_only_two_term: bool = False,  # only compute 2-term KCL loss, skip 3+ term nets
    kcl_huber_delta: float = 0.0,  # 0 = disabled (use MSE), >0 = Huber delta for 3-term logsumexp
    kcl_gt_filter: float = 0.1,  # max GT relative violation for multi-term nets (0 = disabled)
):
    """
    Compute Kirchhoff's Current Law loss on internal net nodes for batched graphs.
    Optimized vectorized implementation using scatter_add for GPU efficiency.

    KCL: Sum of signed currents at each internal net node = 0

    Only applies KCL to nets where ground truth satisfies KCL (truly internal nets).
    Nets where KCL is unsatisfiable (single terminal, all same sign) are filtered out.

    Args:
        node_currents: Predicted currents for all nodes in batch (normalized magnitudes)
        edge_index: Graph connectivity [2, num_edges]
        num_terminals: Number of terminals per graph [batch_size]
        train_mask: Mask for internal nodes (True = internal, where we predict voltage)
        ptr: Node boundaries between graphs [batch_size + 1]
        terminal_current_sign: Sign for each terminal (+1 for drain, -1 for source)
        current_mean: Mean for denormalization (log10 scale)
        current_std: Std for denormalization (log10 scale)
        gt_currents: Ground truth currents (normalized) for filtering valid KCL nets
        kcl_threshold: Threshold for GT sum to consider net valid for KCL (in Amps)
        kcl_include_mask: Mask for terminals to include in KCL (False for gate/bulk)
        kcl_min_current: Minimum total current for a net to be included in KCL (filters noise)

    Returns:
        Tuple of (loss, kcl_supervised_mask) where kcl_supervised_mask is a boolean
        tensor of shape [num_nodes] indicating which terminal nodes are supervised by KCL.
    """
    # Ensure float32 for scatter operations (model may output bfloat16 under AMP)
    node_currents = node_currents.float()
    if gt_currents is not None:
        gt_currents = gt_currents.float()
    device = node_currents.device
    num_nodes = train_mask.size(0)
    empty_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # Build batch index for each node (vectorized)
    graph_sizes = ptr[1:] - ptr[:-1]
    batch_idx = torch.repeat_interleave(torch.arange(len(graph_sizes), device=device), graph_sizes)

    # Terminal mask: local_idx < num_terminals for that graph
    local_idx = torch.arange(num_nodes, device=device) - ptr[batch_idx]
    if isinstance(num_terminals, int):
        terminal_mask = local_idx < num_terminals
    else:
        terminal_mask = local_idx < num_terminals[batch_idx]

    # Net node mask: only nets where we predict voltages (internal + output)
    # Excludes supply (vdda/gnda) and input nets — matches apply_kcl_conservation_projection
    internal_net_mask = train_mask & ~terminal_mask

    # Valid edges: terminal -> internal net
    src, dst = edge_index
    valid_edges = terminal_mask[src] & internal_net_mask[dst]

    if not valid_edges.any():
        if return_stats:
            return torch.tensor(0.0, device=device), empty_mask, {}
        return torch.tensor(0.0, device=device), empty_mask

    valid_src = src[valid_edges]
    valid_dst = dst[valid_edges]

    # Filter to KCL-eligible terminals (exclude gate/bulk with 0 DC current)
    # This must happen BEFORE computing sums, so predicted sums and GT sums
    # are computed consistently over the same set of terminals
    if kcl_include_mask is not None:
        kcl_eligible_pred = kcl_include_mask[valid_src]
        valid_src = valid_src[kcl_eligible_pred]
        valid_dst = valid_dst[kcl_eligible_pred]

    if not valid_src.numel():
        if return_stats:
            return torch.tensor(0.0, device=device), empty_mask, {}
        return torch.tensor(0.0, device=device), empty_mask

    # Filter out nets with fewer than 2 included terminals (KCL unsatisfiable)
    terms_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
    terms_per_net.scatter_add_(0, valid_dst, torch.ones_like(valid_dst))
    has_enough_terms = terms_per_net >= 2

    # Filter out nets where all terminals have the same sign (KCL unsatisfiable)
    if terminal_current_sign is not None:
        sign_at_src = terminal_current_sign[valid_src]
        pos_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
        neg_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
        pos_per_net.scatter_add_(0, valid_dst, (sign_at_src > 0).long())
        neg_per_net.scatter_add_(0, valid_dst, (sign_at_src < 0).long())
        has_both_signs = (pos_per_net > 0) & (neg_per_net > 0)
    else:
        has_both_signs = torch.ones(num_nodes, device=device, dtype=torch.bool)

    valid_net_mask = internal_net_mask & has_enough_terms & has_both_signs

    if not valid_net_mask.any():
        if return_stats:
            return torch.tensor(0.0, device=device), empty_mask, {}
        return torch.tensor(0.0, device=device), empty_mask

    # Split nets into 2-terminal and 3+ terminal groups
    two_term_mask = valid_net_mask & (terms_per_net == 2)
    multi_term_mask = valid_net_mask & (terms_per_net > 2)

    # Build kcl_supervised_mask: all terminal nodes participating in valid KCL equations
    # This includes terminals connected to valid 2-term nets and (if not z_diff) valid multi-term nets
    kcl_supervised_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    # Terminals connected to valid 2-term nets
    two_term_edges = two_term_mask[valid_dst]
    if two_term_edges.any():
        kcl_supervised_mask[valid_src[two_term_edges]] = True
    # Terminals connected to valid multi-term nets (only if we're actually computing multi-term loss)
    if kcl_mode != 'z_diff':
        multi_term_edges = multi_term_mask[valid_dst]
        if multi_term_edges.any():
            kcl_supervised_mask[valid_src[multi_term_edges]] = True

    # Get z-values (normalized predictions) at valid source terminals
    z_at_src = node_currents.squeeze()[valid_src]

    losses = []
    stats = {} if return_stats else None

    # --- 2-terminal nets: pure log-space loss (z_A - z_B)² ---
    # For 2-terminal nets, KCL means |I_A| = |I_B|, i.e. z_A = z_B in log space
    # Gradient is simply 2(z_A - z_B) — no exponential scaling
    if two_term_mask.any() and not kcl_skip_two_term:
        two_term_edges = two_term_mask[valid_dst]
        two_src = valid_src[two_term_edges]
        two_dst = valid_dst[two_term_edges]
        two_z = z_at_src[two_term_edges]

        # Sort by destination net to pair terminals
        sort_idx = torch.argsort(two_dst)
        two_z_sorted = two_z[sort_idx]

        # Each net has exactly 2 terminals, so reshape into pairs
        n_two_nets = two_z_sorted.numel() // 2
        if n_two_nets > 0:
            pairs = two_z_sorted[:n_two_nets * 2].reshape(n_two_nets, 2)

            two_term_loss = None
            if kcl_mode == 'denorm':
                # Denorm mode: relative violation in physical amps
                raw_pairs = torch.pow(10, pairs * current_std + current_mean)
                rel_viol_2t = (raw_pairs[:, 0] - raw_pairs[:, 1]) / (raw_pairs.sum(dim=1) + 1e-12)
                if kcl_violation_threshold > 0:
                    with torch.no_grad():
                        above_thresh = rel_viol_2t.abs() > kcl_violation_threshold
                    if stats is not None:
                        stats['n_two_term_above_thresh'] = above_thresh.sum().item()
                    if above_thresh.any():
                        two_term_loss = (rel_viol_2t[above_thresh] ** 2).mean()
                        losses.append(two_term_loss)
                else:
                    two_term_loss = (rel_viol_2t ** 2).mean()
                    losses.append(two_term_loss)
            else:
                # Z-space mode: (z_A - z_B)²
                if kcl_violation_threshold > 0:
                    with torch.no_grad():
                        raw_pairs = torch.pow(10, pairs * current_std + current_mean)
                        rel_viol = (raw_pairs[:, 0] - raw_pairs[:, 1]).abs() / (raw_pairs.sum(dim=1) + 1e-12)
                        above_thresh = rel_viol > kcl_violation_threshold
                    if stats is not None:
                        stats['n_two_term_above_thresh'] = above_thresh.sum().item()
                    if above_thresh.any():
                        two_term_loss = ((pairs[above_thresh, 0] - pairs[above_thresh, 1]) ** 2).mean()
                        losses.append(two_term_loss)
                else:
                    two_term_loss = ((pairs[:, 0] - pairs[:, 1]) ** 2).mean()
                    losses.append(two_term_loss)

            if stats is not None:
                stats['n_two_term'] = n_two_nets
                stats['two_term_loss'] = two_term_loss.item() if two_term_loss is not None else 0.0
                if gt_currents is not None:
                    with torch.no_grad():
                        gt_two_z = gt_currents.squeeze()[valid_src][two_term_edges]
                        gt_two_sorted = gt_two_z[sort_idx]
                        gt_pairs = gt_two_sorted[:n_two_nets * 2].reshape(n_two_nets, 2)
                        stats['gt_two_term_loss'] = ((gt_pairs[:, 0] - gt_pairs[:, 1]) ** 2).mean().item()

    # --- 3+ terminal nets: mode-dependent formulation ---
    # KCL: sum of positive-sign currents = sum of negative-sign currents
    if multi_term_mask.any() and terminal_current_sign is not None and kcl_mode != 'z_diff' and not kcl_only_two_term:
        multi_edges = multi_term_mask[valid_dst]
        multi_src = valid_src[multi_edges]
        multi_dst = valid_dst[multi_edges]
        multi_z = z_at_src[multi_edges]
        multi_sign = terminal_current_sign[multi_src]

        # Assign each multi-term net a compact index (0..N-1)
        unique_nets, net_indices = torch.unique(multi_dst, return_inverse=True)
        n_multi_nets = unique_nets.size(0)

        # GT filter: exclude nets where GT violates KCL beyond threshold
        valid_multi = torch.ones(n_multi_nets, device=device, dtype=torch.bool)
        n_multi_before_filter = n_multi_nets
        gt_z_at_src = gt_currents.squeeze()[valid_src] if gt_currents is not None else None
        if gt_currents is not None and kcl_gt_filter > 0:
            gt_multi_z = gt_z_at_src[multi_edges]
            gt_raw = torch.pow(10, gt_multi_z * current_std + current_mean)
            gt_signed = gt_raw * multi_sign
            gt_sums = torch.zeros(n_multi_nets, device=device)
            gt_sums.scatter_add_(0, net_indices, gt_signed)
            gt_abs_sums = torch.zeros(n_multi_nets, device=device)
            gt_abs_sums.scatter_add_(0, net_indices, gt_raw)
            gt_rel_viol = gt_sums.abs() / (gt_abs_sums + 1e-12)
            valid_multi = valid_multi & (gt_rel_viol < kcl_gt_filter)

        if stats is not None:
            stats['n_multi_total'] = n_multi_before_filter
            stats['n_multi_dropped'] = n_multi_before_filter - valid_multi.sum().item()

        if kcl_mode == 'denorm':
            # Denormalize to amps, compute relative violation: (Σsigned / Σabs)²
            raw_amps = torch.pow(10, multi_z * current_std + current_mean)
            signed_amps = raw_amps * multi_sign
            net_signed_sum = torch.zeros(n_multi_nets, device=device)
            net_signed_sum.scatter_add_(0, net_indices, signed_amps)
            net_abs_sum = torch.zeros(n_multi_nets, device=device)
            net_abs_sum.scatter_add_(0, net_indices, raw_amps)
            rel_viol = net_signed_sum / (net_abs_sum + 1e-12)

            if valid_multi.any():
                valid_rel_viol = rel_viol[valid_multi]
                if kcl_violation_threshold > 0:
                    with torch.no_grad():
                        above_thresh_denorm = valid_rel_viol.abs() > kcl_violation_threshold
                    if stats is not None:
                        stats['n_multi_above_thresh'] = above_thresh_denorm.sum().item()
                    if above_thresh_denorm.any():
                        multi_term_loss = (valid_rel_viol[above_thresh_denorm] ** 2).mean()
                        losses.append(multi_term_loss)
                        if stats is not None:
                            stats['multi_term_loss'] = multi_term_loss.item()
                else:
                    multi_term_loss = (valid_rel_viol ** 2).mean()
                    losses.append(multi_term_loss)
                    if stats is not None:
                        stats['multi_term_loss'] = multi_term_loss.item()

        elif kcl_mode == 'logsumexp':
            ln10 = 2.302585092994046  # ln(10)

            # Split into pos-sign and neg-sign edges
            pos_edge_mask = multi_sign > 0
            neg_edge_mask = multi_sign < 0

            NEG_INF = -1e9

            def scatter_logsumexp(z_vals, dst_idx, edge_mask, n_nets):
                """Compute logsumexp per net for a subset of edges.

                Converts z-scores back to log10|I| scale (z * std) before summing,
                so that logsumexp correctly computes log10(Σ|I|) up to a constant.
                The mean offset cancels when comparing pos vs neg sums.
                """
                if not edge_mask.any():
                    return torch.full((n_nets,), NEG_INF, device=device)
                z_sub = z_vals[edge_mask] * current_std * ln10
                dst_sub = dst_idx[edge_mask]
                max_vals = torch.full((n_nets,), NEG_INF, device=device)
                max_vals.scatter_reduce_(0, dst_sub, z_sub, reduce='amax', include_self=True)
                z_shifted = torch.exp(z_sub - max_vals[dst_sub])
                sum_exp = torch.zeros(n_nets, device=device)
                sum_exp.scatter_add_(0, dst_sub, z_shifted)
                result = (max_vals + torch.log(sum_exp + 1e-30)) / ln10
                has_edges = torch.zeros(n_nets, device=device, dtype=torch.bool)
                has_edges.scatter_(0, dst_sub, torch.ones_like(dst_sub, dtype=torch.bool))
                result[~has_edges] = NEG_INF
                return result

            log_pos_sum = scatter_logsumexp(multi_z, net_indices, pos_edge_mask, n_multi_nets)
            log_neg_sum = scatter_logsumexp(multi_z, net_indices, neg_edge_mask, n_multi_nets)

            # Valid nets: both pos and neg groups have edges
            valid_multi = valid_multi & (log_pos_sum > NEG_INF + 1) & (log_neg_sum > NEG_INF + 1)

            if valid_multi.any():
                diffs = log_pos_sum[valid_multi] - log_neg_sum[valid_multi]

                multi_term_loss = None
                if kcl_violation_threshold > 0:
                    # Only penalize nets where relative violation exceeds threshold
                    with torch.no_grad():
                        raw_pos = torch.pow(10, log_pos_sum[valid_multi])
                        raw_neg = torch.pow(10, log_neg_sum[valid_multi])
                        rel_viol_3t = (raw_pos - raw_neg).abs() / (raw_pos + raw_neg + 1e-12)
                        above_thresh_3t = rel_viol_3t > kcl_violation_threshold
                    if stats is not None:
                        stats['n_multi_above_thresh'] = above_thresh_3t.sum().item()
                    if above_thresh_3t.any():
                        d = diffs[above_thresh_3t]
                        if kcl_huber_delta > 0:
                            multi_term_loss = F.huber_loss(d, torch.zeros_like(d), delta=kcl_huber_delta, reduction='mean')
                        else:
                            multi_term_loss = (d ** 2).mean()
                        losses.append(multi_term_loss)
                else:
                    if kcl_huber_delta > 0:
                        multi_term_loss = F.huber_loss(diffs, torch.zeros_like(diffs), delta=kcl_huber_delta, reduction='mean')
                    else:
                        multi_term_loss = (diffs ** 2).mean()
                    losses.append(multi_term_loss)

                if stats is not None:
                    stats['multi_term_loss'] = multi_term_loss.item() if multi_term_loss is not None else 0.0

                    # GT KCL floor: same logsumexp but with GT currents
                    if gt_currents is not None:
                        with torch.no_grad():
                            gt_multi_z_norm = gt_z_at_src[multi_edges]
                            gt_log_pos = scatter_logsumexp(gt_multi_z_norm, net_indices, pos_edge_mask, n_multi_nets)
                            gt_log_neg = scatter_logsumexp(gt_multi_z_norm, net_indices, neg_edge_mask, n_multi_nets)
                            gt_diffs = gt_log_pos[valid_multi] - gt_log_neg[valid_multi]
                            stats['gt_multi_term_loss'] = (gt_diffs ** 2).mean().item()

    if not losses:
        if return_stats:
            return torch.tensor(0.0, device=device), kcl_supervised_mask, stats
        return torch.tensor(0.0, device=device), kcl_supervised_mask

    # Average the losses (both are scale-invariant, similar magnitude)
    total_loss = sum(losses) / len(losses)
    if return_stats:
        return total_loss, kcl_supervised_mask, stats
    return total_loss, kcl_supervised_mask


def compute_kcl_per_net_debug(
    node_currents: torch.Tensor,
    edge_index: torch.Tensor,
    num_terminals: torch.Tensor,
    train_mask: torch.Tensor,
    ptr: torch.Tensor,
    terminal_current_sign: torch.Tensor,
    current_mean: float,
    current_std: float,
    node_names: list,
    kcl_include_mask: torch.Tensor = None,
    graph_idx: int = 0,
) -> dict:
    """
    Compute per-net KCL violations for debugging.

    Returns dict of {net_name: relative_violation} for internal nets.
    """
    # Get graph boundaries
    start = ptr[graph_idx].item()
    end = ptr[graph_idx + 1].item()
    n_terms = num_terminals[graph_idx].item() if hasattr(num_terminals[graph_idx], 'item') else num_terminals[graph_idx]

    # Get names for this graph
    if isinstance(node_names[0], list):
        names = node_names[graph_idx]
    else:
        names = node_names[start:end]

    results = {}

    # For each internal net
    for local_idx in range(n_terms, end - start):
        global_idx = start + local_idx
        if not train_mask[global_idx]:
            continue
        net_name = names[local_idx] if local_idx < len(names) else f"net_{local_idx}"

        # Find terminals connected to this net
        src, dst = edge_index
        mask = (dst == global_idx) & (src >= start) & (src < start + n_terms)
        term_indices = src[mask]

        if len(term_indices) == 0:
            continue

        # Compute sum
        total_signed = 0.0
        total_abs = 0.0

        for term_idx in term_indices:
            if kcl_include_mask is not None and not kcl_include_mask[term_idx]:
                continue

            pred_norm = node_currents[term_idx].item()
            pred_raw = 10 ** (pred_norm * current_std + current_mean)
            sign = terminal_current_sign[term_idx].item()

            total_signed += pred_raw * sign
            total_abs += pred_raw

        if total_abs > 1e-9:  # Filter nets with < 1nA total current (e.g., vc, vg2)
            rel_viol = abs(total_signed) / total_abs
            results[net_name] = rel_viol

    return results


# gm Self-Consistency Physics Loss
def compute_gm_physics_loss(
    ss_gm_pred: torch.Tensor,
    pred_currents: torch.Tensor,
    full_voltage_pred: torch.Tensor,
    mosfet_info: torch.Tensor,
    node_mosfet_vth: torch.Tensor,
    mosfet_region_labels: torch.Tensor,
    ptr: torch.Tensor,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    min_vov: float = 0.0,
    mosfet_gt_vov: torch.Tensor = None,
    mosfet_ptr: torch.Tensor = None,
    ss_gds_pred: torch.Tensor = None,
    ss_gds_mean: float = 0.0,
    ss_gds_std: float = 1.0,
    use_clm: bool = False,
    node_voltage_targets: torch.Tensor = None,
) -> torch.Tensor:
    """
    Saturation gm physics self-consistency loss.

    Without CLM correction (use_clm=False):
        gm_pred vs gm_physics = 2*I_D / Vov
        Compared in log10 space.

    With CLM correction (use_clm=True):
        gm·Vov + 2·gds·Vds = 2·Id
        Compared as: log10(gm·Vov + 2·gds·Vds) vs log10(2·Id)
        This ties together all four prediction heads (V, I, gm, gds)
        and accounts for channel length modulation.

    Only applied to MOSFETs in saturation with Vov >= min_vov.

    Args:
        ss_gm_pred: [num_nodes] SS head gm prediction (z-score normalized log10)
        pred_currents: [num_nodes] current head predictions (z-score normalized log10)
        full_voltage_pred: [num_nodes] voltage predictions (z-score normalized)
        mosfet_info: [num_mosfets, 7] cols: gate_t, drain_t, source_t, gate_n, drain_n, source_n, is_nmos
        node_mosfet_vth: [num_nodes] SPICE Vth at drain terminal positions (|Vth| in volts)
        mosfet_region_labels: [num_mosfets] 0=cutoff, 1=triode, 2=saturation
        ptr: [batch_size+1] node boundaries
        vdc_mean/std: voltage denormalization (V_real = pred * vdc_std + vdc_mean)
        current_mean/std: current denormalization (I_real = 10^(pred * std + mean))
        ss_gm_mean/std: SS gm denormalization (log10_gm = pred * std + mean)
        min_vov: minimum overdrive voltage (V) — transistors with Vov < min_vov are excluded.
        mosfet_gt_vov: [num_mosfets] pre-computed ground truth Vov (from SPICE gm/I_D).
                       If provided, used for min_vov filtering instead of predicted Vov.
        ss_gds_pred: [num_nodes] SS head gds prediction (z-score normalized log10). Required if use_clm=True.
        ss_gds_mean/std: SS gds denormalization (log10_gds = pred * std + mean)
        use_clm: if True, use CLM-corrected equation gm·Vov + 2·gds·Vds = 2·Id
    """
    device = ss_gm_pred.device

    if (mosfet_info is None or len(mosfet_info) == 0 or
        pred_currents is None or full_voltage_pred is None or
        node_mosfet_vth is None):
        return torch.tensor(0.0, device=device)

    if use_clm and ss_gds_pred is None:
        return torch.tensor(0.0, device=device)

    num_mosfets = len(mosfet_info)
    num_graphs = len(ptr) - 1

    # Build global mosfet indices with batch offsets
    mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, device)
    node_offsets = ptr[mosfet_graph_idx]

    # Terminal indices (for I_D and gm)
    drain_term_idx = mosfet_info[:, 1] + node_offsets
    # Net node indices (for Vgs, Vds — voltages predicted at net nodes)
    gate_net_idx = mosfet_info[:, 3] + node_offsets
    drain_net_idx = mosfet_info[:, 4] + node_offsets
    source_net_idx = mosfet_info[:, 5] + node_offsets

    # Saturation mask + Vth validity
    sat_mask = (mosfet_region_labels.to(device) == 2)
    vth_at_drain = node_mosfet_vth[drain_term_idx]
    valid_mask = sat_mask & (vth_at_drain.abs() > 1e-6)

    # Apply min_vov filter using GT Vov (stable) or predicted Vov (fallback)
    has_gt_vov_filter = mosfet_gt_vov is not None and min_vov > 0
    if has_gt_vov_filter:
        gt_vov = mosfet_gt_vov.to(device)
        valid_mask = valid_mask & (gt_vov >= min_vov)

    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    # gm from SS head (denormalize to real value)
    # ss_gm_pred is per-MOSFET (from concat head), not per-node
    log10_gm_pred = ss_gm_pred[valid_mask] * ss_gm_std + ss_gm_mean

    # I_D from current head (denormalize to real amps)
    Id_real = torch.pow(10, pred_currents[drain_term_idx[valid_mask]] * current_std + current_mean).clamp(min=1e-12)

    # Voltages for Vov computation: use GT voltages when available (accurate), else predicted (noisy)
    if node_voltage_targets is not None:
        V_gate = node_voltage_targets[gate_net_idx[valid_mask]]
        V_source = node_voltage_targets[source_net_idx[valid_mask]]
    else:
        V_gate = full_voltage_pred[gate_net_idx[valid_mask]] * vdc_std + vdc_mean
        V_source = full_voltage_pred[source_net_idx[valid_mask]] * vdc_std + vdc_mean
    Vgs = V_gate - V_source

    # Vth from SPICE (already |Vth| in volts)
    Vth = vth_at_drain[valid_mask]

    # Compute overdrive with correct sign per device type
    is_nmos = mosfet_info[:, 6][valid_mask].bool()
    Vov = torch.where(is_nmos, Vgs - Vth, -Vgs - Vth)

    # Filter by predicted Vov only when GT Vov filter was NOT applied
    # (GT filter is stable; predicted Vov filter lets model game the loss)
    vov_threshold = max(min_vov, 1e-3)
    if not has_gt_vov_filter:
        vov_mask = Vov >= vov_threshold
        if not vov_mask.any():
            return torch.tensor(0.0, device=device)
        log10_gm_pred = log10_gm_pred[vov_mask]
        Id_real = Id_real[vov_mask]
        Vov_filtered = Vov[vov_mask].clamp(min=vov_threshold)
    else:
        Vov_filtered = Vov.clamp(min=vov_threshold)

    if use_clm:
        # CLM-corrected: gm·Vov + 2·gds·Vds = 2·Id
        # Compared in log10 space: log10(LHS) vs log10(RHS)
        gm_real = torch.pow(10, log10_gm_pred).clamp(min=1e-15)

        # gds from SS head (per-MOSFET) — apply vov_mask only when predicted Vov filter was used
        if not has_gt_vov_filter:
            log10_gds_pred = ss_gds_pred[valid_mask][vov_mask] * ss_gds_std + ss_gds_mean
        else:
            log10_gds_pred = ss_gds_pred[valid_mask] * ss_gds_std + ss_gds_mean
        gds_real = torch.pow(10, log10_gds_pred).clamp(min=1e-15)

        # Vds: use GT voltages when available, else predicted
        if not has_gt_vov_filter:
            sel_mask = vov_mask
        else:
            sel_mask = slice(None)
        if node_voltage_targets is not None:
            V_drain = node_voltage_targets[drain_net_idx[valid_mask]][sel_mask]
            V_src = node_voltage_targets[source_net_idx[valid_mask]][sel_mask]
        else:
            V_drain = full_voltage_pred[drain_net_idx[valid_mask]][sel_mask] * vdc_std + vdc_mean
            V_src = full_voltage_pred[source_net_idx[valid_mask]][sel_mask] * vdc_std + vdc_mean
        Vds_raw = V_drain - V_src
        Vds = torch.where(is_nmos if has_gt_vov_filter else is_nmos[vov_mask], Vds_raw, -Vds_raw).clamp(min=1e-6)

        lhs = (gm_real * Vov_filtered + 2.0 * gds_real * Vds).clamp(min=1e-12)
        rhs = (2.0 * Id_real).clamp(min=1e-12)
        return F.mse_loss(torch.log10(lhs), torch.log10(rhs))
    else:
        # 0th order: gm_physics = 2 * I_D / Vov
        gm_physics = (2.0 * Id_real / Vov_filtered).clamp(min=1e-12)
        log10_gm_physics = torch.log10(gm_physics)
        return F.mse_loss(log10_gm_pred, log10_gm_physics)


def compute_cutoff_physics_loss(
    ss_gm_pred: torch.Tensor,
    pred_currents: torch.Tensor,
    mosfet_info: torch.Tensor,
    mosfet_region_labels: torch.Tensor,
    ptr: torch.Tensor,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    n_nmos: float = 1.5,
    n_pmos: float = 2.0,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """
    Subthreshold gm physics loss for cutoff devices: gm = Id / (n * Vt).

    In subthreshold, gm/Id = 1/(n*Vt) where n is the subthreshold slope
    factor (process-dependent, ~1.5 for NMOS and ~2.0 for PMOS in SKY130)
    and Vt = kT/q = 25.85mV at 27°C.

    Compared in log10 space: log10(gm_pred) vs log10(Id / (n * Vt)).

    Args:
        ss_gm_pred: [num_nodes] SS head gm prediction (z-score normalized log10)
        pred_currents: [num_nodes] current head predictions (z-score normalized log10)
        mosfet_info: [num_mosfets, 7] cols: gate_t, drain_t, source_t, gate_n, drain_n, source_n, is_nmos
        mosfet_region_labels: [num_mosfets] 0=cutoff, 1=triode, 2=saturation
        ptr: [batch_size+1] node boundaries
        current_mean/std: current denormalization (I_real = 10^(pred * std + mean))
        ss_gm_mean/std: SS gm denormalization (log10_gm = pred * std + mean)
        n_nmos: subthreshold slope factor for NMOS (SKY130 default: 1.5)
        n_pmos: subthreshold slope factor for PMOS (SKY130 default: 2.0)
    """
    device = ss_gm_pred.device
    Vt = 0.02585  # kT/q at 27°C

    if (mosfet_info is None or len(mosfet_info) == 0 or
        pred_currents is None or mosfet_region_labels is None):
        return torch.tensor(0.0, device=device)

    num_mosfets = len(mosfet_info)
    num_graphs = len(ptr) - 1

    mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, device)
    node_offsets = ptr[mosfet_graph_idx]

    drain_term_idx = mosfet_info[:, 1] + node_offsets

    # Cutoff mask (region == 0)
    cutoff_mask = (mosfet_region_labels.to(device) == 0)
    if not cutoff_mask.any():
        return torch.tensor(0.0, device=device)

    # gm from SS head (per-MOSFET, denormalize to log10 scale)
    log10_gm_pred = ss_gm_pred[cutoff_mask] * ss_gm_std + ss_gm_mean

    # Id from current head (denormalize to real amps)
    Id_real = torch.pow(10, pred_currents[drain_term_idx[cutoff_mask]] * current_std + current_mean).clamp(min=1e-15)

    # Build per-device n*Vt
    is_nmos = mosfet_info[:, 6][cutoff_mask].bool()
    n_vt = torch.where(is_nmos,
                       torch.tensor(n_nmos * Vt, device=device),
                       torch.tensor(n_pmos * Vt, device=device))

    # gm_physics = Id / (n * Vt)
    gm_physics = (Id_real / n_vt).clamp(min=1e-15)
    log10_gm_physics = torch.log10(gm_physics)

    return F.mse_loss(log10_gm_pred, log10_gm_physics)


def compute_triode_physics_loss(
    ss_gm_pred: torch.Tensor,
    ss_gds_pred: torch.Tensor,
    pred_currents: torch.Tensor,
    full_voltage_pred: torch.Tensor,
    mosfet_info: torch.Tensor,
    node_mosfet_vth: torch.Tensor,
    mosfet_region_labels: torch.Tensor,
    ptr: torch.Tensor,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    ss_gds_mean: float = 0.0,
    ss_gds_std: float = 1.0,
    min_vov: float = 0.0,
    eq1_enabled: bool = True,
    eq1_weight: float = 1.0,
    eq2_enabled: bool = True,
    eq2_weight: float = 1.0,
    eq3_enabled: bool = True,
    eq3_weight: float = 1.0,
    eq2_max_vds_vov: float = 1.0,
    mosfet_gt_vov: torch.Tensor = None,
    mosfet_ptr: torch.Tensor = None,
    node_voltage_targets: torch.Tensor = None,
) -> tuple:
    """
    Triode physics regularizer losses (training only).

    Three equations for MOSFETs in triode region (region_label == 1):
      Eq1: gm = I_DS / (Vov - Vds/2)
      Eq2: gds = (I_DS/Vds) * (Vov - Vds) / (Vov - Vds/2)
      Eq3: gm * (Vov - Vds) = gds * Vds   [self-consistency]

    All comparisons are in log10 space (MSE of log10 values).

    Args:
        ss_gm_pred: [num_nodes] SS head gm prediction (z-score normalized log10)
        ss_gds_pred: [num_nodes] SS head gds prediction (z-score normalized log10)
        pred_currents: [num_nodes] current head predictions (z-score normalized log10)
        full_voltage_pred: [num_nodes] voltage predictions (z-score normalized)
        mosfet_info: [num_mosfets, 7] cols: gate_t, drain_t, source_t, gate_n, drain_n, source_n, is_nmos
        node_mosfet_vth: [num_nodes] SPICE |Vth| at drain terminal positions
        mosfet_region_labels: [num_mosfets] 0=cutoff, 1=triode, 2=saturation
        ptr: [batch_size+1] node boundaries
        min_vov: minimum overdrive voltage filter (V)
        eq1/2/3_enabled: enable each equation
        eq1/2/3_weight: relative weight for each equation
        mosfet_gt_vov: [num_mosfets] pre-computed ground truth Vov (from SPICE gm/I_D).
                       If provided, used for min_vov filtering instead of predicted Vov.

    Returns:
        (combined_loss, eq1_loss, eq2_loss, eq3_loss) as tensors
    """
    device = ss_gm_pred.device
    zero = torch.tensor(0.0, device=device)

    if (mosfet_info is None or len(mosfet_info) == 0 or
        node_mosfet_vth is None or mosfet_region_labels is None):
        return zero, zero, zero, zero

    # Need at least one equation enabled
    if not (eq1_enabled or eq2_enabled or eq3_enabled):
        return zero, zero, zero, zero

    # Eq1/Eq2 need current predictions
    needs_current = (eq1_enabled or eq2_enabled) and pred_currents is not None
    # Eq3 only needs gm+gds (no current), but all need voltages for Vov/Vds
    if full_voltage_pred is None:
        return zero, zero, zero, zero
    if ss_gm_pred is None and (eq1_enabled or eq3_enabled):
        return zero, zero, zero, zero
    if ss_gds_pred is None and (eq2_enabled or eq3_enabled):
        return zero, zero, zero, zero

    num_mosfets = len(mosfet_info)
    num_graphs = len(ptr) - 1

    # Build global mosfet indices with batch offsets
    mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, device)
    node_offsets = ptr[mosfet_graph_idx]

    # Terminal indices (for I_D, gm, gds)
    drain_term_idx = mosfet_info[:, 1] + node_offsets
    # Net node indices (for voltages)
    gate_net_idx = mosfet_info[:, 3] + node_offsets
    drain_net_idx = mosfet_info[:, 4] + node_offsets
    source_net_idx = mosfet_info[:, 5] + node_offsets

    # Triode mask + Vth validity
    triode_mask = (mosfet_region_labels.to(device) == 1)
    vth_at_drain = node_mosfet_vth[drain_term_idx]
    valid_mask = triode_mask & (vth_at_drain.abs() > 1e-6)

    # Apply min_vov filter using GT Vov (stable) or predicted Vov (fallback)
    if mosfet_gt_vov is not None and min_vov > 0:
        gt_vov = mosfet_gt_vov.to(device)
        valid_mask = valid_mask & (gt_vov >= min_vov)

    if not valid_mask.any():
        return zero, zero, zero, zero

    # Voltages for physics equations: use GT when available (accurate), else predicted (detached)
    if node_voltage_targets is not None:
        V_gate = node_voltage_targets[gate_net_idx[valid_mask]]
        V_drain = node_voltage_targets[drain_net_idx[valid_mask]]
        V_source = node_voltage_targets[source_net_idx[valid_mask]]
    else:
        V_gate = full_voltage_pred[gate_net_idx[valid_mask]].detach() * vdc_std + vdc_mean
        V_drain = full_voltage_pred[drain_net_idx[valid_mask]].detach() * vdc_std + vdc_mean
        V_source = full_voltage_pred[source_net_idx[valid_mask]].detach() * vdc_std + vdc_mean

    # Compute Vgs, Vds, Vov with correct NMOS/PMOS polarity
    Vgs_raw = V_gate - V_source
    Vds_raw = V_drain - V_source
    Vth = vth_at_drain[valid_mask]
    is_nmos = mosfet_info[:, 6][valid_mask].bool()

    Vov = torch.where(is_nmos, Vgs_raw - Vth, -Vgs_raw - Vth)
    Vds = torch.where(is_nmos, Vds_raw, -Vds_raw)

    # Dynamic filter: predicted Vov > min_vov, Vds > 0, and Vov > Vds (true triode condition)
    vov_threshold = min_vov if min_vov > 0 else 1e-3
    vov_mask = (Vov > vov_threshold) & (Vds > 1e-6) & (Vov > Vds)
    if not vov_mask.any():
        return zero, zero, zero, zero

    # Apply filter
    Vov_f = Vov[vov_mask].clamp(min=1e-6)
    Vds_f = Vds[vov_mask].clamp(min=1e-6)
    drain_term_f = drain_term_idx[valid_mask][vov_mask]

    # Denormalize gm/gds predictions (per-MOSFET, log10 scale)
    log10_gm_pred_f = ss_gm_pred[valid_mask][vov_mask] * ss_gm_std + ss_gm_mean
    log10_gds_pred_f = ss_gds_pred[valid_mask][vov_mask] * ss_gds_std + ss_gds_mean

    # Denormalize current (real amps, still per-node)
    if needs_current:
        Id_real_f = torch.pow(10, pred_currents[drain_term_f] * current_std + current_mean).clamp(min=1e-12)

    # --- Eq1: gm = I_DS / (Vov - Vds/2) ---
    eq1_loss = zero
    if eq1_enabled and needs_current:
        denom1 = (Vov_f - Vds_f / 2.0).clamp(min=1e-6)
        gm_physics = (Id_real_f / denom1).clamp(min=1e-12)
        log10_gm_physics = torch.log10(gm_physics)
        eq1_loss = F.mse_loss(log10_gm_pred_f, log10_gm_physics)

    # --- Eq2: gds = (I_DS/Vds) * (Vov - Vds) / (Vov - Vds/2) ---
    # Near the saturation boundary (Vds → Vov), (Vov-Vds) → 0 and the equation
    # becomes numerically unstable. Filter by max Vds/Vov ratio.
    eq2_loss = zero
    if eq2_enabled and needs_current:
        eq2_mask = torch.ones_like(Vov_f, dtype=torch.bool)
        if eq2_max_vds_vov < 1.0:
            eq2_mask = (Vds_f / Vov_f) < eq2_max_vds_vov
        if eq2_mask.any():
            numer2 = (Vov_f[eq2_mask] - Vds_f[eq2_mask]).clamp(min=1e-6)
            denom2 = (Vov_f[eq2_mask] - Vds_f[eq2_mask] / 2.0).clamp(min=1e-6)
            gds_physics = (Id_real_f[eq2_mask] / Vds_f[eq2_mask] * numer2 / denom2).clamp(min=1e-12)
            log10_gds_physics = torch.log10(gds_physics)
            eq2_loss = F.mse_loss(log10_gds_pred_f[eq2_mask], log10_gds_physics)

    # --- Eq3: gm * (Vov - Vds) = gds * Vds  [self-consistency] ---
    # In log space: log10(gm) + log10(Vov - Vds) = log10(gds) + log10(Vds)
    eq3_loss = zero
    if eq3_enabled:
        vov_minus_vds = (Vov_f - Vds_f).clamp(min=1e-6)
        lhs = log10_gm_pred_f + torch.log10(vov_minus_vds)
        rhs = log10_gds_pred_f + torch.log10(Vds_f)
        eq3_loss = F.mse_loss(lhs, rhs)

    combined = eq1_weight * eq1_loss + eq2_weight * eq2_loss + eq3_weight * eq3_loss
    return combined, eq1_loss, eq2_loss, eq3_loss


def compute_vov_loss(vov_pred: torch.Tensor, mosfet_gt_vov: torch.Tensor,
                     mosfet_vov_valid: torch.Tensor = None) -> torch.Tensor:
    """MSE loss on predicted vs GT Vov, skipping cutoff devices."""
    if mosfet_vov_valid is not None:
        mask = mosfet_vov_valid
    else:
        mask = (mosfet_gt_vov > 0) & (mosfet_gt_vov < 1.8)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=vov_pred.device)
    return F.mse_loss(vov_pred[mask], mosfet_gt_vov[mask])


def compute_vth_loss(vth_pred: torch.Tensor, mosfet_vth: torch.Tensor) -> torch.Tensor:
    """MSE loss on predicted vs GT Vth, skipping missing values."""
    valid = mosfet_vth.abs() > 1e-6
    if valid.sum() == 0:
        return torch.tensor(0.0, device=vth_pred.device)
    return F.mse_loss(vth_pred[valid], mosfet_vth[valid])


# Supervised gm/gds Prediction Loss (mask-based, like voltage/current)
def compute_ss_loss(
    gm_pred: torch.Tensor,
    gds_pred: torch.Tensor,
    mosfet_gm: torch.Tensor,
    mosfet_gds: torch.Tensor,
    gm_mean: float,
    gm_std: float,
    gds_mean: float,
    gds_std: float,
    huber_delta: float = 0.0,
    region_stats: dict = None,
    mosfet_region_labels: torch.Tensor = None,
) -> tuple:
    """
    Supervised loss for per-MOSFET gm/gds predictions.

    Predictions are per-MOSFET (from gate+drain+source terminal concat).
    Targets are raw linear-scale mosfet_gm/mosfet_gds, normalized inline.

    If region_stats is provided, uses per-region (mean, std) for target normalization.

    Args:
        gm_pred: [num_mosfets] predicted z-scored log10(gm)
        gds_pred: [num_mosfets] predicted z-scored log10(gds)
        mosfet_gm: [num_mosfets] raw gm values (linear scale)
        mosfet_gds: [num_mosfets] raw gds values (linear scale)
        gm_mean/gm_std: global z-score normalization stats for log10(gm)
        gds_mean/gds_std: global z-score normalization stats for log10(gds)
        region_stats: Optional per-region stats dict {0: {'gm_mean', 'gm_std', ...}, ...}
        mosfet_region_labels: Optional [num_mosfets] region labels (0=cutoff, 1=triode, 2=sat)

    Returns:
        Tuple of (gm_loss, gds_loss) scalar tensors
    """
    device = gm_pred.device
    mosfet_gm = mosfet_gm.to(device)
    mosfet_gds = mosfet_gds.to(device)
    valid = mosfet_gm > 1e-12

    if valid.any():
        log_gm = torch.log10(mosfet_gm[valid])
        log_gds = torch.log10(mosfet_gds[valid].clamp(min=1e-20))

        if region_stats is not None and mosfet_region_labels is not None:
            labels = mosfet_region_labels.to(device)[valid]
            gm_means = torch.full_like(log_gm, gm_mean)
            gm_stds = torch.full_like(log_gm, gm_std)
            gds_means = torch.full_like(log_gds, gds_mean)
            gds_stds = torch.full_like(log_gds, gds_std)
            for r in range(3):
                rmask = (labels == r)
                if rmask.any():
                    gm_means[rmask] = region_stats[r]['gm_mean']
                    gm_stds[rmask] = region_stats[r]['gm_std']
                    gds_means[rmask] = region_stats[r]['gds_mean']
                    gds_stds[rmask] = region_stats[r]['gds_std']
            gm_target = (log_gm - gm_means) / gm_stds
            gds_target = (log_gds - gds_means) / gds_stds
        else:
            gm_target = (log_gm - gm_mean) / gm_std
            gds_target = (log_gds - gds_mean) / gds_std

        if huber_delta > 0:
            gm_loss = F.huber_loss(gm_pred[valid], gm_target, delta=huber_delta)
            gds_loss = F.huber_loss(gds_pred[valid], gds_target, delta=huber_delta)
        else:
            gm_loss = F.mse_loss(gm_pred[valid], gm_target)
            gds_loss = F.mse_loss(gds_pred[valid], gds_target)
    else:
        gm_loss = torch.tensor(0.0, device=device)
        gds_loss = torch.tensor(0.0, device=device)

    return gm_loss, gds_loss


# IV Model ID Prediction Loss
def compute_iv_id_loss(
    log_abs_id_pred: torch.Tensor,
    mosfet_info: torch.Tensor,
    node_current_targets: torch.Tensor,
    current_mean: float,
    current_std: float,
    ptr: torch.Tensor,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """
    MSE loss on IV model's log10(|ID|) vs GT drain current.

    GT drain current comes from node_current_targets at drain terminal indices,
    which are stored in z-scored log10(|I|) space.

    Args:
        log_abs_id_pred: [num_mosfets] predicted log10(|ID|) from IV model
        mosfet_info: [num_mosfets, 7] MOSFET info tensor
        node_current_targets: [num_nodes] z-scored log10(|I|) targets
        current_mean: Mean for current denormalization (log10 scale)
        current_std: Std for current denormalization (log10 scale)
        ptr: Node boundaries per graph [num_graphs + 1]
        mosfet_ptr: Cumulative MOSFET counts per graph [num_graphs + 1], or None

    Returns:
        Scalar loss tensor
    """
    device = log_abs_id_pred.device
    num_mosfets = mosfet_info.shape[0]
    drain_idx = mosfet_info[:, 1].long()

    # Compute per-MOSFET node offsets for batched graphs
    num_graphs = ptr.shape[0] - 1
    if mosfet_ptr is not None:
        mg_idx = torch.bucketize(
            torch.arange(num_mosfets, device=device),
            mosfet_ptr[1:].to(device), right=True,
        )
    else:
        devices_per_graph = num_mosfets // num_graphs
        mg_idx = torch.arange(num_mosfets, device=device) // devices_per_graph
    offsets = ptr[mg_idx]

    # GT drain current: z-scored log10 -> denorm to log10(|I|)
    gt_zscore = node_current_targets.to(device)[drain_idx + offsets]
    gt_log_abs_id = gt_zscore * current_std + current_mean  # log10(|ID|)

    valid = torch.isfinite(gt_log_abs_id) & (gt_log_abs_id > -15)
    if valid.any():
        return F.mse_loss(log_abs_id_pred[valid], gt_log_abs_id[valid])
    return torch.tensor(0.0, device=device)


# AC Prediction Loss
def compute_ac_loss(
    ac_pred: torch.Tensor,
    ac_valid: torch.Tensor,
    ac_mean: torch.Tensor,
    ac_std: torch.Tensor,
    ac_components: list,
    ac_ugbw: torch.Tensor = None,
    ac_pm: torch.Tensor = None,
    ac_am: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute AC prediction loss for configurable components.

    Each AC component (UGBW, PM, AM) can be individually enabled.
    Only uses samples where ac_valid is True (SPICE AC analysis converged).
    Targets are z-score normalized: (value - mean) / std.

    Args:
        ac_pred: [B, N] predicted AC quantities (N = number of enabled components)
        ac_valid: [B] boolean mask for valid AC measurements
        ac_mean: [N] mean for each enabled component from training set
        ac_std: [N] std for each enabled component from training set
        ac_components: List of enabled component names, e.g. ['ugbw', 'pm', 'am']
        ac_ugbw: [B] ground truth UGBW in Hz (if 'ugbw' in ac_components)
        ac_pm: [B] ground truth phase margin in degrees (if 'pm' in ac_components)
        ac_am: [B] ground truth gain margin in dB (if 'am' in ac_components)

    Returns:
        Scalar loss tensor (MSE on normalized targets, only valid samples)
    """
    device = ac_pred.device
    zero = torch.tensor(0.0, device=device)

    if ac_valid is None or not ac_valid.any() or len(ac_components) == 0:
        per_component = {comp: zero for comp in ac_components} if ac_components else {}
        return zero, per_component

    # Filter to valid samples
    valid_pred = ac_pred[ac_valid]  # [N_valid, num_components]

    # Build target tensor dynamically based on enabled components
    target_parts = []
    for comp in ac_components:
        if comp == 'ugbw':
            target_parts.append(torch.log10(ac_ugbw[ac_valid].clamp(min=1.0).to(device)))
        elif comp == 'pm':
            target_parts.append(ac_pm[ac_valid].to(device))
        elif comp == 'am':
            target_parts.append(ac_am[ac_valid].to(device))
    target = torch.stack(target_parts, dim=-1)  # [N_valid, num_components]

    # Z-score normalize targets
    ac_mean = ac_mean.to(device)
    ac_std = ac_std.to(device)
    target_norm = (target - ac_mean) / ac_std.clamp(min=1e-6)

    # Per-component losses
    per_component = {}
    for i, comp in enumerate(ac_components):
        per_component[comp] = F.mse_loss(valid_pred[:, i], target_norm[:, i])

    return F.mse_loss(valid_pred, target_norm), per_component


# Region Classification Loss
def compute_region_loss(
    region_pred: torch.Tensor,
    mosfet_drain_mask: torch.Tensor,
    node_region_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Ordinal regression loss for MOSFET operating region prediction.

    Uses weighted MSE with continuous targets (cutoff=0, triode=1, saturation=2).
    Smooth gradients respect the physical ordering of regions.

    Args:
        region_pred: [num_nodes] scalar prediction per node
        mosfet_drain_mask: [num_nodes] bool - True at valid drain terminals
        node_region_labels: [num_nodes] long - region label at drain nodes, -1 elsewhere

    Returns:
        Scalar region loss
    """
    device = region_pred.device
    mask = mosfet_drain_mask.to(device) & (node_region_labels.to(device) >= 0)

    if not mask.any():
        return torch.tensor(0.0, device=device)

    pred = region_pred[mask]                        # [num_valid]
    target = node_region_labels[mask].float().to(device)  # 0.0, 1.0, 2.0

    return F.mse_loss(pred, target)


def compute_region_loss_coral(
    region_logits: torch.Tensor,
    mosfet_region_labels: torch.Tensor,
) -> torch.Tensor:
    """
    CORAL ordinal cross-entropy loss for MOSFET operating region prediction.

    Uses two binary classifiers for ordinal thresholds:
      - t1: P(region >= triode) = sigma(logit_1)
      - t2: P(region >= saturation) = sigma(logit_2)

    Args:
        region_logits: [num_mosfets, 2] raw logits for ordinal thresholds
        mosfet_region_labels: [num_mosfets] 0=cutoff, 1=triode, 2=saturation

    Returns:
        Scalar CORAL loss (mean BCE over both thresholds)
    """
    device = region_logits.device
    labels = mosfet_region_labels.to(device).long()

    # Binary targets for each threshold
    # t1: 1 if region >= 1 (triode or saturation), 0 if cutoff
    # t2: 1 if region >= 2 (saturation), 0 if cutoff or triode
    t1_target = (labels >= 1).float()
    t2_target = (labels >= 2).float()

    # BCE on each threshold
    loss_t1 = F.binary_cross_entropy_with_logits(region_logits[:, 0], t1_target)
    loss_t2 = F.binary_cross_entropy_with_logits(region_logits[:, 1], t2_target)

    return (loss_t1 + loss_t2) / 2


def _batch_offsets(num_devices, device_ptr, ptr, device):
    """Compute per-device node offsets for batched graphs."""
    if device_ptr is not None:
        graph_idx = torch.bucketize(
            torch.arange(num_devices, device=device),
            device_ptr[1:].to(device), right=True)
    else:
        num_graphs = len(ptr) - 1
        devices_per_graph = num_devices // num_graphs
        graph_idx = torch.arange(num_devices, device=device) // devices_per_graph
    return ptr[graph_idx]


def compute_device_consistency_loss(
    node_currents: torch.Tensor,
    batch,
) -> torch.Tensor:
    """MSE between terminal predictions of the same device.

    Penalizes drain != source for MOSFETs, p != n for 2-terminal devices.
    Operates in normalized z-score space (same as model output).
    """
    total = torch.tensor(0.0, device=node_currents.device)
    count = 0
    ptr = getattr(batch, 'ptr', None)
    is_batched = ptr is not None and len(ptr) > 2

    # MOSFETs: drain vs source
    if hasattr(batch, 'mosfet_info') and batch.mosfet_info is not None and batch.mosfet_info.numel() > 0:
        d = batch.mosfet_info[:, 1].long()
        s = batch.mosfet_info[:, 2].long()
        if is_batched:
            offsets = _batch_offsets(len(d), getattr(batch, 'mosfet_ptr', None), ptr, d.device)
            d = d + offsets
            s = s + offsets
        total = total + (node_currents[d] - node_currents[s]).pow(2).sum()
        count += d.shape[0]

    # 2-terminal devices: p vs n
    for attr, ptr_attr in [('resistor_info', 'resistor_ptr'), ('capacitor_info', 'capacitor_ptr'),
                           ('vsource_info', None), ('isource_info', 'isource_ptr')]:
        info = getattr(batch, attr, None)
        if info is not None and info.numel() > 0:
            p_idx = info[:, 0].long()
            n_idx = info[:, 1].long()
            if is_batched:
                dev_ptr = getattr(batch, ptr_attr, None) if ptr_attr else None
                offsets = _batch_offsets(len(p_idx), dev_ptr, ptr, p_idx.device)
                p_idx = p_idx + offsets
                n_idx = n_idx + offsets
            total = total + (node_currents[p_idx] - node_currents[n_idx]).pow(2).sum()
            count += p_idx.shape[0]

    return total / max(count, 1)


class UncertaintyWeights(nn.Module):
    """Kendall et al. 2018 with normalized weights — prevents magnitude explosion.

    Weights are normalized to sum to N_tasks, preserving relative adaptation
    while keeping total gradient magnitude constant.
    """
    def __init__(self, task_names):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def get_weights(self):
        """Normalized weights that sum to N_tasks."""
        raw = {name: torch.exp(-lv) for name, lv in self.log_vars.items()}
        total = sum(raw.values())
        n = len(raw)
        return {name: n * w / total for name, w in raw.items()}

    def regularizer(self):
        """Sum of log_var / 2 — prevents all weights from collapsing to equal."""
        return sum(lv for lv in self.log_vars.values()) / 2


# Combined Loss
def compute_combined_loss(
    voltage_pred: torch.Tensor,
    voltage_target: torch.Tensor,
    current_pred: torch.Tensor = None,
    current_pred_raw: torch.Tensor = None,
    current_target: torch.Tensor = None,
    current_mask: torch.Tensor = None,
    current_weight: float = 1.0,
    voltage_weight: float = 1.0,
    loss_type: str = 'mse',
    huber_delta: float = 1.0,
    kcl_weight: float = 0.0,
    edge_index: torch.Tensor = None,
    num_terminals: torch.Tensor = None,
    train_mask: torch.Tensor = None,
    ptr: torch.Tensor = None,
    terminal_current_sign: torch.Tensor = None,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    kcl_include_mask: torch.Tensor = None,
    kcl_min_current: float = 1e-9,
    kcl_mode: str = 'logsumexp',
    kcl_violation_threshold: float = 0.0,
    kcl_huber_delta: float = 0.0,
    kcl_gt_filter: float = 0.1,
    kcl_exclusive: bool = False,
    kcl_conservation: bool = False,
    kcl_detach_backbone: bool = False,
    kcl_skip_two_term: bool = False,
    kcl_only_two_term: bool = False,
    node_embeddings: torch.Tensor = None,
    current_head: torch.nn.Module = None,
    # Physics constraint parameters
    constraint_weight: float = 0.0,
    mosfet_info: torch.Tensor = None,
    terminal_features: torch.Tensor = None,
    diff_pair_constraints: torch.Tensor = None,
    mirror_constraints: torch.Tensor = None,
    output_stage_constraints: torch.Tensor = None,
    lambda_mirror_constraints: torch.Tensor = None,
    # Lambda mirror parameters
    node_names: list = None,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    lambda_n: float = 0.05,
    ref_vds_node: str = 'nbias',
    mir_vds_node: str = 'vout',
    # Full voltage predictions (all nodes, not masked) for lambda mirror
    full_voltage_pred: torch.Tensor = None,
    # Voltage node weights for stage2 upweighting
    voltage_node_weights: torch.Tensor = None,
    # gm physics self-consistency loss parameters
    gm_physics_loss_weight: float = 0.0,
    gm_physics_min_vov: float = 0.0,
    gm_physics_use_clm: bool = False,
    node_mosfet_vth: torch.Tensor = None,
    mosfet_region_labels: torch.Tensor = None,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    ss_gds_mean: float = 0.0,
    ss_gds_std: float = 1.0,
    # AC prediction loss parameters
    ac_loss_weight: float = 0.0,
    ac_pred: torch.Tensor = None,
    ac_ugbw: torch.Tensor = None,
    ac_pm: torch.Tensor = None,
    ac_am: torch.Tensor = None,
    ac_valid: torch.Tensor = None,
    ac_mean: torch.Tensor = None,
    ac_std: torch.Tensor = None,
    ac_components: list = None,
    # Supervised gm/gds prediction loss parameters (per-MOSFET)
    ss_gm_loss_weight: float = 0.0,
    ss_gds_loss_weight: float = 0.0,
    ss_gm_pred: torch.Tensor = None,
    ss_gds_pred: torch.Tensor = None,
    mosfet_gm: torch.Tensor = None,
    mosfet_gds: torch.Tensor = None,
    ss_huber_delta: float = 0.0,
    ss_region_stats: dict = None,
    # Triode physics regularizer loss parameters
    triode_physics_loss_weight: float = 0.0,
    triode_physics_config: dict = None,
    # Cutoff/subthreshold physics loss parameters
    cutoff_physics_loss_weight: float = 0.0,
    cutoff_physics_n_nmos: float = 1.5,
    cutoff_physics_n_pmos: float = 2.0,
    # Ground truth voltages for physics loss Vov computation
    node_voltage_targets: torch.Tensor = None,
    # Ground truth Vov for physics loss filtering
    mosfet_gt_vov: torch.Tensor = None,
    # Device ptr tensors for variable-topology batches
    mosfet_ptr: torch.Tensor = None,
    # Region loss parameters
    region_loss_weight: float = 0.0,
    region_pred: torch.Tensor = None,
    node_region_labels: torch.Tensor = None,
    mosfet_drain_mask: torch.Tensor = None,
    region_logits: torch.Tensor = None,
    # Device consistency loss
    device_consistency_weight: float = 0.0,
    batch=None,
    # Intermediate KCL (deep supervision)
    kcl_intermediate_weight: float = 0.0,
    aux_node_currents: torch.Tensor = None,
    # Vov prediction loss
    vov_loss_weight: float = 0.0,
    vov_pred: torch.Tensor = None,
    mosfet_vov_valid: torch.Tensor = None,
    # Vth prediction loss
    vth_loss_weight: float = 0.0,
    vth_pred: torch.Tensor = None,
    mosfet_vth: torch.Tensor = None,
    # IV model ID prediction loss
    iv_id_loss_weight: float = 0.0,
    iv_log_abs_id: torch.Tensor = None,
    # DC gain prediction loss
    dc_gain_loss_weight: float = 0.0,
    dc_gain_pred: torch.Tensor = None,
    dc_gain_target: torch.Tensor = None,
    dc_gain_mean: float = 0.0,
    dc_gain_std: float = 1.0,
    # Uncertainty weighting
    uncertainty_weights = None,
) -> tuple:
    """
    Compute combined loss from all components.

    Total loss = voltage_loss + current_weight * current_loss + kcl_weight * kcl_loss
                 + constraint_weight * constraint_loss

    Args:
        voltage_pred: Predicted voltage values
        voltage_target: Target voltage values
        current_pred: Predicted current values (optional)
        current_target: Target current values (optional)
        current_mask: Mask for nodes with current targets (optional)
        current_weight: Weight for current loss term
        loss_type: Loss function name ('mse', 'huber', 'mae')
        huber_delta: Delta parameter for Huber loss
        kcl_weight: Weight for KCL physics loss (0 to disable)
        edge_index: Graph connectivity [2, num_edges] (for KCL)
        num_terminals: Number of terminals per graph [batch_size] (for KCL)
        train_mask: Mask for internal nodes (for KCL)
        ptr: Node boundaries between graphs [batch_size + 1] (for KCL)
        terminal_current_sign: Sign for each terminal (+1 drain, -1 source) (for KCL)
        current_mean: Mean for current denormalization (log10 scale, for KCL)
        current_std: Std for current denormalization (log10 scale, for KCL)
        kcl_include_mask: Mask for terminals to include in KCL (False for gate/bulk)
        kcl_min_current: Minimum total current for a net to be included in KCL
        constraint_weight: Weight for physics constraint losses (0 to disable)
        mosfet_info: MOSFET info tensor [num_mosfets, 7] (for constraints)
        terminal_features: Node features [num_nodes, num_features] (for constraints)
        diff_pair_constraints: Diff pair constraint indices [N, 3]
        mirror_constraints: Mirror constraint indices [M, 2]
        output_stage_constraints: Output stage constraint indices [K, 2]
        lambda_mirror_constraints: Lambda-corrected mirror indices [L, 2]
        node_names: Node names per graph (for lambda mirror)
        vdc_mean: Mean for voltage denormalization
        vdc_std: Std for voltage denormalization
        lambda_n: Channel length modulation parameter
        ref_vds_node: Node name for reference Vds (lambda mirror)
        mir_vds_node: Node name for mirror Vds (lambda mirror)
        full_voltage_pred: Full voltage predictions (all nodes, not masked) for lambda mirror

    Returns:
        Tuple of (total_loss, voltage_loss, current_loss, kcl_loss, diff_pair_loss,
                  mirror_loss, output_stage_loss, lambda_mirror_loss, gm_physics_loss,
                  ac_loss, ss_loss, triode_physics_loss, triode_eq1/2/3_loss)
    """
    voltage_loss = compute_voltage_loss(
        voltage_pred, voltage_target, loss_type, huber_delta,
        node_weights=voltage_node_weights
    )

    # Apply KCL conservation projection (structural enforcement, replaces soft KCL loss)
    if kcl_conservation and current_pred is not None and edge_index is not None and ptr is not None:
        current_pred = apply_kcl_conservation_projection(
            current_pred, edge_index, num_terminals, train_mask, ptr,
            terminal_current_sign, current_mean, current_std, kcl_include_mask,
        )

    # Compute KCL first to get kcl_supervised_mask (needed for kcl_exclusive current loss)
    kcl_loss = torch.tensor(0.0, device=voltage_loss.device)
    kcl_supervised_mask = None
    if not kcl_conservation and current_pred is not None and edge_index is not None and ptr is not None:
        # When kcl_detach_backbone=True, recompute currents from detached embeddings
        # so KCL gradients only update the current head, not the backbone
        # Use pre-projection (raw) currents if available, so KCL loss sees true violations
        kcl_current_pred = current_pred_raw if current_pred_raw is not None else current_pred
        if kcl_detach_backbone and node_embeddings is not None and current_head is not None:
            kcl_current_pred = current_head(node_embeddings.detach()).squeeze(-1)

        kcl_loss, kcl_supervised_mask = compute_kcl_loss(
            kcl_current_pred, edge_index, num_terminals, train_mask, ptr,
            terminal_current_sign=terminal_current_sign,
            current_mean=current_mean,
            current_std=current_std,
            gt_currents=current_target,
            kcl_include_mask=kcl_include_mask,
            kcl_min_current=kcl_min_current,
            kcl_mode=kcl_mode,
            kcl_violation_threshold=kcl_violation_threshold,
            kcl_skip_two_term=kcl_skip_two_term,
            kcl_only_two_term=kcl_only_two_term,
            kcl_huber_delta=kcl_huber_delta,
            kcl_gt_filter=kcl_gt_filter,
        )

    # Intermediate KCL (deep supervision on auxiliary current predictions)
    kcl_intermediate_loss = torch.tensor(0.0, device=voltage_loss.device)
    if kcl_intermediate_weight > 0 and aux_node_currents is not None and edge_index is not None and ptr is not None:
        kcl_intermediate_loss, _ = compute_kcl_loss(
            aux_node_currents, edge_index, num_terminals, train_mask, ptr,
            terminal_current_sign=terminal_current_sign,
            current_mean=current_mean,
            current_std=current_std,
            gt_currents=current_target,
            kcl_include_mask=kcl_include_mask,
            kcl_min_current=kcl_min_current,
            kcl_mode=kcl_mode,
            kcl_violation_threshold=kcl_violation_threshold,
            kcl_skip_two_term=kcl_skip_two_term,
            kcl_only_two_term=kcl_only_two_term,
            kcl_huber_delta=kcl_huber_delta,
            kcl_gt_filter=kcl_gt_filter,
        )

    # When kcl_exclusive=True, exclude KCL-supervised terminals from current MSE
    effective_current_mask = current_mask
    if kcl_exclusive and kcl_supervised_mask is not None and current_mask is not None:
        effective_current_mask = current_mask & ~kcl_supervised_mask

    current_loss = compute_current_loss(
        current_pred, current_target, effective_current_mask, loss_type, huber_delta
    )
    if current_loss.device != voltage_loss.device:
        current_loss = current_loss.to(voltage_loss.device)

    # Physics-based current constraints
    diff_pair_loss = torch.tensor(0.0, device=voltage_loss.device)
    mirror_loss = torch.tensor(0.0, device=voltage_loss.device)
    output_stage_loss = torch.tensor(0.0, device=voltage_loss.device)
    lambda_mirror_loss = torch.tensor(0.0, device=voltage_loss.device)

    if constraint_weight > 0 and current_pred is not None and mosfet_info is not None and ptr is not None:
        # Note: For lambda_mirror, we need full voltage predictions, not masked ones
        # voltage_pred here is masked (only internal nodes). We pass current_pred which is
        # full (all terminals), so lambda_mirror uses current_pred's indexing for voltages.
        # Actually, we need to pass full_voltage_pred separately - see loops.py
        _, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss = compute_current_constraint_loss(
            pred_currents=current_pred,
            mosfet_info=mosfet_info,
            terminal_features=terminal_features,
            diff_pair_constraints=diff_pair_constraints,
            mirror_constraints=mirror_constraints,
            output_stage_constraints=output_stage_constraints,
            lambda_mirror_constraints=lambda_mirror_constraints,
            ptr=ptr,
            current_mean=current_mean,
            current_std=current_std,
            pred_voltages=full_voltage_pred,
            node_names=node_names,
            vdc_mean=vdc_mean,
            vdc_std=vdc_std,
            lambda_n=lambda_n,
            ref_vds_node=ref_vds_node,
            mir_vds_node=mir_vds_node,
        )

    constraint_loss = diff_pair_loss + mirror_loss + output_stage_loss + lambda_mirror_loss

    # gm physics self-consistency loss
    gm_physics_loss = torch.tensor(0.0, device=voltage_loss.device)
    if gm_physics_loss_weight > 0 and ss_gm_pred is not None and current_pred is not None and full_voltage_pred is not None:
        gm_physics_loss = compute_gm_physics_loss(
            ss_gm_pred=ss_gm_pred,
            pred_currents=current_pred,
            full_voltage_pred=full_voltage_pred,
            mosfet_info=mosfet_info,
            node_mosfet_vth=node_mosfet_vth,
            mosfet_region_labels=mosfet_region_labels,
            ptr=ptr,
            vdc_mean=vdc_mean,
            vdc_std=vdc_std,
            current_mean=current_mean,
            current_std=current_std,
            ss_gm_mean=ss_gm_mean,
            ss_gm_std=ss_gm_std,
            min_vov=gm_physics_min_vov,
            mosfet_gt_vov=mosfet_gt_vov,
            mosfet_ptr=mosfet_ptr,
            ss_gds_pred=ss_gds_pred,
            ss_gds_mean=ss_gds_mean,
            ss_gds_std=ss_gds_std,
            use_clm=gm_physics_use_clm,
            node_voltage_targets=node_voltage_targets,
        )

    # AC prediction loss
    ac_loss = torch.tensor(0.0, device=voltage_loss.device)
    ac_per_component = {}
    if ac_loss_weight > 0 and ac_pred is not None and ac_mean is not None and ac_components:
        ac_loss, ac_per_component = compute_ac_loss(
            ac_pred=ac_pred,
            ac_valid=ac_valid,
            ac_mean=ac_mean,
            ac_std=ac_std,
            ac_components=ac_components,
            ac_ugbw=ac_ugbw,
            ac_pm=ac_pm,
            ac_am=ac_am,
        )

    # Supervised gm/gds prediction loss (per-MOSFET)
    ss_gm_loss = torch.tensor(0.0, device=voltage_loss.device)
    ss_gds_loss = torch.tensor(0.0, device=voltage_loss.device)
    if (ss_gm_loss_weight > 0 or ss_gds_loss_weight > 0) and ss_gm_pred is not None and ss_gds_pred is not None and mosfet_gm is not None:
        ss_gm_loss, ss_gds_loss = compute_ss_loss(
            gm_pred=ss_gm_pred,
            gds_pred=ss_gds_pred,
            mosfet_gm=mosfet_gm,
            mosfet_gds=mosfet_gds,
            gm_mean=ss_gm_mean,
            gm_std=ss_gm_std,
            gds_mean=ss_gds_mean,
            gds_std=ss_gds_std,
            huber_delta=ss_huber_delta,
            region_stats=ss_region_stats,
            mosfet_region_labels=mosfet_region_labels,
        )

    # Triode physics regularizer loss
    triode_physics_loss = torch.tensor(0.0, device=voltage_loss.device)
    triode_eq1_loss = torch.tensor(0.0, device=voltage_loss.device)
    triode_eq2_loss = torch.tensor(0.0, device=voltage_loss.device)
    triode_eq3_loss = torch.tensor(0.0, device=voltage_loss.device)
    if triode_physics_loss_weight > 0 and ss_gm_pred is not None and ss_gds_pred is not None and full_voltage_pred is not None:
        tri_cfg = triode_physics_config or {}
        triode_physics_loss, triode_eq1_loss, triode_eq2_loss, triode_eq3_loss = compute_triode_physics_loss(
            ss_gm_pred=ss_gm_pred,
            ss_gds_pred=ss_gds_pred,
            pred_currents=current_pred,
            full_voltage_pred=full_voltage_pred,
            mosfet_info=mosfet_info,
            node_mosfet_vth=node_mosfet_vth,
            mosfet_region_labels=mosfet_region_labels,
            ptr=ptr,
            vdc_mean=vdc_mean,
            vdc_std=vdc_std,
            current_mean=current_mean,
            current_std=current_std,
            ss_gm_mean=ss_gm_mean,
            ss_gm_std=ss_gm_std,
            ss_gds_mean=ss_gds_mean,
            ss_gds_std=ss_gds_std,
            min_vov=tri_cfg.get('min_vov', 0.0),
            eq1_enabled=tri_cfg.get('eq1_enabled', True),
            eq1_weight=tri_cfg.get('eq1_weight', 1.0),
            eq2_enabled=tri_cfg.get('eq2_enabled', True),
            eq2_weight=tri_cfg.get('eq2_weight', 1.0),
            eq2_max_vds_vov=tri_cfg.get('eq2_max_vds_vov', 1.0),
            eq3_enabled=tri_cfg.get('eq3_enabled', True),
            eq3_weight=tri_cfg.get('eq3_weight', 1.0),
            mosfet_gt_vov=mosfet_gt_vov,
            mosfet_ptr=mosfet_ptr,
            node_voltage_targets=node_voltage_targets,
        )

    # Cutoff/subthreshold physics loss
    cutoff_physics_loss = torch.tensor(0.0, device=voltage_loss.device)
    if cutoff_physics_loss_weight > 0 and ss_gm_pred is not None and current_pred is not None:
        cutoff_physics_loss = compute_cutoff_physics_loss(
            ss_gm_pred=ss_gm_pred,
            pred_currents=current_pred,
            mosfet_info=mosfet_info,
            mosfet_region_labels=mosfet_region_labels,
            ptr=ptr,
            current_mean=current_mean,
            current_std=current_std,
            ss_gm_mean=ss_gm_mean,
            ss_gm_std=ss_gm_std,
            n_nmos=cutoff_physics_n_nmos,
            n_pmos=cutoff_physics_n_pmos,
            mosfet_ptr=mosfet_ptr,
        )

    # Region loss (CORAL ordinal cross-entropy or MSE fallback)
    region_loss = torch.tensor(0.0, device=voltage_loss.device)
    if region_loss_weight > 0:
        if region_logits is not None and mosfet_region_labels is not None:
            # CORAL: per-MOSFET ordinal logits
            region_loss = compute_region_loss_coral(
                region_logits=region_logits,
                mosfet_region_labels=mosfet_region_labels,
            )
        elif region_pred is not None and node_region_labels is not None and mosfet_drain_mask is not None:
            # Legacy: per-node MSE
            region_loss = compute_region_loss(
                region_pred=region_pred,
                mosfet_drain_mask=mosfet_drain_mask,
                node_region_labels=node_region_labels,
            )

    # Device consistency loss
    if device_consistency_weight > 0 and current_pred is not None and batch is not None:
        dev_consistency_loss = compute_device_consistency_loss(current_pred, batch)
    else:
        dev_consistency_loss = torch.tensor(0.0, device=voltage_pred.device)

    # Vov prediction loss
    vov_loss = torch.tensor(0.0, device=voltage_pred.device)
    if vov_loss_weight > 0 and vov_pred is not None and mosfet_gt_vov is not None:
        vov_loss = compute_vov_loss(vov_pred, mosfet_gt_vov, mosfet_vov_valid)

    # Vth prediction loss
    vth_loss = torch.tensor(0.0, device=voltage_pred.device)
    if vth_loss_weight > 0 and vth_pred is not None and mosfet_vth is not None:
        vth_loss = compute_vth_loss(vth_pred, mosfet_vth.to(voltage_pred.device))

    # IV model ID prediction loss
    iv_id_loss = torch.tensor(0.0, device=voltage_pred.device)
    if iv_id_loss_weight > 0 and iv_log_abs_id is not None and current_target is not None:
        iv_id_loss = compute_iv_id_loss(
            log_abs_id_pred=iv_log_abs_id,
            mosfet_info=mosfet_info,
            node_current_targets=current_target,
            current_mean=current_mean,
            current_std=current_std,
            ptr=ptr,
            mosfet_ptr=mosfet_ptr,
        )

    # DC gain prediction loss (z-scored MSE, all samples)
    dc_gain_loss = torch.tensor(0.0, device=voltage_pred.device)
    if dc_gain_loss_weight > 0 and dc_gain_pred is not None and dc_gain_target is not None:
        dc_gain_target_z = (dc_gain_target - dc_gain_mean) / max(dc_gain_std, 1e-6)
        dc_gain_loss = F.mse_loss(dc_gain_pred, dc_gain_target_z)

    if uncertainty_weights is not None:
        # Normalized uncertainty weighting (Kendall et al. 2018, weights sum to N_tasks)
        uw = uncertainty_weights.get_weights()
        total_loss = torch.tensor(0.0, device=voltage_pred.device)
        if voltage_weight > 0:
            total_loss = total_loss + uw['voltage'] * voltage_loss
        if current_weight > 0:
            total_loss = total_loss + uw['current'] * current_loss
        if ss_gm_loss_weight > 0:
            total_loss = total_loss + uw['ss_gm'] * ss_gm_loss
        if ss_gds_loss_weight > 0:
            total_loss = total_loss + uw['ss_gds'] * ss_gds_loss
        if vov_loss_weight > 0:
            total_loss = total_loss + uw['vov'] * vov_loss
        # Log_var regularizer (prevents all weights from becoming equal)
        total_loss = total_loss + uncertainty_weights.regularizer()
        # Regularizers: keep static weights
        total_loss = total_loss + (kcl_weight * kcl_loss +
                                   constraint_weight * constraint_loss +
                                   gm_physics_loss_weight * gm_physics_loss +
                                   ac_loss_weight * ac_loss +
                                   triode_physics_loss_weight * triode_physics_loss +
                                   cutoff_physics_loss_weight * cutoff_physics_loss +
                                   region_loss_weight * region_loss +
                                   device_consistency_weight * dev_consistency_loss +
                                   kcl_intermediate_weight * kcl_intermediate_loss +
                                   iv_id_loss_weight * iv_id_loss +
                                   dc_gain_loss_weight * dc_gain_loss)
    else:
        total_loss = (voltage_weight * voltage_loss +
                      current_weight * current_loss +
                      kcl_weight * kcl_loss +
                      constraint_weight * constraint_loss +
                      gm_physics_loss_weight * gm_physics_loss +
                      ac_loss_weight * ac_loss +
                      ss_gm_loss_weight * ss_gm_loss +
                      ss_gds_loss_weight * ss_gds_loss +
                      triode_physics_loss_weight * triode_physics_loss +
                      cutoff_physics_loss_weight * cutoff_physics_loss +
                      region_loss_weight * region_loss +
                      device_consistency_weight * dev_consistency_loss +
                      kcl_intermediate_weight * kcl_intermediate_loss +
                      vov_loss_weight * vov_loss +
                      vth_loss_weight * vth_loss +
                      iv_id_loss_weight * iv_id_loss +
                      dc_gain_loss_weight * dc_gain_loss)

    return total_loss, voltage_loss, current_loss, kcl_loss, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss, gm_physics_loss, ac_loss, ss_gm_loss, ss_gds_loss, triode_physics_loss, triode_eq1_loss, triode_eq2_loss, triode_eq3_loss, region_loss, cutoff_physics_loss, kcl_intermediate_loss, vov_loss, vth_loss, iv_id_loss, ac_per_component, dc_gain_loss
