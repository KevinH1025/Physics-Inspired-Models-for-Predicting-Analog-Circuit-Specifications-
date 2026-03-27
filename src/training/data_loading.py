"""
Data loading utilities for training.

This module provides utilities for loading pre-batched datasets
and iterating over batches during training.
"""

import pickle
import random
from pathlib import Path

import torch
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .losses import get_device_graph_idx


class PrebatchedLoader:
    """Iterator that shuffles and yields pre-batched graphs each epoch."""

    def __init__(self, batches: List, shuffle: bool = True):
        self.batches = batches
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = list(range(len(self.batches)))
            random.shuffle(indices)
            for i in indices:
                yield self.batches[i]
        else:
            for batch in self.batches:
                yield batch

    def __len__(self):
        return len(self.batches)


class CircuitGraphDataset(Dataset):
    """Dataset for loading individual circuit graphs (non-prebatched mode)."""

    def __init__(self, samples_file: Path):
        samples_file = Path(samples_file)
        with open(samples_file, 'rb') as f:
            data = pickle.load(f)
        # Handle both formats: list of dicts or list of Data objects
        if len(data) > 0 and isinstance(data[0], dict):
            self.graphs = [s['graph'] for s in data]
        else:
            self.graphs = data

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def _sort_edge_index(batch):
    """Sort edge_index by destination node for deterministic scatter operations."""
    if hasattr(batch, 'edge_index') and batch.edge_index.numel() > 0:
        edge_index = batch.edge_index
        # Sort by destination (row 1), then source (row 0) for full determinism
        idx = torch.argsort(edge_index[1] * edge_index.max() + edge_index[0])
        batch.edge_index = edge_index[:, idx]
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr[idx]
    return batch


def load_prebatched_variant(split_dir, variant_id: int = 0, device=None) -> List:
    """Load a specific batching variant from a split directory."""
    split_dir = Path(split_dir)
    variant_path = split_dir / f'variant_{variant_id}.pkl'
    with open(variant_path, 'rb') as f:
        batches = pickle.load(f)
    # Sort edge_index for deterministic scatter operations
    batches = [_sort_edge_index(batch) for batch in batches]
    if device:
        batches = [batch.to(device) for batch in batches]
    return batches


def load_prebatched_metadata(split_dir) -> Dict[str, Any]:
    """Load metadata for a pre-batched split directory."""
    split_dir = Path(split_dir)
    metadata_path = split_dir / 'metadata.pkl'
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)


def compute_vdc_normalization(
    dataset_path: Path,
    train_batches: List,
    target_norm_type: str = 'zscore',
    vdd: float = 1.8
) -> tuple:
    """
    Compute VDC normalization statistics from training data.

    Args:
        dataset_path: Path to dataset directory
        train_batches: List of training batches (fallback if raw data not available)
        target_norm_type: 'zscore' or 'minmax'
        vdd: VDD voltage for minmax normalization

    Returns:
        Tuple of (vdc_mean, vdc_std)
    """
    train_dataset_path = dataset_path / 'dataset_train.pkl'

    if train_dataset_path.exists():
        with open(train_dataset_path, 'rb') as f:
            train_raw = pickle.load(f)
        train_vdc_values = []
        for item in train_raw:
            graph = item['graph']
            if hasattr(graph, 'vdc') and graph.vdc is not None:
                train_vdc_values.extend(graph.vdc.flatten().tolist())
        train_vdc_values = np.array(train_vdc_values)
        del train_raw

        if target_norm_type == 'minmax':
            vdc_mean = 0.0
            vdc_std = vdd
        else:
            vdc_mean = float(train_vdc_values.mean())
            vdc_std = float(train_vdc_values.std())
            if vdc_std == 0:
                vdc_std = 1.0
    else:
        sample = train_batches[0]
        if target_norm_type == 'minmax':
            vdc_mean = 0.0
            vdc_std = vdd
        else:
            vdc_mean = sample.vdc.mean().item()
            vdc_std = sample.vdc.std().item()
            if vdc_std == 0:
                vdc_std = 1.0

    return vdc_mean, vdc_std


def compute_current_normalization(train_variants: List[List]) -> tuple:
    """
    Compute current normalization statistics from training data.

    Uses log10 z-score normalization for currents.

    Args:
        train_variants: List of training batch variants

    Returns:
        Tuple of (current_mean, current_std) in log10 space
    """
    train_current_values = []
    for variant_batches in train_variants:
        for batch in variant_batches:
            if hasattr(batch, 'node_current_targets') and hasattr(batch, 'has_current_mask'):
                if batch.node_current_targets is not None and batch.has_current_mask is not None:
                    mask = batch.has_current_mask
                    if mask.any():
                        current = batch.node_current_targets[mask]
                        train_current_values.extend(current.abs().cpu().tolist())

    if train_current_values:
        current_arr = np.array(train_current_values)
        log_epsilon = 1e-12
        log_current_arr = np.log10(current_arr + log_epsilon)
        current_mean = float(log_current_arr.mean())
        current_std = float(log_current_arr.std())
        if current_std == 0:
            current_std = 1.0
    else:
        current_mean = 0.0
        current_std = 1.0

    return current_mean, current_std


def normalize_batches_vdc(batches: List, vdc_mean: float, vdc_std: float) -> None:
    """Normalize VDC targets in batches in-place."""
    for batch in batches:
        batch.vdc = (batch.vdc - vdc_mean) / vdc_std
        if hasattr(batch, 'terminal_vdc') and batch.terminal_vdc is not None:
            batch.terminal_vdc = (batch.terminal_vdc - vdc_mean) / vdc_std


def normalize_batches_current(batches: List, current_mean: float, current_std: float,
                              z_clip: float = 0.0) -> None:
    """Normalize current targets in batches in-place using log10 z-score.

    Args:
        batches: List of batches to normalize in-place
        current_mean: Log10 current mean
        current_std: Log10 current std
        z_clip: Soft clip threshold (0 to disable). Values beyond ±z_clip
                are smoothly compressed using tanh. Recommended: 4.0
    """
    log_epsilon = 1e-12
    for batch in batches:
        if hasattr(batch, 'node_current_targets') and batch.node_current_targets is not None:
            log_targets = torch.log10(batch.node_current_targets.abs() + log_epsilon)
            z_score = (log_targets - current_mean) / current_std

            # Soft clipping: compress values beyond threshold using tanh
            # This preserves ordering while preventing extreme outliers
            if z_clip > 0:
                z_score = torch.where(
                    z_score.abs() > z_clip,
                    z_clip * torch.tanh(z_score / z_clip),
                    z_score
                )

            batch.node_current_targets = z_score


def attach_normalization_stats(
    batches: List,
    voltage_mean: float,
    voltage_std: float,
    current_mean: float = 0.0,
    current_std: float = 1.0,
) -> None:
    """
    Attach normalization stats to batches for physics-based current prediction.

    These stats are needed by the voltage-derived current computation to properly
    scale resistor currents from predicted voltages.

    Args:
        batches: List of PyG batch objects
        voltage_mean: Mean for voltage denormalization
        voltage_std: Std for voltage denormalization
        current_mean: Mean for current normalization (log10 scale)
        current_std: Std for current normalization (log10 scale)
    """
    for batch in batches:
        batch.voltage_mean = voltage_mean
        batch.voltage_std = voltage_std
        batch.current_mean = current_mean
        batch.current_std = current_std


def add_ss_node_targets(batches: List) -> None:
    """
    Add per-node gm/gds targets and drain mask to pre-batched data.

    For datasets that store mosfet_gm/mosfet_gds as per-MOSFET tensors,
    this converts them to per-node tensors with a drain mask. Per-node
    tensors are correctly handled by PyG batching (simple concatenation)
    without needing manual index offsetting.

    Creates:
        mosfet_drain_mask: [num_nodes] bool - True at drain terminal nodes
        node_log_gm: [num_nodes] float - log10(gm) at drain nodes, 0 elsewhere
        node_log_gds: [num_nodes] float - log10(gds) at drain nodes, 0 elsewhere
    """
    import math
    for batch in batches:
        if hasattr(batch, 'mosfet_drain_mask'):
            continue  # Already has per-node targets

        mosfet_info = getattr(batch, 'mosfet_info', None)
        mosfet_gm = getattr(batch, 'mosfet_gm', None)
        mosfet_gds = getattr(batch, 'mosfet_gds', None)
        if mosfet_info is None or mosfet_gm is None or mosfet_gds is None:
            continue

        num_nodes = batch.x.shape[0]
        ptr = batch.ptr
        drain_idx = mosfet_info[:, 1].long()

        # Compute global drain indices (offset for batched graphs)
        num_mosfets = mosfet_info.shape[0]
        num_graphs = ptr.shape[0] - 1
        mosfet_ptr = getattr(batch, 'mosfet_ptr', None)
        mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, ptr.device)
        node_offsets = ptr[mosfet_graph_idx]
        drain_idx_global = drain_idx + node_offsets

        # Build per-node mask and targets (on CPU, then move to device)
        device = batch.x.device
        drain_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_log_gm = torch.zeros(num_nodes, dtype=torch.float, device=device)
        node_log_gds = torch.zeros(num_nodes, dtype=torch.float, device=device)

        valid = mosfet_gm > 1e-12
        valid_indices = drain_idx_global[valid]
        drain_mask[valid_indices] = True
        node_log_gm[valid_indices] = torch.log10(mosfet_gm[valid].to(device))
        gds_valid = mosfet_gds[valid].clamp(min=1e-20)
        node_log_gds[valid_indices] = torch.log10(gds_valid.to(device))

        batch.mosfet_drain_mask = drain_mask
        batch.node_log_gm = node_log_gm
        batch.node_log_gds = node_log_gds


def add_region_node_targets(batches: List) -> None:
    """
    Add per-node region labels to pre-batched data for region classification head.

    Converts per-MOSFET mosfet_region_labels to per-node node_region_labels.
    Requires mosfet_drain_mask to already exist (call add_ss_node_targets first).

    Creates:
        node_region_labels: [num_nodes] long - region label at drain nodes, -1 elsewhere
    """
    for batch in batches:
        if hasattr(batch, 'node_region_labels'):
            continue

        mosfet_info = getattr(batch, 'mosfet_info', None)
        region_labels = getattr(batch, 'mosfet_region_labels', None)
        if mosfet_info is None or region_labels is None:
            continue

        num_nodes = batch.x.shape[0]
        ptr = batch.ptr
        drain_idx = mosfet_info[:, 1].long()

        num_mosfets = mosfet_info.shape[0]
        num_graphs = ptr.shape[0] - 1
        mosfet_ptr = getattr(batch, 'mosfet_ptr', None)
        mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, ptr.device)
        node_offsets = ptr[mosfet_graph_idx]
        drain_idx_global = drain_idx + node_offsets

        device = batch.x.device
        node_region = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_region[drain_idx_global] = region_labels.to(device)

        batch.node_region_labels = node_region


def add_vth_node_targets(batches: List) -> None:
    """
    Add per-node Vth targets to pre-batched data for gm physics loss.

    Converts per-MOSFET mosfet_vth to per-node node_mosfet_vth.
    If mosfet_vth is missing (old datasets), creates zeros (loss will skip).

    Creates:
        node_mosfet_vth: [num_nodes] float - Vth at drain nodes, 0 elsewhere
    """
    for batch in batches:
        if hasattr(batch, 'node_mosfet_vth'):
            continue

        mosfet_info = getattr(batch, 'mosfet_info', None)
        mosfet_vth = getattr(batch, 'mosfet_vth', None)
        if mosfet_info is None or mosfet_vth is None:
            continue

        num_nodes = batch.x.shape[0]
        ptr = batch.ptr
        drain_idx = mosfet_info[:, 1].long()

        num_mosfets = mosfet_info.shape[0]
        num_graphs = ptr.shape[0] - 1
        mosfet_ptr = getattr(batch, 'mosfet_ptr', None)
        mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, ptr.device)
        node_offsets = ptr[mosfet_graph_idx]
        drain_idx_global = drain_idx + node_offsets

        device = batch.x.device
        node_vth = torch.zeros(num_nodes, dtype=torch.float, device=device)
        node_vth[drain_idx_global] = mosfet_vth.to(device)

        batch.node_mosfet_vth = node_vth


def add_mosfet_gt_vov(batches: List) -> None:
    """
    Pre-compute ground truth Vov per MOSFET from SPICE gm, gds, I_D.

    Must be called BEFORE current normalization (needs raw node_current_targets).

    For saturation: Vov = 2 * I_D / gm
    For triode: Vds = I_D / gm, Vov = Vds * (1 + gds/gm)
    For cutoff: Vov = 0

    Creates:
        mosfet_gt_vov: [num_mosfets] float - ground truth |Vov| per MOSFET
    """
    for batch in batches:
        if hasattr(batch, 'mosfet_gt_vov'):
            continue

        mosfet_info = getattr(batch, 'mosfet_info', None)
        mosfet_gm = getattr(batch, 'mosfet_gm', None)
        mosfet_gds = getattr(batch, 'mosfet_gds', None)
        nct = getattr(batch, 'node_current_targets', None)
        region_labels = getattr(batch, 'mosfet_region_labels', None)
        if mosfet_info is None or mosfet_gm is None or nct is None:
            continue

        ptr = batch.ptr
        num_mosfets = mosfet_info.shape[0]
        num_graphs = ptr.shape[0] - 1
        mosfet_ptr = getattr(batch, 'mosfet_ptr', None)
        mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, ptr.device)
        node_offsets = ptr[mosfet_graph_idx]
        drain_idx_global = mosfet_info[:, 1].long() + node_offsets

        device = batch.x.device
        gm = mosfet_gm.to(device).clamp(min=1e-15)
        I_D = nct[drain_idx_global].abs().clamp(min=1e-15)

        # Default: saturation formula Vov = 2*I_D/gm
        gt_vov = (2.0 * I_D / gm)

        # Triode: Vds = I_D/gm, Vov = Vds * (1 + gds/gm)
        if mosfet_gds is not None and region_labels is not None:
            gds = mosfet_gds.to(device).clamp(min=1e-15)
            triode = (region_labels.to(device) == 1)
            if triode.any():
                Vds_tri = I_D[triode] / gm[triode]
                gt_vov[triode] = Vds_tri * (1.0 + gds[triode] / gm[triode])

        # Cutoff devices: Vov = 0
        if region_labels is not None:
            cutoff = (region_labels.to(device) == 0)
            gt_vov[cutoff] = 0.0

        batch.mosfet_gt_vov = gt_vov.clamp(min=0.0, max=1.8)


def compute_ss_normalization(train_variants: List[List], per_region: bool = False) -> tuple:
    """
    Compute gm/gds normalization statistics from training data.

    Computes mean and std of log10(gm) and log10(gds) across all training
    MOSFETs for z-score normalization (same approach as current normalization).

    Args:
        train_variants: List of training batch variants (after add_ss_node_targets)
        per_region: If True, also compute per-region (cutoff/triode/sat) stats

    Returns:
        If per_region=False: Tuple of (gm_mean, gm_std, gds_mean, gds_std)
        If per_region=True: Tuple of (gm_mean, gm_std, gds_mean, gds_std, region_stats)
            where region_stats = {0: {'gm_mean', 'gm_std', 'gds_mean', 'gds_std'}, 1: ..., 2: ...}
    """
    all_log_gm = []
    all_log_gds = []
    per_region_gm = {0: [], 1: [], 2: []}
    per_region_gds = {0: [], 1: [], 2: []}

    for variant_batches in train_variants:
        for batch in variant_batches:
            if hasattr(batch, 'mosfet_drain_mask') and hasattr(batch, 'node_log_gm'):
                mask = batch.mosfet_drain_mask
                if mask.any():
                    all_log_gm.extend(batch.node_log_gm[mask].cpu().tolist())
                    all_log_gds.extend(batch.node_log_gds[mask].cpu().tolist())

            if per_region and hasattr(batch, 'mosfet_gm') and hasattr(batch, 'mosfet_region_labels'):
                gm_raw = batch.mosfet_gm
                gds_raw = batch.mosfet_gds
                labels = batch.mosfet_region_labels
                valid = gm_raw > 1e-12
                for r in range(3):
                    rmask = valid & (labels == r)
                    if rmask.any():
                        per_region_gm[r].extend(torch.log10(gm_raw[rmask]).cpu().tolist())
                        per_region_gds[r].extend(torch.log10(gds_raw[rmask].clamp(min=1e-20)).cpu().tolist())

    if all_log_gm:
        gm_arr = np.array(all_log_gm)
        gds_arr = np.array(all_log_gds)
        gm_mean = float(gm_arr.mean())
        gm_std = float(gm_arr.std())
        gds_mean = float(gds_arr.mean())
        gds_std = float(gds_arr.std())
        if gm_std == 0:
            gm_std = 1.0
        if gds_std == 0:
            gds_std = 1.0
    else:
        gm_mean, gm_std = 0.0, 1.0
        gds_mean, gds_std = 0.0, 1.0

    if per_region:
        region_stats = {}
        for r in range(3):
            if per_region_gm[r]:
                gm_arr_r = np.array(per_region_gm[r])
                gds_arr_r = np.array(per_region_gds[r])
                region_stats[r] = {
                    'gm_mean': float(gm_arr_r.mean()),
                    'gm_std': max(float(gm_arr_r.std()), 1e-6),
                    'gds_mean': float(gds_arr_r.mean()),
                    'gds_std': max(float(gds_arr_r.std()), 1e-6),
                }
            else:
                region_stats[r] = {'gm_mean': gm_mean, 'gm_std': gm_std, 'gds_mean': gds_mean, 'gds_std': gds_std}
        return gm_mean, gm_std, gds_mean, gds_std, region_stats

    return gm_mean, gm_std, gds_mean, gds_std


def normalize_batches_ss(batches: List, gm_mean: float, gm_std: float,
                         gds_mean: float, gds_std: float,
                         region_stats: dict = None) -> None:
    """Normalize SS targets (log10 gm/gds) in batches in-place using z-score.

    If region_stats is provided, uses per-region (mean, std) for each MOSFET
    based on its mosfet_region_labels. Falls back to global stats for unknown regions.
    """
    for batch in batches:
        if not (hasattr(batch, 'node_log_gm') and hasattr(batch, 'mosfet_drain_mask')):
            continue
        if region_stats is not None and hasattr(batch, 'mosfet_region_labels') and hasattr(batch, 'mosfet_info'):
            labels = batch.mosfet_region_labels
            gm_raw = batch.mosfet_gm
            valid = gm_raw > 1e-12
            drain_idxs = batch.mosfet_info[:, 1].long()
            for i in range(len(labels)):
                if not valid[i]:
                    continue
                r = labels[i].item()
                if 0 <= r <= 2:
                    rs = region_stats[r]
                else:
                    rs = {'gm_mean': gm_mean, 'gm_std': gm_std, 'gds_mean': gds_mean, 'gds_std': gds_std}
                didx = drain_idxs[i].item()
                batch.node_log_gm[didx] = (batch.node_log_gm[didx] - rs['gm_mean']) / rs['gm_std']
                batch.node_log_gds[didx] = (batch.node_log_gds[didx] - rs['gds_mean']) / rs['gds_std']
        else:
            batch.node_log_gm = (batch.node_log_gm - gm_mean) / gm_std
            batch.node_log_gds = (batch.node_log_gds - gds_mean) / gds_std


def compute_vov_normalization(train_variants: List[List]) -> tuple:
    """Compute Vov normalization statistics (log10 z-score) from training data."""
    all_log_vov = []
    for variant_batches in train_variants:
        for batch in variant_batches:
            if hasattr(batch, 'mosfet_gt_vov'):
                vov = batch.mosfet_gt_vov
                valid = vov > 0
                if valid.any():
                    all_log_vov.extend(torch.log10(vov[valid].clamp(min=1e-15)).cpu().tolist())

    if all_log_vov:
        arr = np.array(all_log_vov)
        vov_mean = float(arr.mean())
        vov_std = float(arr.std())
        if vov_std == 0:
            vov_std = 1.0
    else:
        vov_mean, vov_std = 0.0, 1.0

    return vov_mean, vov_std


def normalize_batches_vov(batches: List, vov_mean: float, vov_std: float) -> None:
    """Normalize Vov targets (log10 z-score) in batches in-place."""
    for batch in batches:
        if hasattr(batch, 'mosfet_gt_vov'):
            valid = batch.mosfet_gt_vov > 0
            batch.mosfet_vov_valid = valid
            log_vov = torch.zeros_like(batch.mosfet_gt_vov)
            log_vov[valid] = (torch.log10(batch.mosfet_gt_vov[valid].clamp(min=1e-15)) - vov_mean) / vov_std
            batch.mosfet_gt_vov = log_vov


def get_prediction_mask(batch):
    """
    Get the mask for prediction targets from a batch.

    Uses train_mask if available, otherwise falls back to output_node_mask.

    Args:
        batch: PyTorch Geometric batch with mask attributes

    Returns:
        Boolean mask tensor for prediction targets
    """
    if hasattr(batch, 'train_mask') and batch.train_mask is not None:
        return batch.train_mask
    return batch.output_node_mask
