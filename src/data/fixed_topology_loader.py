"""
Fast data loader for fixed-topology circuit datasets.

All samples in the same circuit topology (e.g. 3-stage opamp) share
identical graph structure: same nodes, same edges, same masks.
Only node features (x) and simulation targets vary per sample.

This loader exploits that by:
  1. Storing variable tensors as compact [N_samples, ...] GPU tensors
  2. Precomputing fixed topology components (edge_index, type_tens, etc.) once
  3. Assembling batches via fast tensor indexing — no PyG Batch.from_data_list()

Memory footprint for 16k samples (3-stage opamp):
  ~100MB total vs ~3GB for 10-variant prebatched approach
"""

import pickle
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch


class FixedTopologyDataset:
    """Compact GPU-resident dataset for fixed-topology circuits."""

    def __init__(self, samples: List[dict], device: str = 'cpu'):
        g0 = samples[0]['graph']
        self.device = device
        self.num_nodes = g0.x.shape[0]         # 134
        self.num_edges = g0.edge_index.shape[1] # 228
        self.num_mosfets = g0.mosfet_info.shape[0]  # 24

        # ── Fixed topology (store once) ──────────────────────────────────
        self.edge_index = g0.edge_index.to(device)              # [2, E]
        self.type_tens = g0.type_tens.to(device)                # [N, 12]
        self.output_node_mask = g0.output_node_mask.to(device)  # [N]
        self.train_mask = g0.train_mask.to(device)              # [N]
        self.known_voltage_mask = g0.known_voltage_mask.to(device)
        self.has_current_mask = g0.has_current_mask.to(device)  # [N]
        # Per-sample masks for normalization (has_current_mask varies across samples)
        self.all_has_current_mask = torch.stack(
            [s['graph'].has_current_mask for s in samples]
        ).to(device)  # [n, N]
        self.kcl_include_mask = g0.kcl_include_mask.to(device)
        self.mosfet_info = g0.mosfet_info.to(device)           # [M, 7]
        if hasattr(g0, 'mosfet_drain_mask') and g0.mosfet_drain_mask is not None:
            self.mosfet_drain_mask = g0.mosfet_drain_mask.to(device)
        else:
            self.mosfet_drain_mask = None
        self.terminal_train_mask = g0.terminal_train_mask.to(device)
        self.resistor_info = g0.resistor_info.to(device)        # [R, 5]
        self.capacitor_info = g0.capacitor_info.to(device)      # [C, 4]
        self.num_terminals = g0.num_terminals
        self.num_resistors = g0.resistor_info.shape[0]
        self.num_capacitors = g0.capacitor_info.shape[0]

        # ── Variable tensors (stacked across all samples) ─────────────────
        self.all_x = torch.stack([s['graph'].x for s in samples]).to(device)
        # [n, N, 9]

        vdc_list = [s['graph'].vdc.squeeze(-1) for s in samples]
        self.all_vdc = torch.stack(vdc_list).to(device)
        # [n, num_output_nodes]  (14 for 3-stage)

        self.all_node_voltage = torch.stack(
            [s['graph'].node_voltage_targets for s in samples]
        ).to(device)  # [n, N]

        tvdc_list = [s['graph'].terminal_vdc.squeeze(-1) for s in samples]
        self.all_terminal_vdc = torch.stack(tvdc_list).to(device)
        # [n, num_terminal_nodes]

        self.all_currents = torch.stack(
            [s['graph'].node_current_targets for s in samples]
        ).to(device)  # [n, N]

        # terminal_current_sign varies per sample (resistor signs are op-point dependent)
        self.all_terminal_current_sign = torch.stack(
            [s['graph'].terminal_current_sign for s in samples]
        ).to(device)  # [n, N]

        # ── SS / physics targets (optional, per-sample varying) ───────────
        g0 = samples[0]['graph']
        if hasattr(g0, 'node_log_gm') and g0.node_log_gm is not None:
            self.all_node_log_gm = torch.stack(
                [s['graph'].node_log_gm for s in samples]
            ).to(device)  # [n, N]
            self.all_node_log_gds = torch.stack(
                [s['graph'].node_log_gds for s in samples]
            ).to(device)  # [n, N]
        else:
            self.all_node_log_gm = None
            self.all_node_log_gds = None

        if hasattr(g0, 'mosfet_region_labels') and g0.mosfet_region_labels is not None:
            self.all_mosfet_region_labels = torch.stack(
                [s['graph'].mosfet_region_labels for s in samples]
            ).to(device)  # [n, M]
        else:
            self.all_mosfet_region_labels = None

        if hasattr(g0, 'node_mosfet_vth') and g0.node_mosfet_vth is not None:
            self.all_node_mosfet_vth = torch.stack(
                [s['graph'].node_mosfet_vth for s in samples]
            ).to(device)  # [n, N]
        else:
            self.all_node_mosfet_vth = None

        if hasattr(g0, 'mosfet_vth') and g0.mosfet_vth is not None:
            self.all_mosfet_vth = torch.stack(
                [s['graph'].mosfet_vth for s in samples]
            ).to(device)  # [n, M]
        else:
            self.all_mosfet_vth = None

        # AC targets (per-graph scalars)
        if hasattr(g0, 'ac_dc_gain') and g0.ac_dc_gain is not None:
            self.all_ac_dc_gain = torch.tensor(
                [s['graph'].ac_dc_gain.item() if s['graph'].ac_dc_gain.dim() == 0
                 else s['graph'].ac_dc_gain[0].item() for s in samples]
            ).to(device)  # [n]
            self.all_ac_ugbw = torch.tensor(
                [s['graph'].ac_ugbw.item() if s['graph'].ac_ugbw.dim() == 0
                 else s['graph'].ac_ugbw[0].item() for s in samples]
            ).to(device)
            self.all_ac_pm = torch.tensor(
                [s['graph'].ac_pm.item() if s['graph'].ac_pm.dim() == 0
                 else s['graph'].ac_pm[0].item() for s in samples]
            ).to(device)
            self.all_ac_am = torch.tensor(
                [s['graph'].ac_am.item() if s['graph'].ac_am.dim() == 0
                 else s['graph'].ac_am[0].item() for s in samples]
            ).to(device) if hasattr(g0, 'ac_am') and g0.ac_am is not None else None
            self.all_ac_valid = torch.tensor(
                [s['graph'].ac_valid.item() if hasattr(s['graph'], 'ac_valid') and s['graph'].ac_valid is not None
                 else True for s in samples], dtype=torch.bool
            ).to(device)
        else:
            self.all_ac_dc_gain = None
            self.all_ac_ugbw = None
            self.all_ac_pm = None
            self.all_ac_am = None
            self.all_ac_valid = None

        # SS normalization stats (filled in later)
        self.ss_gm_mean: float = 0.0
        self.ss_gm_std: float = 1.0
        self.ss_gds_mean: float = 0.0
        self.ss_gds_std: float = 1.0

        self.n_samples = len(samples)

        # Normalization stats (filled in after normalize_* calls)
        self.voltage_mean: float = 0.0
        self.voltage_std: float = 1.0
        self.current_mean: float = 0.0
        self.current_std: float = 1.0

        # Cache batch templates keyed by batch size
        self._batch_template_cache: dict = {}

    # ── Normalization ────────────────────────────────────────────────────

    def normalize_vdc(self, vdc_mean: float, vdc_std: float) -> None:
        self.all_vdc = (self.all_vdc - vdc_mean) / vdc_std
        self.all_node_voltage = (self.all_node_voltage - vdc_mean) / vdc_std
        self.all_terminal_vdc = (self.all_terminal_vdc - vdc_mean) / vdc_std
        self.voltage_mean = vdc_mean
        self.voltage_std = vdc_std

    def normalize_currents(self, current_mean: float, current_std: float) -> None:
        log_eps = 1e-12
        log_c = torch.log10(self.all_currents.abs() + log_eps)
        self.all_currents = (log_c - current_mean) / current_std
        self.current_mean = current_mean
        self.current_std = current_std

    def normalize_ss(self, gm_mean: float, gm_std: float,
                     gds_mean: float, gds_std: float) -> None:
        if self.all_node_log_gm is not None:
            self.all_node_log_gm = (self.all_node_log_gm - gm_mean) / gm_std
        if self.all_node_log_gds is not None:
            self.all_node_log_gds = (self.all_node_log_gds - gds_mean) / gds_std
        self.ss_gm_mean = gm_mean
        self.ss_gm_std = gm_std
        self.ss_gds_mean = gds_mean
        self.ss_gds_std = gds_std

    # ── Batch template (precomputed fixed parts for a given B) ───────────

    def _get_batch_template(self, B: int) -> dict:
        """Precompute and cache fixed topology components for batch size B."""
        if B in self._batch_template_cache:
            return self._batch_template_cache[B]

        N = self.num_nodes
        E = self.num_edges
        M = self.num_mosfets
        R = self.num_resistors
        C = self.num_capacitors
        dev = self.device

        # edge_index repeated with per-graph node offset
        node_offsets = torch.arange(B, device=dev) * N  # [B]
        edge_offsets = node_offsets.repeat_interleave(E)  # [B*E]
        edge_index = self.edge_index.repeat(1, B) + edge_offsets.unsqueeze(0)

        # batch vector and ptr
        batch_vec = torch.arange(B, device=dev).repeat_interleave(N)
        ptr = torch.arange(B + 1, device=dev) * N

        # type_tens tiled
        type_tens = self.type_tens.repeat(B, 1)

        # mosfet_info repeated (local per-graph indices — consumers add offsets via ptr)
        mosfet_info = self.mosfet_info.repeat(B, 1)
        mosfet_ptr = torch.arange(B + 1, device=dev) * M

        # resistor / capacitor
        resistor_info = self.resistor_info.repeat(B, 1)
        resistor_ptr = torch.arange(B + 1, device=dev) * R
        capacitor_info = self.capacitor_info.repeat(B, 1)
        capacitor_ptr = torch.arange(B + 1, device=dev) * C

        # boolean masks tiled
        has_current_mask = self.has_current_mask.repeat(B)
        output_node_mask = self.output_node_mask.repeat(B)
        train_mask = self.train_mask.repeat(B)
        mosfet_drain_mask = self.mosfet_drain_mask.repeat(B) if self.mosfet_drain_mask is not None else None
        kcl_include_mask = self.kcl_include_mask.repeat(B)
        terminal_train_mask = self.terminal_train_mask.repeat(B)

        tmpl = dict(
            edge_index=edge_index,
            batch_vec=batch_vec,
            ptr=ptr,
            type_tens=type_tens,
            mosfet_info=mosfet_info,
            mosfet_ptr=mosfet_ptr,
            resistor_info=resistor_info,
            resistor_ptr=resistor_ptr,
            capacitor_info=capacitor_info,
            capacitor_ptr=capacitor_ptr,
            has_current_mask=has_current_mask,
            output_node_mask=output_node_mask,
            train_mask=train_mask,
            mosfet_drain_mask=mosfet_drain_mask,
            kcl_include_mask=kcl_include_mask,
            terminal_train_mask=terminal_train_mask,
        )
        self._batch_template_cache[B] = tmpl
        return tmpl

    # ── Batch assembly ───────────────────────────────────────────────────

    def get_batch(self, indices: List[int]):
        """Assemble a batch for the given sample indices.

        Returns a SimpleNamespace that looks like a PyG Batch to train_v3.py.
        """
        B = len(indices)
        N = self.num_nodes
        M = self.num_mosfets
        idx = torch.tensor(indices, device=self.device)

        # Variable: index into stacked tensors
        x = self.all_x[idx].reshape(B * N, -1)
        node_voltage = self.all_node_voltage[idx].reshape(B * N)
        vdc = self.all_vdc[idx].reshape(-1, 1)              # [B*out_nodes, 1]
        terminal_vdc = self.all_terminal_vdc[idx].reshape(-1, 1)
        currents = self.all_currents[idx].reshape(B * N)
        terminal_current_sign = self.all_terminal_current_sign[idx].reshape(B * N)

        # Fixed: get precomputed template
        t = self._get_batch_template(B)

        # SS / physics targets (variable per sample)
        node_log_gm = self.all_node_log_gm[idx].reshape(B * N) if self.all_node_log_gm is not None else None
        node_log_gds = self.all_node_log_gds[idx].reshape(B * N) if self.all_node_log_gds is not None else None
        mosfet_region_labels = self.all_mosfet_region_labels[idx].reshape(B * M) if self.all_mosfet_region_labels is not None else None
        # Convert per-MOSFET region labels to per-node (needed by loss and eval code)
        if mosfet_region_labels is not None:
            node_region_labels = torch.full((B * N,), -1, dtype=torch.long, device=self.device)
            drain_idx_local = t['mosfet_info'][:, 1].long()  # [B*M] local drain indices
            graph_idx = torch.arange(B, device=self.device).repeat_interleave(M)
            drain_idx_global = drain_idx_local + t['ptr'][graph_idx]
            node_region_labels[drain_idx_global] = mosfet_region_labels.to(self.device)
        else:
            node_region_labels = None
        node_mosfet_vth = self.all_node_mosfet_vth[idx].reshape(B * N) if self.all_node_mosfet_vth is not None else None
        mosfet_vth = self.all_mosfet_vth[idx].reshape(B * M) if self.all_mosfet_vth is not None else None

        # AC targets (per-graph scalars)
        ac_dc_gain = self.all_ac_dc_gain[idx] if self.all_ac_dc_gain is not None else None
        ac_ugbw = self.all_ac_ugbw[idx] if self.all_ac_ugbw is not None else None
        ac_pm = self.all_ac_pm[idx] if self.all_ac_pm is not None else None
        ac_am = self.all_ac_am[idx] if self.all_ac_am is not None else None
        ac_valid = self.all_ac_valid[idx] if self.all_ac_valid is not None else None

        batch = SimpleNamespace(
            # graph structure
            x=x,
            edge_index=t['edge_index'],
            batch=t['batch_vec'],
            ptr=t['ptr'],
            num_graphs=B,
            num_terminals=self.num_terminals,
            # type info
            type_tens=t['type_tens'],
            net_type=None,
            # device info
            mosfet_info=t['mosfet_info'],
            mosfet_ptr=t['mosfet_ptr'],
            resistor_info=t['resistor_info'],
            resistor_ptr=t['resistor_ptr'],
            capacitor_info=t['capacitor_info'],
            capacitor_ptr=t['capacitor_ptr'],
            isource_info=None,
            isource_ptr=None,
            # masks
            has_current_mask=self.all_has_current_mask[idx].reshape(B * N),
            output_node_mask=t['output_node_mask'],
            train_mask=t['train_mask'],
            mosfet_drain_mask=t['mosfet_drain_mask'],
            terminal_current_sign=terminal_current_sign,
            kcl_include_mask=t['kcl_include_mask'],
            terminal_train_mask=t['terminal_train_mask'],
            # targets
            vdc=vdc,
            node_voltage_targets=node_voltage,
            terminal_vdc=terminal_vdc,
            node_current_targets=currents,
            # SS / physics targets
            node_log_gm=node_log_gm,
            node_log_gds=node_log_gds,
            mosfet_region_labels=mosfet_region_labels,
            node_region_labels=node_region_labels,
            node_mosfet_vth=node_mosfet_vth,
            mosfet_vth=mosfet_vth,
            # AC targets
            ac_dc_gain=ac_dc_gain,
            ac_ugbw=ac_ugbw,
            ac_pm=ac_pm,
            ac_am=ac_am,
            ac_valid=ac_valid,
            # normalization stats
            voltage_mean=self.voltage_mean,
            voltage_std=self.voltage_std,
            current_mean=self.current_mean,
            current_std=self.current_std,
            ss_gm_mean=self.ss_gm_mean,
            ss_gm_std=self.ss_gm_std,
            ss_gds_mean=self.ss_gds_mean,
            ss_gds_std=self.ss_gds_std,
        )
        return batch

    def __len__(self):
        return self.n_samples


class FixedTopologyLoader:
    """Drop-in replacement for PrebatchedLoader for fixed-topology datasets.

    Yields batches assembled from pre-loaded GPU tensors with minimal overhead.
    """

    def __init__(
        self,
        dataset: FixedTopologyDataset,
        batch_size: int = 1024,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        n = len(dataset)
        self._n_batches = (n + batch_size - 1) // batch_size

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            random.shuffle(indices)
        # Pre-build all batches for this epoch
        batches = []
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batches.append(self.dataset.get_batch(batch_idx))
        for batch in batches:
            yield batch

    def __len__(self):
        return self._n_batches


def build_fixed_topology_dataset(
    samples_file: Path,
    device: str = 'cpu',
) -> FixedTopologyDataset:
    """Load a dataset pkl and build a FixedTopologyDataset."""
    samples_file = Path(samples_file)
    print(f"  Loading {samples_file.name}...")
    with open(samples_file, 'rb') as f:
        samples = pickle.load(f)
    print(f"  Building fixed-topology dataset ({len(samples)} samples)...")
    ds = FixedTopologyDataset(samples, device=device)
    print(f"  Topology: {ds.num_nodes} nodes, {ds.num_edges} edges, "
          f"{ds.num_mosfets} MOSFETs")
    return ds
