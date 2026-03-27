"""
Graph builder for converting circuit netlists to PyTorch Geometric graphs.

Implements hierarchical type encoding and feature extraction for circuit
components (MOSFETs, passives, sources) and net nodes.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.data.encoding import build_circuit_trie
from src.circuits.parser import SPICENetlistParser


@dataclass
class TerminalNode:
    """Represents a device terminal node in the circuit graph."""
    name: str
    device_name: str
    device_type: str  # 'M', 'R', 'C', 'V', 'I'
    terminal_type: str  # 'D', 'G', 'S', 'B' for MOSFET; 'P', 'M' for others
    net: str
    props: Dict[str, float]  # w for MOSFET, dc for V, value for R/C
    ntype: Tuple[str, ...]  # ('M', 'N', 'D') etc


@dataclass
class NetNode:
    """Represents a net (voltage node) in the circuit graph."""
    name: str
    net_type: str  # 'gnd', 'vdd', 'input', 'output', 'internal'
    ntype: Tuple[str, ...]  # ('VNode', 'GND') or ('VNode', 'NGND')


class CircuitGraphBuilder:
    """Build PyTorch Geometric graphs from circuit netlists."""

    def __init__(self, normalize_props=False, param_specs=None, vdd=1.8,
                 device_to_group=None):
        """Initialize graph builder.

        Args:
            normalize_props: If True, apply min-max normalization to W and L using param ranges
            param_specs: Dict of parameter specs with 'min' and 'max' keys
            vdd: Supply voltage for voltage normalization (default: 1.8V)
            device_to_group: Optional mapping from device ID to group ID for group-based
                normalization, e.g. {'M8': 'GM1', 'M9': 'GM1', 'M10': 'GM2'}
        """
        self.trie = build_circuit_trie()
        self.enc_map = self.trie.get_leaf_encodings()
        self.type_dim = max(len(v) for v in self.enc_map.values())

        # Pad all encodings to same length
        for k in self.enc_map:
            enc = self.enc_map[k]
            if len(enc) < self.type_dim:
                self.enc_map[k] = enc + [0] * (self.type_dim - len(enc))

        # Normalization configuration
        self.normalize_props = normalize_props
        self.param_specs = param_specs or {}
        self.vdd = vdd
        self.device_to_group = device_to_group or {}
        self.prop_stats = {}

    def _normalize_mosfet_props(self, device_name: str, props: dict) -> dict:
        """Apply min-max normalization to W, L, and W/L ratio if normalize_props is enabled."""
        if not self.normalize_props:
            return props

        device_id = device_name.upper().replace('X', '')
        group_id = self.device_to_group.get(device_id)

        normalized = {}
        for prop_name in ['w', 'l']:
            value = props.get(prop_name, 0)
            # Try per-device key first (e.g. W_M1), then group key (e.g. W_GM1)
            param_key = f"{prop_name.upper()}_{device_id}"
            if param_key not in self.param_specs and group_id:
                param_key = f"{prop_name.upper()}_{group_id}"

            if param_key in self.param_specs:
                spec = self.param_specs[param_key]
                min_val = spec.get('min', value)
                max_val = spec.get('max', value)
                scale = spec.get('scale', 'linear')

                if scale == 'log':
                    log_val = math.log10(max(value, 1e-30))
                    log_min = math.log10(max(min_val, 1e-30))
                    log_max = math.log10(max(max_val, 1e-30))
                    normalized[prop_name] = (log_val - log_min) / (log_max - log_min) if log_max > log_min else 0.0
                else:
                    normalized[prop_name] = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
            else:
                normalized[prop_name] = value

        if 'wl_ratio' in props:
            wl = props['wl_ratio']
            # Look up W and L specs to compute proper log range
            w_key = f"W_{device_id}"
            if w_key not in self.param_specs and group_id:
                w_key = f"W_{group_id}"
            l_key = f"L_{device_id}"
            if l_key not in self.param_specs and group_id:
                l_key = f"L_{group_id}"
            w_spec = self.param_specs.get(w_key, {})
            l_spec = self.param_specs.get(l_key, {})
            if w_spec and l_spec and w_spec.get('scale') == 'log' and l_spec.get('scale') == 'log':
                wl_min = w_spec['min'] / l_spec['max']
                wl_max = w_spec['max'] / l_spec['min']
                log_wl = math.log10(max(wl, 1e-30))
                log_wl_min = math.log10(wl_min)
                log_wl_max = math.log10(wl_max)
                normalized['wl_ratio'] = (log_wl - log_wl_min) / (log_wl_max - log_wl_min) if log_wl_max > log_wl_min else 0.5
            else:
                log_wl = math.log10(max(wl, 1.0))
                normalized['wl_ratio'] = min(log_wl / 3.0, 1.0)

        if 'm' in props:
            m = props['m']
            # Log-normalize: effective M ranges from 1 to ~800
            # (base 1-100 * hardcoded multipliers up to 8x)
            # log10(1)=0, log10(10)=1.0, log10(100)=2.0, log10(800)=2.9
            normalized['m'] = min(math.log10(max(m, 1.0)) / 3.0, 1.0)

        return normalized

    def _normalize_passive_value(self, device_name: str, value: float, dev_type: str) -> float:
        """Normalize resistor/capacitor values using log-scale then min-max normalization."""
        if not self.normalize_props:
            return value

        if value <= 0:
            return 0.0

        device_id = device_name.upper().replace('X', '')
        group_id = self.device_to_group.get(device_id)

        param_candidates = [device_id]
        if group_id:
            param_candidates.append(group_id)

        spec = None
        for candidate in param_candidates:
            if candidate in self.param_specs:
                spec = self.param_specs[candidate]
                break

        if spec:
            min_val = spec.get('min', value)
            max_val = spec.get('max', value)
        else:
            if dev_type == 'R':
                min_val = 1.0
                max_val = 1e11
            else:
                min_val = 1e-16
                max_val = 1e-4

        log_value = np.log10(value)
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)

        if log_max > log_min:
            normalized = (log_value - log_min) / (log_max - log_min)
            return max(0.0, min(1.0, normalized))
        else:
            return 0.5

    def _normalize_voltage(self, voltage: float) -> float:
        """Normalize voltage using 2(V/VDD) - 1 to get [-1, 1] range."""
        return 2.0 * (voltage / self.vdd) - 1.0

    def _get_mos_type(self, model: str) -> str:
        """Determine if MOSFET is N or P type."""
        model_lower = model.lower() if model else ''
        if 'nmos' in model_lower or 'nfet' in model_lower or 'n_' in model_lower:
            return 'N'
        return 'P'

    def _get_terminal_type(self, dev_type: str, term: str, model: str = '') -> Tuple[str, ...]:
        """Get hierarchical type tuple for a terminal."""
        if dev_type == 'M':
            mos_type = self._get_mos_type(model)
            term_map = {'drain': 'D', 'gate': 'G', 'source': 'S', 'bulk': 'B',
                       'd': 'D', 'g': 'G', 's': 'S', 'b': 'B'}
            return ('M', mos_type, term_map.get(term.lower(), 'D'))
        else:
            term_map = {'p': 'P', 'n': 'M', '+': 'P', '-': 'M'}
            return (dev_type, term_map.get(term.lower(), 'P'))

    def _classify_net(self, net_name: str) -> str:
        """Classify net type based on standard SPICE naming conventions.

        Templates should use consistent names: vdd/vdda for supply, gnd/gnda/0 for ground,
        vin*/vp/vn for inputs, vout* for outputs. Everything else is internal.
        """
        net_lower = net_name.lower()
        if net_lower in ('0', 'gnd', 'vss') or 'gnda' in net_lower:
            return 'gnd'
        if 'vdd' in net_lower or 'vcc' in net_lower:
            return 'vdd'
        if 'vin' in net_lower or net_lower in ('vp', 'vn', 'vsig', 'inp', 'inn'):
            return 'input'
        if 'vout' in net_lower:
            return 'output'
        return 'internal'

    def build_from_netlist(self, netlist_path: str, sim_results: Dict,
                           vdd: float = 1.8, vcm: float = 0.9,
                           vin_p: float = 0.9, vin_n: float = 0.9,
                           i_ref: float = 20e-6,
                           mosfet_regions: Dict = None,
                           ac_metrics: Dict = None,
                           mosfet_ss_params: Dict = None,
                           excluded_current_devices: set = None) -> Data:
        """Build a graph from netlist and simulation results."""
        parser = SPICENetlistParser()
        components = parser.parse_file(netlist_path)

        # Collect all nets
        all_nets = set()
        for comp in components:
            all_nets.update(comp.nets)

        # Build terminal nodes
        terminals = []
        net_to_terminals = defaultdict(list)

        for comp in components:
            type_map = {
                'mosfet': 'M', 'resistor': 'R', 'capacitor': 'C',
                'voltage_source': 'V', 'current_source': 'I',
                'inductor': 'L', 'diode': 'D'
            }
            dev_type = type_map.get(comp.type, comp.type[0].upper())

            if dev_type == 'X':
                continue

            params = comp.params or {}

            if dev_type == 'M':
                w = params.get('w', 1e-6)
                l = params.get('l', 180e-9)
                m = params.get('m', 1.0)
                props = {'w': w, 'l': l, 'wl_ratio': w / l, 'm': m}
                props = self._normalize_mosfet_props(comp.name, props)
                term_names = ['drain', 'gate', 'source', 'bulk']
            elif dev_type == 'V':
                dc_val = params.get('dc', params.get('value', 0.0))
                dc_val_norm = self._normalize_voltage(dc_val)
                props = {'dc': dc_val_norm}
                term_names = ['p', 'n']
            elif dev_type == 'I':
                dc_val = params.get('dc', params.get('value', 0.0))
                props = {'dc': dc_val}
                term_names = ['p', 'n']
            elif dev_type in ['R', 'C']:
                value = params.get('value', 1.0)
                value = self._normalize_passive_value(comp.name, value, dev_type)
                props = {'value': value}
                term_names = ['p', 'n']
            else:
                continue

            model = comp.model or ''

            for term_name, net in zip(term_names, comp.nets):
                ntype = self._get_terminal_type(dev_type, term_name, model)
                term = TerminalNode(
                    name=f"{comp.name}_{term_name}",
                    device_name=comp.name,
                    device_type=dev_type,
                    terminal_type=term_name,
                    net=net,
                    props=props.copy(),
                    ntype=ntype
                )
                terminals.append(term)
                net_to_terminals[net].append(len(terminals) - 1)

        # Build net nodes
        nets = []
        for net_name in sorted(all_nets):
            net_type = self._classify_net(net_name)
            if net_type == 'gnd':
                ntype = ('VNode', 'GND')
            else:
                ntype = ('VNode', 'NGND')
            nets.append(NetNode(name=net_name, net_type=net_type, ntype=ntype))

        # Build edges (terminal <-> net) with optional edge type features
        edges = []
        edge_type_list = []
        EDGE_TYPE_MAP = {'gate': 0, 'drain': 1, 'source': 2, 'bulk': 3, 'p': 4, 'n': 5}
        num_edge_types = len(EDGE_TYPE_MAP)
        net_to_idx = {net.name: len(terminals) + i for i, net in enumerate(nets)}

        for term_idx, term in enumerate(terminals):
            net_idx = net_to_idx.get(term.net)
            if net_idx is not None:
                edges.append([term_idx, net_idx])
                edges.append([net_idx, term_idx])
                # One-hot edge type for both directions
                etype = [0.0] * num_edge_types
                etype[EDGE_TYPE_MAP.get(term.terminal_type, 0)] = 1.0
                edge_type_list.append(etype)
                edge_type_list.append(etype)

        # Build feature tensors
        x_list = []
        type_enc_list = []

        for term in terminals:
            node_x = list(term.props.values())
            x_list.append(torch.tensor(node_x, dtype=torch.float))
            enc = self.enc_map.get(term.ntype, [0] * self.type_dim)
            type_enc_list.append(torch.tensor(enc, dtype=torch.float))

        for net in nets:
            x_list.append(torch.tensor([], dtype=torch.float))
            enc = self.enc_map.get(net.ntype, [0] * self.type_dim)
            type_enc_list.append(torch.tensor(enc, dtype=torch.float))

        x = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True)
        type_tens = torch.stack(type_enc_list)

        # Add global features
        num_nodes = len(terminals) + len(nets)
        global_features = torch.tensor([[
            self._normalize_voltage(vdd),
            self._normalize_voltage(vcm),
            self._normalize_voltage(vin_p),
            self._normalize_voltage(vin_n),
            np.log10(i_ref) + 5,
        ]], dtype=torch.float).expand(num_nodes, -1)

        x = torch.cat([x, global_features], dim=-1)

        # Build masks
        output_node_mask = torch.zeros(len(terminals) + len(nets), dtype=torch.bool)
        for i, net in enumerate(nets):
            if net.ntype == ('VNode', 'NGND'):
                output_node_mask[len(terminals) + i] = True

        known_voltage_mask = torch.zeros(len(terminals) + len(nets), dtype=torch.bool)
        for i, term in enumerate(terminals):
            net_type = self._classify_net(term.net)
            known_voltage_mask[i] = net_type in ['gnd', 'vdd', 'input']
        for i, net in enumerate(nets):
            known_voltage_mask[len(terminals) + i] = net.net_type in ['gnd', 'vdd', 'input']

        # Build voltage targets
        node_voltages = sim_results.get('node_voltages', {})
        voltage_targets = torch.zeros(len(terminals) + len(nets), dtype=torch.float)

        for i, term in enumerate(terminals):
            v = self._get_voltage(term.net, node_voltages)
            voltage_targets[i] = v
        for i, net in enumerate(nets):
            v = self._get_voltage(net.name, node_voltages)
            voltage_targets[len(terminals) + i] = v

        train_mask = output_node_mask & ~known_voltage_mask
        train_indices = train_mask.nonzero(as_tuple=True)[0]
        vdc = voltage_targets[train_indices].unsqueeze(-1)

        # Terminal supervision mask: terminals connected to unknown internal nets
        terminal_train_mask = torch.zeros(len(terminals) + len(nets), dtype=torch.bool)
        for i, term in enumerate(terminals):
            net_type = self._classify_net(term.net)
            if net_type not in ['gnd', 'vdd', 'input']:
                terminal_train_mask[i] = True

        terminal_train_indices = terminal_train_mask.nonzero(as_tuple=True)[0]
        terminal_vdc = voltage_targets[terminal_train_indices].unsqueeze(-1)

        # Build current targets
        device_currents = sim_results.get('device_currents', {})
        current_targets, has_current_mask, terminal_current_sign, kcl_include_mask = self._create_current_targets(
            terminals, nets, device_currents, i_ref,
            excluded_current_devices=excluded_current_devices,
        )

        # Build MOSFET info
        mosfet_info, mosfet_device_names = self._create_mosfet_info(terminals, net_to_idx)
        mosfet_region_labels = self._create_mosfet_region_labels(
            mosfet_device_names, mosfet_regions
        )
        mosfet_vth = self._create_mosfet_vth(mosfet_device_names, mosfet_regions)

        # Build device info for physics-based current computation
        resistor_info = self._create_resistor_info(terminals, net_to_idx)
        capacitor_info = self._create_capacitor_info(terminals, net_to_idx)
        vsource_info = self._create_vsource_info(terminals, net_to_idx)
        isource_info = self._create_isource_info(terminals, net_to_idx, i_ref)

        # Build loop info for loop attention
        loop_edge_index, device_terminal_map = self._create_loop_info(terminals, net_to_terminals)

        # Create physics constraint tensors (lazy import to avoid circular dependency)
        from src.training.current_constraints import create_constraint_tensors
        diff_pair_constraints, mirror_constraints, output_stage_constraints, lambda_mirror_constraints = create_constraint_tensors(
            mosfet_device_names
        )

        # Create Data object
        edge_attr = torch.tensor(edge_type_list, dtype=torch.float) if edge_type_list else torch.zeros((0, num_edge_types), dtype=torch.float)
        data = Data(
            x=x,
            type_tens=type_tens,
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long),
            edge_attr=edge_attr,
            output_node_mask=output_node_mask,
            known_voltage_mask=known_voltage_mask,
            train_mask=train_mask,
            node_voltage_targets=voltage_targets,
            vdc=vdc,
            node_current_targets=current_targets,
            has_current_mask=has_current_mask,
            terminal_current_sign=terminal_current_sign,
            kcl_include_mask=kcl_include_mask,
            mosfet_info=mosfet_info,
            mosfet_region_labels=mosfet_region_labels,
            resistor_info=resistor_info,
            capacitor_info=capacitor_info,
            vsource_info=vsource_info,
            isource_info=isource_info,
            diff_pair_constraints=diff_pair_constraints,
            mirror_constraints=mirror_constraints,
            output_stage_constraints=output_stage_constraints,
            lambda_mirror_constraints=lambda_mirror_constraints,
            terminal_train_mask=terminal_train_mask,
            terminal_vdc=terminal_vdc,
            loop_edge_index=loop_edge_index,
            device_terminal_map=device_terminal_map,
        )
        data.num_devices = device_terminal_map.shape[0]

        data.num_terminals = len(terminals)
        data.num_nets = len(nets)
        data.node_types = [t.ntype for t in terminals] + [n.ntype for n in nets]
        data.node_names = [t.name for t in terminals] + [n.name for n in nets]

        # AC metrics (graph-level scalars from .ac SPICE simulation)
        if ac_metrics is not None:
            data.ac_ugbw = torch.tensor([ac_metrics.get('ugbw') or 0.0], dtype=torch.float)
            data.ac_pm = torch.tensor([ac_metrics.get('pm') or 0.0], dtype=torch.float)
            data.ac_am = torch.tensor([ac_metrics.get('am') or 0.0], dtype=torch.float)
            data.ac_dc_gain = torch.tensor([ac_metrics.get('dc_gain') or 0.0], dtype=torch.float)
            # Track which metrics are valid (requires positive DC gain for meaningful UGBW/PM)
            dc_gain = ac_metrics.get('dc_gain')
            data.ac_valid = torch.tensor([
                ac_metrics.get('ugbw') is not None and
                ac_metrics.get('pm') is not None and
                dc_gain is not None and dc_gain > 0
            ], dtype=torch.bool)

        # Small-signal parameters per MOSFET (ordered same as mosfet_info)
        if mosfet_ss_params is not None and mosfet_device_names:
            gm_list = []
            gds_list = []
            for name in mosfet_device_names:
                ss = mosfet_ss_params.get(name.lower(), {})
                gm_list.append(ss.get('gm', 0.0))
                gds_list.append(ss.get('gds', 0.0))
            data.mosfet_gm = torch.tensor(gm_list, dtype=torch.float)
            data.mosfet_gds = torch.tensor(gds_list, dtype=torch.float)

            # Per-node SS targets and mask (PyG-compatible batching via concatenation)
            # This avoids manual index offsetting needed by per-MOSFET tensors
            import math
            num_all_nodes = len(terminals) + len(nets)
            mosfet_drain_mask = torch.zeros(num_all_nodes, dtype=torch.bool)
            node_log_gm = torch.zeros(num_all_nodes, dtype=torch.float)
            node_log_gds = torch.zeros(num_all_nodes, dtype=torch.float)
            for i, row in enumerate(mosfet_info):
                drain_term_idx = row[1].item()
                gm_val = gm_list[i]
                gds_val = gds_list[i]
                if gm_val > 1e-12:
                    mosfet_drain_mask[drain_term_idx] = True
                    node_log_gm[drain_term_idx] = math.log10(gm_val)
                    node_log_gds[drain_term_idx] = math.log10(max(gds_val, 1e-20))
            data.mosfet_drain_mask = mosfet_drain_mask
            data.node_log_gm = node_log_gm
            data.node_log_gds = node_log_gds

        # Per-MOSFET Vth (for physics losses)
        data.mosfet_vth = mosfet_vth

        # Per-node Vth at drain terminals (same scatter as node_log_gm)
        num_all_nodes = len(terminals) + len(nets)
        node_mosfet_vth = torch.zeros(num_all_nodes, dtype=torch.float)
        for i, row in enumerate(mosfet_info):
            drain_term_idx = row[1].item()
            node_mosfet_vth[drain_term_idx] = mosfet_vth[i].item()
        data.node_mosfet_vth = node_mosfet_vth

        return data

    def _get_voltage(self, net_name: str, voltages: Dict[str, float]) -> float:
        """Get voltage with case-insensitive matching."""
        if net_name in voltages:
            return voltages[net_name]
        net_upper = net_name.upper()
        for k, v in voltages.items():
            if k.upper() == net_upper:
                return v
        return 0.0

    def _get_current(self, device_name: str, currents: Dict[str, float]) -> float:
        """Get device current with case-insensitive matching."""
        if not currents:
            return 0.0
        if device_name in currents:
            return currents[device_name]
        device_upper = device_name.upper()
        for k, v in currents.items():
            if k.upper() == device_upper:
                return v
        return 0.0

    def _create_current_targets(
        self,
        terminals: List[TerminalNode],
        nets: List[NetNode],
        device_currents: Dict[str, float],
        i_ref: float = 0.0,
        excluded_current_devices: set = None,
        min_current_threshold: float = 1e-15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create current targets for terminal nodes.

        Unified sign convention from SPICE current direction:
        - SPICE: positive current flows p → n
        - p terminal: current leaves net → sign = -1
        - n terminal: current enters net → sign = +1
        - If SPICE current is negative (flows n → p), signs flip
        - MOSFET gate/bulk: zero DC current, excluded (sign=0)
        """
        num_nodes = len(terminals) + len(nets)
        current_targets = torch.zeros(num_nodes, dtype=torch.float)
        has_current_mask = torch.zeros(num_nodes, dtype=torch.bool)
        terminal_current_sign = torch.zeros(num_nodes, dtype=torch.float)
        kcl_include_mask = torch.zeros(num_nodes, dtype=torch.bool)

        excl_devices = excluded_current_devices or set()

        for i, term in enumerate(terminals):
            device_type = term.device_type
            term_type = term.terminal_type
            device_name = term.device_name

            if device_name.lower() in excl_devices:
                continue

            # MOSFET gate/bulk: zero DC current, excluded
            if device_type == 'M' and term_type in ('gate', 'bulk'):
                continue

            # Get device current
            if device_type == 'I':
                device_current = i_ref
                has_target = True
            else:
                device_current = self._get_current(device_name, device_currents)
                has_target = device_name.lower() in [k.lower() for k in device_currents.keys()]

            abs_current = abs(device_current)
            significant = bool(abs_current > min_current_threshold)

            # Target is always magnitude — supervised via MSE regardless of KCL
            current_targets[i] = abs_current
            # Capacitors: DC current is 0 but include for KCL completeness
            if device_type == 'C':
                has_current_mask[i] = True
                kcl_include_mask[i] = True
                current_targets[i] = 0.0
            else:
                has_current_mask[i] = has_target and significant
                # KCL inclusion: all devices with significant current
                # (including V-sources for supply net KCL)
                if device_type == 'M':
                    kcl_include_mask[i] = True
                else:
                    kcl_include_mask[i] = significant

            # Sign convention for KCL:
            # - MOSFET: fixed by device type (NMOS drain=-1, source=+1, etc.)
            #   Drain current direction doesn't reverse in normal operation.
            # - All others (R, I, C, V): use actual SPICE current direction
            #   SPICE convention: positive current flows p→n
            #   If device_current > 0: p leaves net (-1), n enters net (+1)
            #   If device_current < 0: p enters net (+1), n leaves net (-1)
            #   This correctly handles resistors whose current reverses
            #   (e.g. Rfb at vout when Xm6 enters cutoff).
            if device_type == 'M':
                is_nmos = term.ntype[1] == 'N'
                if term_type == 'drain':
                    terminal_current_sign[i] = -1.0 if is_nmos else 1.0
                elif term_type == 'source':
                    terminal_current_sign[i] = 1.0 if is_nmos else -1.0
            else:
                if device_current >= 0:
                    terminal_current_sign[i] = -1.0 if term_type == 'p' else 1.0
                else:
                    terminal_current_sign[i] = 1.0 if term_type == 'p' else -1.0

        return current_targets, has_current_mask, terminal_current_sign, kcl_include_mask

    def _create_mosfet_info(
        self,
        terminals: List[TerminalNode],
        net_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, List[str]]:
        """Create MOSFET terminal info for operating region loss."""
        device_terminals = {}
        for i, term in enumerate(terminals):
            if term.device_type != 'M':
                continue
            dev_name = term.device_name
            if dev_name not in device_terminals:
                device_terminals[dev_name] = {}
            device_terminals[dev_name][term.terminal_type] = (i, term)

        mosfet_info = []
        device_names = []
        for dev_name, terms in device_terminals.items():
            gate_term = terms.get('gate') or terms.get('g')
            drain_term = terms.get('drain') or terms.get('d')
            source_term = terms.get('source') or terms.get('s')

            if not all([gate_term, drain_term, source_term]):
                continue

            gate_term_idx, gate_node = gate_term
            drain_term_idx, drain_node = drain_term
            source_term_idx, source_node = source_term

            gate_net_idx = net_to_idx.get(gate_node.net, -1)
            drain_net_idx = net_to_idx.get(drain_node.net, -1)
            source_net_idx = net_to_idx.get(source_node.net, -1)

            if -1 in [gate_net_idx, drain_net_idx, source_net_idx]:
                continue

            is_nmos = 1 if gate_node.ntype[1] == 'N' else 0

            mosfet_info.append([
                gate_term_idx, drain_term_idx, source_term_idx,
                gate_net_idx, drain_net_idx, source_net_idx,
                is_nmos
            ])
            device_names.append(dev_name)

        if mosfet_info:
            return torch.tensor(mosfet_info, dtype=torch.long), device_names
        else:
            return torch.zeros((0, 7), dtype=torch.long), []

    def _create_loop_info(
        self,
        terminals: List,
        net_to_terminals: Dict[str, List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect circuit loops via NetworkX and build loop edge index + device terminal map.

        Returns:
            loop_edge_index: [2, E_loop] device-level edges connecting devices in same loop
            device_terminal_map: [D, 4] terminal node indices per device (-1 padded)
        """
        # Build device → terminal indices mapping
        device_terms = {}  # dev_name -> list of terminal node indices
        device_nets = {}   # dev_name -> set of net names
        for i, term in enumerate(terminals):
            dev = term.device_name
            if dev not in device_terms:
                device_terms[dev] = []
                device_nets[dev] = set()
            device_terms[dev].append(i)
            device_nets[dev].add(term.net)

        device_names = list(device_terms.keys())
        dev_to_idx = {name: i for i, name in enumerate(device_names)}
        num_devices = len(device_names)

        # Build device-level graph: edge if two devices share a net
        G = nx.Graph()
        G.add_nodes_from(range(num_devices))

        # net → list of device indices
        net_to_devices = defaultdict(set)
        for dev_name, nets in device_nets.items():
            for net in nets:
                net_to_devices[net].add(dev_to_idx[dev_name])

        for net, devs in net_to_devices.items():
            devs = list(devs)
            for i in range(len(devs)):
                for j in range(i + 1, len(devs)):
                    G.add_edge(devs[i], devs[j])

        # Find fundamental cycles
        cycles = nx.cycle_basis(G)

        # Build loop_edge_index: connect all device pairs within each cycle
        loop_edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                for j in range(i + 1, len(cycle)):
                    a, b = min(cycle[i], cycle[j]), max(cycle[i], cycle[j])
                    loop_edges.add((a, b))

        # Make bidirectional
        if loop_edges:
            src, dst = [], []
            for a, b in loop_edges:
                src.extend([a, b])
                dst.extend([b, a])
            loop_edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            loop_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Build device_terminal_map: [D, 4] padded with -1
        max_terms = 4
        device_terminal_map = torch.full((num_devices, max_terms), -1, dtype=torch.long)
        for dev_name, term_indices in device_terms.items():
            d = dev_to_idx[dev_name]
            for t, idx in enumerate(term_indices[:max_terms]):
                device_terminal_map[d, t] = idx

        return loop_edge_index, device_terminal_map

    def _create_mosfet_region_labels(
        self,
        device_names: List[str],
        mosfet_regions: Dict
    ) -> torch.Tensor:
        """Create ground-truth operating region labels from SPICE simulation results."""
        if not mosfet_regions or not device_names:
            return torch.full((len(device_names),), -1, dtype=torch.long)

        region_to_label = {'cutoff': 0, 'triode': 1, 'saturation': 2, 'unknown': -1}

        labels = []
        for dev_name in device_names:
            info = mosfet_regions.get(dev_name.lower())
            if info is None:
                alt_name = dev_name.lower().lstrip('x')
                info = mosfet_regions.get(alt_name)

            if info is not None:
                region = info.get('region', 'unknown')
                labels.append(region_to_label.get(region, -1))
            else:
                labels.append(-1)

        return torch.tensor(labels, dtype=torch.long)

    def _create_mosfet_vth(
        self,
        device_names: List[str],
        mosfet_regions: Dict
    ) -> torch.Tensor:
        """Extract per-MOSFET Vth from SPICE simulation results for gm physics loss."""
        if not mosfet_regions or not device_names:
            return torch.zeros(len(device_names), dtype=torch.float)

        vth_list = []
        for dev_name in device_names:
            info = mosfet_regions.get(dev_name.lower())
            if info is None:
                alt_name = dev_name.lower().lstrip('x')
                info = mosfet_regions.get(alt_name)

            if info is not None and info.get('vth') is not None:
                vth_list.append(info['vth'])
            else:
                vth_list.append(0.0)

        return torch.tensor(vth_list, dtype=torch.float)

    def _create_resistor_info(
        self,
        terminals: List[TerminalNode],
        net_to_idx: Dict[str, int]
    ) -> torch.Tensor:
        """Create resistor info tensor for physics-based current computation.

        Returns tensor [num_resistors, 5]:
            [term_p_idx, term_n_idx, net_p_idx, net_n_idx, r_value_normalized]
        """
        device_terminals = defaultdict(dict)
        for i, term in enumerate(terminals):
            if term.device_type == 'R':
                device_terminals[term.device_name][term.terminal_type] = (i, term)

        resistor_info = []
        for dev_name, terms in device_terminals.items():
            if 'p' in terms and 'n' in terms:
                p_idx, p_term = terms['p']
                n_idx, n_term = terms['n']
                net_p_idx = net_to_idx.get(p_term.net, -1)
                net_n_idx = net_to_idx.get(n_term.net, -1)

                if -1 in [net_p_idx, net_n_idx]:
                    continue

                # Get normalized R value from terminal props
                r_value = p_term.props.get('value', 1.0)
                resistor_info.append([p_idx, n_idx, net_p_idx, net_n_idx, r_value])

        if resistor_info:
            return torch.tensor(resistor_info, dtype=torch.float)
        return torch.zeros((0, 5), dtype=torch.float)

    def _create_capacitor_info(
        self,
        terminals: List[TerminalNode],
        net_to_idx: Dict[str, int]
    ) -> torch.Tensor:
        """Create capacitor info tensor. For DC analysis, I=0 so we only need terminal indices.

        Returns tensor [num_capacitors, 4]:
            [term_p_idx, term_n_idx, net_p_idx, net_n_idx]
        """
        device_terminals = defaultdict(dict)
        for i, term in enumerate(terminals):
            if term.device_type == 'C':
                device_terminals[term.device_name][term.terminal_type] = (i, term)

        capacitor_info = []
        for dev_name, terms in device_terminals.items():
            if 'p' in terms and 'n' in terms:
                p_idx, p_term = terms['p']
                n_idx, n_term = terms['n']
                net_p_idx = net_to_idx.get(p_term.net, -1)
                net_n_idx = net_to_idx.get(n_term.net, -1)

                if -1 in [net_p_idx, net_n_idx]:
                    continue

                capacitor_info.append([p_idx, n_idx, net_p_idx, net_n_idx])

        if capacitor_info:
            return torch.tensor(capacitor_info, dtype=torch.long)
        return torch.zeros((0, 4), dtype=torch.long)

    def _create_vsource_info(
        self,
        terminals: List[TerminalNode],
        net_to_idx: Dict[str, int]
    ) -> torch.Tensor:
        """Create voltage source info tensor.

        Returns tensor [num_vsources, 4]:
            [term_p_idx, term_n_idx, net_p_idx, net_n_idx]
        """
        device_terminals = defaultdict(dict)
        for i, term in enumerate(terminals):
            if term.device_type == 'V':
                device_terminals[term.device_name][term.terminal_type] = (i, term)

        vsource_info = []
        for dev_name, terms in device_terminals.items():
            if 'p' in terms and 'n' in terms:
                p_idx, p_term = terms['p']
                n_idx, n_term = terms['n']
                net_p_idx = net_to_idx.get(p_term.net, -1)
                net_n_idx = net_to_idx.get(n_term.net, -1)

                if -1 in [net_p_idx, net_n_idx]:
                    continue

                vsource_info.append([p_idx, n_idx, net_p_idx, net_n_idx])

        if vsource_info:
            return torch.tensor(vsource_info, dtype=torch.long)
        return torch.zeros((0, 4), dtype=torch.long)

    def _create_isource_info(
        self,
        terminals: List[TerminalNode],
        net_to_idx: Dict[str, int],
        i_ref: float
    ) -> torch.Tensor:
        """Create current source info tensor for physics-based current computation.

        Returns tensor [num_isources, 5]:
            [term_p_idx, term_n_idx, net_p_idx, net_n_idx, i_ref_normalized]

        The i_ref value is normalized to log10 scale (same as current targets).
        """
        device_terminals = defaultdict(dict)
        for i, term in enumerate(terminals):
            if term.device_type == 'I':
                device_terminals[term.device_name][term.terminal_type] = (i, term)

        isource_info = []
        for dev_name, terms in device_terminals.items():
            if 'p' in terms and 'n' in terms:
                p_idx, p_term = terms['p']
                n_idx, n_term = terms['n']
                net_p_idx = net_to_idx.get(p_term.net, -1)
                net_n_idx = net_to_idx.get(n_term.net, -1)

                if -1 in [net_p_idx, net_n_idx]:
                    continue

                # Normalize i_ref to log10 scale (same as current targets)
                # log10(20uA) ≈ -4.7, after +5 offset ≈ 0.3
                i_ref_norm = np.log10(abs(i_ref)) if i_ref != 0 else -15.0
                isource_info.append([p_idx, n_idx, net_p_idx, net_n_idx, i_ref_norm])

        if isource_info:
            return torch.tensor(isource_info, dtype=torch.float)
        return torch.zeros((0, 5), dtype=torch.float)
