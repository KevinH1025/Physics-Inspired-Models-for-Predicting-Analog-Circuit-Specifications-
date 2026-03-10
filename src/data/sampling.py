"""
Parameter sampling and netlist generation utilities.

This module provides Latin Hypercube Sampling (LHS) for efficient parameter
space exploration and netlist generation from templates.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def generate_lhs_samples(param_specs: Dict, num_samples: int, seed: int = 42) -> List[Dict]:
    """
    Generate samples using Latin Hypercube Sampling.

    Args:
        param_specs: Dict of {name: {'min': float, 'max': float, 'scale': 'log'|'linear'}}
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of parameter dictionaries
    """
    var_params = [(name, spec) for name, spec in param_specs.items()
                  if 'min' in spec and 'max' in spec]

    if not var_params:
        return [{}] * num_samples

    n_dims = len(var_params)

    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        lhs_samples = sampler.random(n=num_samples)
    else:
        np.random.seed(seed)
        lhs_samples = np.random.rand(num_samples, n_dims)

    all_params = []
    for i in range(num_samples):
        params = {}
        for j, (name, spec) in enumerate(var_params):
            u = lhs_samples[i, j]
            min_val, max_val = spec['min'], spec['max']
            scale = spec.get('scale', 'linear')

            if scale == 'log':
                log_min, log_max = np.log10(max(min_val, 1e-15)), np.log10(max_val)
                params[name] = 10 ** (log_min + u * (log_max - log_min))
            else:
                params[name] = min_val + u * (max_val - min_val)

            # Round to integer if type: int specified
            if spec.get('type') == 'int':
                params[name] = int(round(params[name]))

        for name, spec in param_specs.items():
            if 'value' in spec:
                params[name] = spec['value']

        all_params.append(params)

    return all_params


def generate_netlist(template_path: str, params: Dict, output_path: str) -> str:
    """Generate netlist from template with given parameters."""
    with open(template_path, 'r') as f:
        template = f.read()

    netlist = template
    for name, value in params.items():
        placeholder = f'{{{name}}}'
        if isinstance(value, float):
            if abs(value) < 1e-9:
                formatted = f'{value*1e12:.3f}p'
            elif abs(value) < 1e-6:
                formatted = f'{value*1e9:.3f}n'
            elif abs(value) < 1e-3:
                formatted = f'{value*1e6:.3f}u'
            elif abs(value) < 1:
                formatted = f'{value*1e3:.3f}m'
            else:
                formatted = f'{value:.6g}'
        else:
            formatted = str(value)
        netlist = netlist.replace(placeholder, formatted)

    with open(output_path, 'w') as f:
        f.write(netlist)

    return output_path


def _eval_constraint_expr(expr: str, params: Dict):
    """Evaluate a constraint expression, supporting multiplication.

    Examples:
        'W_GM3'          -> params['W_GM3']
        'W_GM3*M_GM3'    -> params['W_GM3'] * params['M_GM3']
        '0.5*W_GM3'      -> 0.5 * params['W_GM3']

    Returns float value or None if any parameter is missing.
    """
    parts = [p.strip() for p in expr.split('*')]
    result = 1.0
    for part in parts:
        val = params.get(part)
        if val is None:
            try:
                val = float(part)
            except ValueError:
                return None
        result *= float(val)
    return result


def _check_constraint(params: Dict, constraint: str) -> bool:
    """Check a parameter ordering constraint.

    Supports chained >= and <= comparisons with expressions:
        'W_GM3 >= W_GM2 >= W_GM1'              (simple)
        'W_GM3*M_GM3 >= W_GM2*M_GM2'           (with multipliers)
    Returns True if the constraint is satisfied.
    """
    if '>=' in constraint:
        parts = [p.strip() for p in constraint.split('>=')]
        for i in range(len(parts) - 1):
            a = _eval_constraint_expr(parts[i], params)
            b = _eval_constraint_expr(parts[i + 1], params)
            if a is None or b is None:
                return True  # skip if params not present
            if a < b:
                return False
    elif '<=' in constraint:
        parts = [p.strip() for p in constraint.split('<=')]
        for i in range(len(parts) - 1):
            a = _eval_constraint_expr(parts[i], params)
            b = _eval_constraint_expr(parts[i + 1], params)
            if a is None or b is None:
                return True
            if a > b:
                return False
    return True


def worker_generate_sample(args: Tuple) -> Tuple[int, Optional[Dict], str]:
    """
    Worker function for parallel sample generation.

    Args:
        args: Tuple of (sample_idx, params, template_path, output_dir, vdd, rail_margin,
              filter_enabled, exclude_nodes, normalize_props, param_specs, saturation_filter,
              ac_template_path, device_to_group, excluded_current_devices)

    Returns:
        Tuple of (sample_idx, sample_dict or None, status: 'ok'/'failed'/'filtered')
    """
    from src.circuits.simulator import CircuitSimulator
    from src.data.graph_builder import CircuitGraphBuilder

    logging.getLogger('PySpice.Spice.NgSpice.Shared').setLevel(logging.ERROR)

    # Unpack args - optional trailing elements after the 11 base args
    base_args = args[:11]
    (sample_idx, params, template_path, output_dir, vdd, rail_margin,
     filter_enabled, exclude_nodes, normalize_props, param_specs,
     saturation_filter) = base_args
    ac_template_path = args[11] if len(args) > 11 else None
    device_to_group = args[12] if len(args) > 12 else None
    excluded_current_devices = args[13] if len(args) > 13 else None
    param_constraints = args[14] if len(args) > 14 else None
    max_cutoff_devices = args[15] if len(args) > 15 else None
    require_ac_convergence = args[16] if len(args) > 16 else False
    min_ugbw_hz = args[17] if len(args) > 17 else None

    try:
        # Check parameter constraints (e.g., W_GM3 >= W_GM2 >= W_GM1)
        if param_constraints:
            for constraint in param_constraints:
                if not _check_constraint(params, constraint):
                    return (sample_idx, None, 'filtered_constraint')

        # Handle VIN_P/VIN_N computation from VCM/VDIFF
        if 'VIN_P' not in params and 'VCM' in params:
            params['VIN_P'] = params['VCM'] + params.get('VDIFF', 0) / 2
            params['VIN_N'] = params['VCM'] - params.get('VDIFF', 0) / 2

        # Generate original netlist (for graph parsing - preserves graph structure)
        netlist_path = Path(output_dir) / f'temp_netlist_{sample_idx}.sp'
        generate_netlist(template_path, params, str(netlist_path))

        # Determine which netlist to simulate
        if ac_template_path:
            # Generate AC netlist (for simulation - has loop-breaking + .ac)
            ac_netlist_path = Path(output_dir) / f'temp_ac_netlist_{sample_idx}.sp'
            generate_netlist(ac_template_path, params, str(ac_netlist_path))

            with open(ac_netlist_path, 'r') as f:
                sim_netlist_content = f.read()

            # Simulate with AC template (gets both DC + AC results)
            simulator = CircuitSimulator(analysis_types=['dc', 'ac'])
            sim_results = simulator.simulate(sim_netlist_content)

            ac_netlist_path.unlink(missing_ok=True)

            # Map AC template node names back to original (vp_fb/vp_gate -> vp)
            if sim_results and '_all_net_voltages' in sim_results:
                sim_results['_all_net_voltages'] = CircuitSimulator.map_ac_node_voltages(
                    sim_results['_all_net_voltages']
                )

            # Compute analytical AC metrics for comparison
            cc = params.get('C_C', 1e-12)
            rz = params.get('R_Z', 0.0)
            r_in = params.get('R_IN', 50e3)
            r_fb = params.get('R_F', 50e3)
            w_m6 = params.get('W_M6', 50e-6)
            analytical_ac = simulator.compute_analytical_ac(
                sim_results, cc=cc, rz=rz, r_in=r_in, r_fb=r_fb, w_m6=w_m6
            )
            sim_results.update(analytical_ac)
        else:
            # Original DC-only simulation
            with open(netlist_path, 'r') as f:
                sim_netlist_content = f.read()

            simulator = CircuitSimulator()
            sim_results = simulator.simulate(sim_netlist_content)

        if sim_results is None or '_all_net_voltages' not in sim_results:
            netlist_path.unlink(missing_ok=True)
            return (sample_idx, None, 'failed')

        node_voltages = sim_results.get('_all_net_voltages', {})
        device_currents = sim_results.get('_all_device_currents', {})

        if not node_voltages:
            netlist_path.unlink(missing_ok=True)
            return (sample_idx, None, 'failed')

        # Filter: check if any signal node is near rails
        if filter_enabled:
            for net, v in node_voltages.items():
                if '#' in net or net.startswith('m.') or net.startswith('@'):
                    continue
                if net.lower() in ['0', 'gnd', 'vss', 'vdd']:
                    continue
                if 'vdd' in net.lower() or 'vss' in net.lower():
                    continue
                if net.lower() in exclude_nodes:
                    continue

                if v < rail_margin or v > (vdd - rail_margin):
                    netlist_path.unlink(missing_ok=True)
                    return (sample_idx, None, 'filtered')

        # Saturation filter
        if saturation_filter:
            mosfet_regions = sim_results.get('_mosfet_regions', {})
            if mosfet_regions:
                for device, info in mosfet_regions.items():
                    region = info.get('region', 'unknown')
                    if region not in ('saturation', 'unknown'):
                        netlist_path.unlink(missing_ok=True)
                        return (sample_idx, None, 'filtered_saturation')

        # Max cutoff devices filter — drop circuits where too many MOSFETs are off
        if max_cutoff_devices is not None:
            mosfet_regions = sim_results.get('_mosfet_regions', {})
            if mosfet_regions:
                n_cutoff = sum(
                    1 for info in mosfet_regions.values()
                    if info.get('region', '') == 'cutoff'
                )
                if n_cutoff > max_cutoff_devices:
                    netlist_path.unlink(missing_ok=True)
                    return (sample_idx, None, 'filtered_saturation')

        # AC convergence filter — drop circuits where gain/phase extraction failed
        if require_ac_convergence and ac_template_path:
            ugbw = sim_results.get('_ac_ugbw')
            pm   = sim_results.get('_ac_pm')
            if ugbw is None or pm is None:
                netlist_path.unlink(missing_ok=True)
                return (sample_idx, None, 'filtered_saturation')

        # Minimum UGBW filter — drop circuits with effectively zero gain-bandwidth
        if min_ugbw_hz is not None and ac_template_path:
            ugbw = sim_results.get('_ac_ugbw')
            if ugbw is None or ugbw < min_ugbw_hz:
                netlist_path.unlink(missing_ok=True)
                return (sample_idx, None, 'filtered_saturation')

        # Build graph using ORIGINAL netlist (preserves graph structure)
        vin_p = params.get('VIN_P', params.get('VCM', 0.9) + params.get('VDIFF', 0) / 2)
        vin_n = params.get('VIN_N', params.get('VCM', 0.9) - params.get('VDIFF', 0) / 2)
        vcm = (vin_p + vin_n) / 2
        i_ref = params.get('I_BIAS', params.get('I_REF', 20e-6))

        mosfet_regions = sim_results.get('_mosfet_regions', {})

        # Prepare AC metrics and small-signal params for graph builder
        ac_metrics = None
        if ac_template_path:
            ac_metrics = {
                'ugbw': sim_results.get('_ac_ugbw'),
                'pm': sim_results.get('_ac_pm'),
                'am': sim_results.get('_ac_am'),
                'dc_gain': sim_results.get('_ac_dc_gain'),
            }
        mosfet_ss_params = sim_results.get('_mosfet_ss_params')

        graph_builder = CircuitGraphBuilder(
            normalize_props=normalize_props, param_specs=param_specs, vdd=vdd,
            device_to_group=device_to_group,
        )
        graph = graph_builder.build_from_netlist(
            str(netlist_path),
            {'node_voltages': node_voltages, 'device_currents': device_currents},
            vdd=vdd,
            vcm=vcm,
            vin_p=vin_p,
            vin_n=vin_n,
            i_ref=i_ref,
            mosfet_regions=mosfet_regions,
            ac_metrics=ac_metrics,
            mosfet_ss_params=mosfet_ss_params,
            excluded_current_devices=excluded_current_devices,
        )

        netlist_path.unlink(missing_ok=True)

        return (sample_idx, {
            'graph': graph,
            'params': params,
            'specs': sim_results,
            'sample_id': sample_idx,
            'vdd': vdd,
        }, 'ok')

    except Exception as e:
        try:
            Path(output_dir).joinpath(f'temp_netlist_{sample_idx}.sp').unlink(missing_ok=True)
            Path(output_dir).joinpath(f'temp_ac_netlist_{sample_idx}.sp').unlink(missing_ok=True)
        except:
            pass
        return (sample_idx, None, f'failed: {str(e)[:50]}')
