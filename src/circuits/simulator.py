"""
Circuit Simulation using NgSpice

This module provides a wrapper around PySpice/NgSpice for running
SPICE simulations and extracting circuit specifications.

NgSpice is an open-source SPICE simulator. We use the shared library
interface (NgSpice Shared) for programmatic access.

Simulation Flow:
1. Load netlist into NgSpice
2. Run DC operating point analysis
3. Extract all node voltages and device currents
4. Compute derived specifications (gain, power, etc.)

The key output is '_all_net_voltages': a dict mapping net names to
their DC operating point voltages. This is used to create voltage
targets for the GNN.

Example:
    simulator = CircuitSimulator(input_node='in', output_node='out')
    specs = simulator.simulate(netlist_string)
    voltages = specs['_all_net_voltages']  # {'vdd': 1.8, 'out': 0.9, ...}
"""

import os
import sys
import logging
import numpy as np
from math import log10, pi, atan, degrees, sqrt
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

# Suppress verbose NgSpice logging
logging.getLogger('PySpice.Spice.NgSpice.Shared').setLevel(logging.ERROR)
os.environ['PYSPICE_LIBRARY_PATH'] = '/usr/local/lib'

from PySpice.Spice.NgSpice.Shared import NgSpiceShared


def _find_crossing_down(x: np.ndarray, y: np.ndarray, target: float) -> Optional[float]:
    """
    Find the x-value where y crosses target going downward (from above to below).

    Uses linear interpolation between adjacent points.

    Args:
        x: x-axis values (e.g., frequency), monotonically increasing
        y: y-axis values (e.g., magnitude in dB or phase in degrees)
        target: the y-value to find the crossing for

    Returns:
        Interpolated x-value at crossing, or None if no crossing found
    """
    y_shifted = y - target
    for i in range(len(y_shifted) - 1):
        if y_shifted[i] >= 0 and y_shifted[i + 1] < 0:
            # Linear interpolation
            frac = y_shifted[i] / (y_shifted[i] - y_shifted[i + 1])
            return float(x[i] + frac * (x[i + 1] - x[i]))
    return None


@contextmanager
def suppress_stdout_stderr():
    """
    Suppress stdout/stderr at C-level during NgSpice simulation.
    
    Uses os.dup2 to redirect file descriptors, which catches C-level output
    from NgSpice shared library that Python-level redirection misses.
    """
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    try:
        # Redirect to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


class CircuitSimulator:
    """
    Run SPICE simulations and extract specifications.
    
    Wraps NgSpice shared library for DC operating point analysis.
    Extracts node voltages and device currents from simulation results.
    
    Attributes:
        input_node: Name of input signal node
        output_node: Name of output node
        supply_node: Name of supply voltage node
        input_node_neg: Optional negative input for differential circuits
        analysis_types: List of analyses to run (default: ['dc'])
    """
    
    def __init__(self, input_node: str = 'in', output_node: str = 'out',
                 supply_node: str = 'vdd', input_node_neg: Optional[str] = None,
                 analysis_types: Optional[List[str]] = None):
        self.input_node = input_node
        self.output_node = output_node
        self.supply_node = supply_node
        self.input_node_neg = input_node_neg
        self.analysis_types = analysis_types or ['dc']
    
    def simulate(self, netlist: str) -> Dict[str, float]:
        """
        Run simulation and extract all specifications.
        
        Args:
            netlist: Complete SPICE netlist as string
        
        Returns:
            Dict with:
            - 'dc_output_v': Output node DC voltage
            - 'dc_supply_v': Supply voltage
            - '_all_net_voltages': Dict of all net voltages (for GNN targets)
            - '_all_device_currents': Dict of all device currents
        """
        specs = {}
        ngspice = None
        try:
            # Ensure .save all directive is present
            if '.save all' not in netlist:
                netlist = netlist.replace('.end', '.save all\n.end')
            
            # Create new NgSpice instance (isolated simulation)
            # Suppress verbose NgSpice output (converge messages, etc.)
            with suppress_stdout_stderr():
                ngspice = NgSpiceShared.new_instance()
                ngspice.load_circuit(netlist)
                
                try:
                    ngspice.run()
                except Exception:
                    pass  # Some simulations fail but still produce valid DC results
            
            # Extract DC operating point (also suppress any output)
            if 'dc' in self.analysis_types:
                with suppress_stdout_stderr():
                    specs = self._extract_dc_specs(ngspice)

            # Extract AC frequency response (loop gain)
            if 'ac' in self.analysis_types:
                with suppress_stdout_stderr():
                    ac_specs = self._extract_ac_specs(ngspice)
                    specs.update(ac_specs)

            # Extract small-signal parameters (gm, gds) from .op results
            if 'dc' in self.analysis_types:
                with suppress_stdout_stderr():
                    ss_params = self._extract_smallsignal_params(ngspice)
                    if ss_params:
                        specs['_mosfet_ss_params'] = ss_params

        except Exception as e:
            print(f"Simulation failed: {e}")
            specs = self._get_default_specs()
        finally:
            # Clean up NgSpice instance
            if ngspice is not None:
                try:
                    ngspice.destroy()
                except Exception:
                    pass
        
        return specs
    
    def _extract_dc_specs(self, ngspice: NgSpiceShared) -> Dict[str, float]:
        """
        Extract DC operating point specifications.
        
        Parses NgSpice output to extract:
        - All node voltages (for GNN targets)
        - All device currents (for power computation)
        - Key specifications (output voltage, supply current, etc.)
        """
        specs = {}
        all_net_voltages = {}
        all_device_currents = {}
        
        try:
            # Get operating point results
            plot_op = ngspice.plot('op', 'op1')
            available_vectors = list(plot_op.keys())
            
            for vector_name in available_vectors:
                if vector_name == 'time':
                    continue
                
                try:
                    # Extract scalar value from waveform
                    value = float(plot_op[vector_name].to_waveform()[0])
                    
                    # Classify as voltage or current
                    is_current = False
                    device_name = None
                    vector_lower = vector_name.lower()
                    
                    # Current vectors from .save commands:
                    # 1. @m.xm1.msky130_fd_pr__nfet_01v8[id] - MOSFET drain current
                    # 2. i(rin), i(rfb), i(rz) - resistor currents (NgSpice compatible)
                    # 3. @iref[i] - current source current
                    # 4. device#branch - voltage source currents
                    if vector_lower.startswith('@m.') and '[id]' in vector_lower:
                        parts = vector_name.split('.')
                        if len(parts) >= 2:
                            device_name = parts[1]
                            is_current = True
                    elif vector_lower.startswith('i(') and vector_lower.endswith(')'):
                        # i(device) format for resistor currents
                        device_name = vector_name[2:-1]  # Extract device name from i(device)
                        is_current = True
                    elif vector_lower.startswith('@') and '[i]' in vector_lower:
                        # @device[i] format for current sources
                        device_name = vector_name[1:].split('[')[0]
                        is_current = True
                    elif '#branch' in vector_lower:
                        device_name = vector_name.split('#')[0]
                        is_current = True
                    
                    if is_current:
                        all_device_currents[device_name] = value
                    else:
                        all_net_voltages[vector_name] = value
                except (ValueError, IndexError, AttributeError):
                    pass
            
            # Extract key specifications
            v_out_dc = self._get_node_value(self.output_node, all_net_voltages) or 0.0
            v_supply = self._get_node_value(self.supply_node, all_net_voltages) or 0.0
            
            specs['dc_output_v'] = v_out_dc
            specs['dc_supply_v'] = v_supply
            specs['_all_net_voltages'] = all_net_voltages  # CRITICAL: Used for GNN targets
            specs['_all_device_currents'] = all_device_currents

            # Extract MOSFET operating regions for saturation filtering
            mosfet_regions = self._extract_mosfet_regions(ngspice, available_vectors)
            specs['_mosfet_regions'] = mosfet_regions
            
            # Compute supply current
            supply_current_keys = [k for k in all_device_currents.keys()
                                   if k.upper() in ['VDD', 'VSS', 'VDDA', 'GNDA',
                                                     self.supply_node.upper()]]
            if supply_current_keys:
                specs['dc_supply_current'] = all_device_currents[supply_current_keys[0]]
            
            if self.input_node_neg:
                v_in_p = self._get_node_value(self.input_node, all_net_voltages) or 0.0
                v_in_n = self._get_node_value(self.input_node_neg, all_net_voltages) or 0.0
                specs['dc_input_p_v'] = v_in_p
                specs['dc_input_n_v'] = v_in_n
                specs['dc_input_offset_v'] = v_in_p - v_in_n
                
        except Exception:
            # Simulation didn't converge or no vectors available - return defaults silently
            specs = self._get_default_specs()
        
        return specs
    
    def _get_node_value(self, node_name: str, node_dict: Dict[str, float], default: float = None) -> Optional[float]:
        """Get node value with case-insensitive matching."""
        if node_name in node_dict:
            return node_dict[node_name]
        node_name_upper = node_name.upper()
        for key, value in node_dict.items():
            if key.upper() == node_name_upper:
                return value
        return default
    
    def _get_default_specs(self) -> Dict[str, float]:
        """Return default specs when simulation fails."""
        specs = {
            'dc_output_v': 0.0,
            'dc_supply_v': 0.0,
            '_all_net_voltages': {},
            '_all_device_currents': {},
            '_mosfet_regions': {}
        }
        
        if self.input_node_neg:
            specs.update({
                'dc_input_p_v': 0.0,
                'dc_input_n_v': 0.0,
                'dc_input_offset_v': 0.0,
            })

        return specs

    def _extract_mosfet_regions(self, ngspice, available_vectors) -> Dict[str, Dict]:
        """
        Extract MOSFET operating region information for saturation filtering.

        For each MOSFET, extracts vgs, vds, vdsat and determines operating region:
        - saturation: |Vds| >= |Vdsat| and device is on
        - triode: |Vds| < |Vdsat| and device is on
        - cutoff: device is off (|Vgs| < |Vth|)

        Returns dict mapping device name to {'vgs', 'vds', 'vdsat', 'region', 'is_pmos'}
        """
        mosfet_regions = {}

        # Find MOSFET device names from current vectors (format: @m.xm1.msky130....[id])
        mosfet_devices = set()
        for vec in available_vectors:
            vec_lower = vec.lower()
            if vec_lower.startswith('@m.') and '[id]' in vec_lower:
                # Extract device instance name (e.g., xm1 from @m.xm1.msky130_fd_pr__nfet_01v8[id])
                parts = vec.split('.')
                if len(parts) >= 3:
                    instance = parts[1]  # xm1, xm2, etc.
                    model_part = parts[2].split('[')[0]  # msky130_fd_pr__nfet_01v8
                    is_pmos = 'pfet' in model_part.lower()
                    mosfet_devices.add((instance, model_part, is_pmos))

        # Extract vgs, vds, vdsat for each MOSFET
        try:
            plot_op = ngspice.plot('op', 'op1')

            for instance, model_part, is_pmos in mosfet_devices:
                device_info = {'is_pmos': is_pmos, 'region': 'unknown'}

                # Construct parameter vector names
                base = f'@m.{instance}.{model_part}'
                vgs_key = f'{base}[vgs]'
                vds_key = f'{base}[vds]'
                vdsat_key = f'{base}[vdsat]'
                vth_key = f'{base}[vth]'

                # Try to get values (case-insensitive search)
                vgs = self._get_vector_value(plot_op, vgs_key)
                vds = self._get_vector_value(plot_op, vds_key)
                vdsat = self._get_vector_value(plot_op, vdsat_key)
                vth = self._get_vector_value(plot_op, vth_key)

                device_info['vgs'] = vgs
                device_info['vds'] = vds
                device_info['vdsat'] = vdsat
                device_info['vth'] = vth

                # Determine operating region
                if vgs is not None and vds is not None and vdsat is not None:
                    # For PMOS, voltages are typically negative in SPICE
                    # Use absolute values for comparison
                    abs_vds = abs(vds)
                    abs_vdsat = abs(vdsat)
                    abs_vgs = abs(vgs)
                    abs_vth = abs(vth) if vth is not None else 0.4  # Default Vth

                    # Check if device is on (|Vgs| > |Vth|)
                    if abs_vgs < abs_vth * 0.8:  # 80% margin for near-cutoff
                        device_info['region'] = 'cutoff'
                    elif abs_vds >= abs_vdsat - 0.01:  # 10mV margin for numerical tolerance
                        device_info['region'] = 'saturation'
                    else:
                        device_info['region'] = 'triode'

                mosfet_regions[instance] = device_info

        except Exception:
            # If extraction fails, return empty dict (filter will be skipped)
            pass

        return mosfet_regions

    @staticmethod
    def _get_raw_value(vec) -> Optional[float]:
        """Extract scalar float from a NgSpice Vector, handling unit conversion failures."""
        try:
            return float(vec.to_waveform()[0])
        except (AttributeError, TypeError, IndexError):
            pass
        # Fallback: access raw numpy data directly (needed for gm, gds, etc.)
        try:
            return float(vec._data[0])
        except (AttributeError, TypeError, IndexError):
            pass
        return None

    def _get_vector_value(self, plot_op, vector_name: str) -> Optional[float]:
        """Get a vector value from the plot with case-insensitive matching."""
        # Direct match
        if vector_name in plot_op:
            val = self._get_raw_value(plot_op[vector_name])
            if val is not None:
                return val

        # Case-insensitive search
        vector_lower = vector_name.lower()
        for key in plot_op.keys():
            if key.lower() == vector_lower:
                val = self._get_raw_value(plot_op[key])
                if val is not None:
                    return val
        return None

    def _extract_ac_specs(self, ngspice) -> Dict:
        """
        Extract AC frequency response and compute loop gain metrics.

        Uses Middlebrook method: loop gain T(f) = -V(vp_fb) / V(vp_gate)
        where vp_fb is the feedback side and vp_gate is the amplifier side
        of the broken loop at the inverting input.

        Returns dict with:
        - '_ac_ugbw': Unity-gain bandwidth in Hz
        - '_ac_pm': Phase margin in degrees
        - '_ac_am': Gain margin in dB (positive = stable)
        - '_ac_dc_gain': DC gain in dB
        """
        ac_specs = {
            '_ac_ugbw': None,
            '_ac_pm': None,
            '_ac_am': None,
            '_ac_dc_gain': None,
        }

        try:
            # Try to get AC plot
            plot_ac = ngspice.plot('ac', 'ac1')
            if plot_ac is None:
                return ac_specs

            available = list(plot_ac.keys())

            # Get frequency vector
            freq_waveform = None
            for key in available:
                if key.lower() == 'frequency':
                    freq_waveform = plot_ac[key].to_waveform()
                    break
            if freq_waveform is None:
                return ac_specs
            freq = np.abs(np.array(freq_waveform))

            # Auto-detect Middlebrook loop-breaking nodes: *_fb and *_gate
            # 2-stage uses vp_fb/vp_gate, 3-stage uses vinn_fb/vinn_gate, etc.
            v_fb = None
            v_gate = None
            for key in available:
                kl = key.lower().replace('v(', '').replace(')', '')
                if kl.endswith('_fb'):
                    v_fb = np.array(plot_ac[key].to_waveform())
                elif kl.endswith('_gate'):
                    v_gate = np.array(plot_ac[key].to_waveform())

            if v_fb is None or v_gate is None:
                return ac_specs

            # Compute loop gain: T(f) = -V(vp_fb) / V(vp_gate)
            # Avoid division by zero
            v_gate_safe = np.where(np.abs(v_gate) < 1e-30, 1e-30, v_gate)
            T = -v_fb / v_gate_safe

            mag = np.abs(T)
            mag_db = 20 * np.log10(mag + 1e-30)
            phase_deg = np.degrees(np.unwrap(np.angle(T)))

            # DC gain: magnitude at lowest frequency
            ac_specs['_ac_dc_gain'] = float(mag_db[0])

            # UGBW: frequency where mag_db crosses 0 dB (from above to below)
            ugbw = _find_crossing_down(freq, mag_db, 0.0)
            ac_specs['_ac_ugbw'] = ugbw

            # PM: 180 + phase(T) at UGBW
            if ugbw is not None:
                pm = 180.0 + float(np.interp(ugbw, freq, phase_deg))
                ac_specs['_ac_pm'] = pm

            # AM: gain margin = -mag_db at frequency where phase crosses -180°
            f_180 = _find_crossing_down(freq, phase_deg, -180.0)
            if f_180 is not None:
                am = -float(np.interp(f_180, freq, mag_db))
                ac_specs['_ac_am'] = am

        except Exception as e:
            logging.debug(f"AC extraction failed: {e}")

        return ac_specs

    def _extract_smallsignal_params(self, ngspice) -> Dict[str, Dict]:
        """
        Extract small-signal parameters (gm, gds) from DC operating point.

        Returns dict mapping device instance name to {'gm': float, 'gds': float}.
        """
        ss_params = {}

        try:
            plot_op = ngspice.plot('op', 'op1')
            available = list(plot_op.keys())

            # Find MOSFET devices from available vectors
            mosfet_devices = set()
            for vec in available:
                vec_lower = vec.lower()
                if vec_lower.startswith('@m.') and '[id]' in vec_lower:
                    parts = vec.split('.')
                    if len(parts) >= 3:
                        instance = parts[1]
                        model_part = parts[2].split('[')[0]
                        mosfet_devices.add((instance, model_part))

            for instance, model_part in mosfet_devices:
                base = f'@m.{instance}.{model_part}'
                gm = self._get_vector_value(plot_op, f'{base}[gm]')
                gds = self._get_vector_value(plot_op, f'{base}[gds]')

                if gm is not None or gds is not None:
                    ss_params[instance] = {
                        'gm': gm if gm is not None else 0.0,
                        'gds': gds if gds is not None else 0.0,
                    }

        except Exception as e:
            logging.debug(f"Small-signal param extraction failed: {e}")

        return ss_params

    @staticmethod
    def map_ac_node_voltages(voltages: Dict[str, float]) -> Dict[str, float]:
        """
        Map AC template node names back to original template names.

        The AC template splits a feedback node (e.g. 'vp' or 'vinn') into
        '{node}_fb' and '{node}_gate' for Middlebrook loop-breaking.
        At DC these are identical (connected by inductor = DC short).
        Also filters out 'vac_node' (injection source node).
        """
        mapped = {}
        for name, value in voltages.items():
            nl = name.lower()
            if nl.endswith('_fb'):
                # Map back to original node name: vp_fb -> vp, vinn_fb -> vinn
                mapped[nl[:-3]] = value
            elif nl.endswith('_gate'):
                continue  # Same DC voltage as _fb, skip duplicate
            elif nl == 'vac_node':
                continue  # Skip injection source node
            else:
                mapped[name] = value
        return mapped

    def compute_analytical_ac(self, specs: Dict, cc: float, rz: float = 0.0,
                              r_in: float = 50e3, r_fb: float = 50e3,
                              cl: float = 10e-12, w_m6: float = 50e-6) -> Dict:
        """
        Compute analytical loop-gain AC metrics from DC operating point.

        Computes T(s) = A_OL(s) * beta to match the SPICE Middlebrook
        measurement (loop gain, not open-loop gain).

        Key insight: in this topology, Cc is interstage coupling (vout_stage1
        to vg2), NOT Miller feedback from vout. The actual compensation comes
        from M6's intrinsic Cgd (gate-drain overlap cap) which gets Miller-
        multiplied by stage 2 gain.

        Accounts for:
        1. Feedback factor beta = r_in / (r_in + r_fb)
        2. Output loading by feedback network: R2_eff = R2 || (r_in + r_fb)
        3. Miller compensation from Cgd6: C_Miller = Cgd6 * (1 + gm6*R2_eff)
        4. Rz-Cc zero: z1 = 1/(Cc*(1/gm6 - Rz))
        5. Full pole-zero phase margin calculation

        Args:
            specs: Simulation results dict (must contain '_mosfet_ss_params')
            cc: Compensation capacitor value (F)
            rz: Nulling resistor value (ohms)
            r_in: Input resistor of inverting amplifier (ohms)
            r_fb: Feedback resistor of inverting amplifier (ohms)
            cl: Load capacitor value (F, default 10pF)
            w_m6: Width of M6 in meters (for Cgd estimation)

        Returns:
            Dict with '_analytical_dc_gain' (dB), '_analytical_ugbw' (Hz),
            '_analytical_pm' (degrees)
        """
        analytical = {
            '_analytical_dc_gain': None,
            '_analytical_ugbw': None,
            '_analytical_pm': None,
        }

        ss = specs.get('_mosfet_ss_params', {})
        if not ss:
            return analytical

        # This analytical model is specific to the 2-stage opamp topology.
        # Guard: return empty if the required devices aren't present.
        required_devices = ['xm1', 'xm2', 'xm4', 'xm6', 'xm7']
        if not all(d in ss for d in required_devices):
            return analytical

        try:
            gm1 = abs(ss.get('xm1', {}).get('gm', 0))
            gm6 = abs(ss.get('xm6', {}).get('gm', 0))
            gds2 = ss.get('xm2', {}).get('gds', 1e-15)
            gds4 = ss.get('xm4', {}).get('gds', 1e-15)
            gds6 = ss.get('xm6', {}).get('gds', 1e-15)
            gds7 = ss.get('xm7', {}).get('gds', 1e-15)

            if gm1 == 0 or gm6 == 0:
                return analytical

            # --- Stage resistances ---
            R1 = 1.0 / (gds2 + gds4) if (gds2 + gds4) > 0 else 1e15
            R2 = 1.0 / (gds6 + gds7) if (gds6 + gds7) > 0 else 1e15

            # Feedback network loads the output
            R_fb_total = r_in + r_fb
            R2_eff = (R2 * R_fb_total) / (R2 + R_fb_total) if (R2 + R_fb_total) > 0 else R2

            # --- DC loop gain ---
            beta = r_in / (r_in + r_fb) if (r_in + r_fb) > 0 else 1.0
            A_v0 = gm1 * R1 * gm6 * R2_eff
            T_0 = A_v0 * beta
            analytical['_analytical_dc_gain'] = 20 * log10(T_0 + 1e-30)

            # --- Cgd6 estimation from device width ---
            # SKY130 PMOS gate-drain overlap cap ≈ 0.25 fF/um
            CGDO_PMOS = 0.25e-15  # F/um
            Cgd6 = CGDO_PMOS * (w_m6 * 1e6)  # convert m to um

            # Miller multiplication: C_Miller = Cgd6 * (1 + gm6*R2_eff)
            C_Miller = Cgd6 * (1.0 + gm6 * R2_eff)

            # --- Poles and zeros (angular frequency, rad/s) ---
            # Dominant pole from Miller-reflected Cgd6 at vout_stage1
            wp1 = 1.0 / (R1 * C_Miller)
            # Non-dominant pole at output
            wp2 = gm6 / cl if cl > 0 else 1e15

            # Zero from Cc-Rz: z1 = 1 / (Cc * (1/gm6 - Rz))
            inv_gm6 = 1.0 / gm6
            rz_diff = inv_gm6 - rz
            if abs(rz_diff) > 1e-15:
                wz1 = abs(1.0 / (cc * rz_diff))
                z1_is_rhp = (rz_diff > 0)  # RHP when 1/gm6 > Rz
            else:
                wz1 = 1e15
                z1_is_rhp = False

            # --- UGBW via iterative solver ---
            # |T(jw)| = (gm1*beta)/(w*C_Miller) * sqrt(1+(w/wz1)^2) / sqrt(1+(w/wp2)^2)
            wu = gm1 * beta / C_Miller  # initial guess (angular freq)
            for _ in range(50):
                T_base = gm1 * beta / (wu * C_Miller)
                zero_factor = sqrt(1.0 + (wu / wz1) ** 2)
                pole2_factor = sqrt(1.0 + (wu / wp2) ** 2)
                T_mag = T_base * zero_factor / pole2_factor
                if abs(T_mag - 1.0) < 0.001:
                    break
                wu *= T_mag ** 0.7  # damped scaling

            analytical['_analytical_ugbw'] = wu / (2 * pi)

            # --- Phase margin ---
            phase_p1 = -degrees(atan(wu / wp1))
            phase_p2 = -degrees(atan(wu / wp2))
            if wz1 < 1e14:
                phase_z1 = -degrees(atan(wu / wz1)) if z1_is_rhp else degrees(atan(wu / wz1))
            else:
                phase_z1 = 0.0
            analytical['_analytical_pm'] = 180.0 + phase_p1 + phase_p2 + phase_z1

        except Exception as e:
            logging.debug(f"Analytical AC computation failed: {e}")

        return analytical
