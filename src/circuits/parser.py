"""
SPICE Netlist Parser for Circuit Analysis

This module parses SPICE netlists into structured Python objects.
It extracts components (MOSFETs, resistors, etc.) with their
terminals, net connections, and parameter values.

SPICE Netlist Format:
- Lines starting with M or X: MOSFETs/subcircuits
- Lines starting with R: Resistors
- Lines starting with C: Capacitors
- Lines starting with V: Voltage sources
- Lines starting with I: Current sources
- Lines starting with * or .: Comments/directives

Example MOSFET line:
    M1 drain gate source bulk nmos_model W=10u L=1u
    
    Parsed as:
    - name: M1
    - type: mosfet
    - terminals: [drain, gate, source, bulk]
    - nets: [actual net names from netlist]
    - params: {W: 10e-6, L: 1e-6}
    - model: nmos_model
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Component:
    """
    Represents a circuit component (MOSFET, resistor, etc.).
    
    Attributes:
        name: Component instance name (e.g., 'M1', 'R_load')
        type: Component type ('mosfet', 'resistor', 'capacitor', etc.)
        terminals: List of terminal names (e.g., ['drain', 'gate', 'source', 'bulk'])
        nets: List of connected net names in same order as terminals
        params: Dict of parameter values (e.g., {'W': 10e-6, 'L': 1e-6})
        model: Optional model name (e.g., 'nmos_33')
    """
    name: str
    type: str
    terminals: List[str]
    nets: List[str]
    params: Dict[str, float]
    model: Optional[str] = None


@dataclass
class Terminal:
    """
    Represents a device terminal (becomes a graph node).
    
    Each component has multiple terminals (e.g., MOSFET has 4: D, G, S, B).
    Each terminal connects to exactly one net.
    
    Attributes:
        name: Unique terminal name (e.g., 'M1_drain')
        device_name: Parent component name (e.g., 'M1')
        device_type: Component type ('mosfet', 'resistor', etc.)
        terminal_type: Terminal role ('drain', 'gate', 'p', 'n', etc.)
        net: Connected net name
        device_params: Parameters from parent component
        device_model: Model name from parent component
    """
    name: str
    device_name: str
    device_type: str
    terminal_type: str
    net: str
    device_params: Dict[str, float]
    device_model: Optional[str] = None


@dataclass
class NetNode:
    """
    Represents an electrical net in the circuit (becomes a graph node).
    
    A net is an electrical connection point where multiple terminals meet.
    Examples: VDD, GND, internal signal nodes.
    
    Attributes:
        name: Net name from netlist (e.g., 'vdd', 'out', 'net1')
        voltage: DC operating point voltage (from simulation)
        net_type: Classification ('gnd', 'vdd', 'input', 'output', 'internal')
    """
    name: str
    voltage: float
    net_type: str  # 'gnd', 'vdd', 'input', 'output', 'internal'


class SPICENetlistParser:
    """
    Parse SPICE netlist file into list of Component objects.
    
    Supports common component types:
    - MOSFETs (M or X prefix)
    - Resistors (R prefix)
    - Capacitors (C prefix)
    - Voltage sources (V prefix)
    - Current sources (I prefix)
    
    Usage:
        parser = SPICENetlistParser()
        components = parser.parse_file('circuit.sp')
        all_nets = parser.all_nets
    """
    
    def __init__(self):
        self.components = []
        self.models = {}
        self.all_nets = set()  # Track all net names encountered
        
    def parse_file(self, filepath: str) -> List[Component]:
        """
        Parse SPICE netlist file.
        
        Args:
            filepath: Path to .sp or .spice file
        
        Returns:
            List of Component objects
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, comments, and directives
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            self._parse_line(line)
        
        return self.components
    
    def _parse_line(self, line: str):
        """Parse a single netlist line."""
        parts = line.split()
        if not parts:
            return
        
        name = parts[0]
        component_type = name[0].upper()
        
        if component_type in ('M', 'X'):
            self._parse_mosfet(parts, name)
        elif component_type == 'R':
            self._parse_resistor(parts, name)
        elif component_type == 'C':
            self._parse_capacitor(parts, name)
        elif component_type == 'V':
            self._parse_voltage_source(parts, name)
        elif component_type == 'I':
            self._parse_current_source(parts, name)
    
    def _parse_mosfet(self, parts: List[str], name: str):
        """Parse MOSFET/subcircuit line."""
        if len(parts) < 5:
            return
        terminals = ['drain', 'gate', 'source', 'bulk']
        nets = parts[1:5]
        model = parts[5] if len(parts) > 5 else None
        params = self._extract_params(parts[6:])
        
        self.components.append(Component(
            name=name, type='mosfet', terminals=terminals, nets=nets, params=params, model=model
        ))
        self.all_nets.update(nets)
    
    def _parse_resistor(self, parts: List[str], name: str):
        """Parse resistor line."""
        if len(parts) < 4:
            return
        terminals = ['p', 'n']
        nets = parts[1:3]
        value = self._parse_value(parts[3])
        
        self.components.append(Component(
            name=name, type='resistor', terminals=terminals, nets=nets, params={'value': value}
        ))
        self.all_nets.update(nets)
    
    def _parse_capacitor(self, parts: List[str], name: str):
        """Parse capacitor line."""
        if len(parts) < 4:
            return
        terminals = ['p', 'n']
        nets = parts[1:3]
        value = self._parse_value(parts[3])
        
        self.components.append(Component(
            name=name, type='capacitor', terminals=terminals, nets=nets, params={'value': value}
        ))
        self.all_nets.update(nets)
    
    def _parse_voltage_source(self, parts: List[str], name: str):
        """Parse voltage source line."""
        if len(parts) < 3:
            return
        terminals = ['p', 'n']
        nets = parts[1:3]
        
        value = 0.0
        if len(parts) > 3:
            if parts[3].upper() == 'DC' and len(parts) > 4:
                try:
                    value = self._parse_value(parts[4])
                except (ValueError, IndexError):
                    value = 0.0
            elif not parts[3].upper().startswith('PULSE'):
                try:
                    value = self._parse_value(parts[3])
                except ValueError:
                    value = 0.0
        
        self.components.append(Component(
            name=name, type='voltage_source', terminals=terminals, nets=nets, params={'value': value}
        ))
        self.all_nets.update(nets)
    
    def _parse_current_source(self, parts: List[str], name: str):
        """Parse current source line."""
        if len(parts) < 3:
            return
        terminals = ['p', 'n']
        nets = parts[1:3]
        
        value = 0.0
        if len(parts) > 3:
            if parts[3].upper() == 'DC' and len(parts) > 4:
                try:
                    value = self._parse_value(parts[4])
                except (ValueError, IndexError):
                    value = 0.0
            else:
                try:
                    value = self._parse_value(parts[3])
                except ValueError:
                    value = 0.0
        
        self.components.append(Component(
            name=name, type='current_source', terminals=terminals, nets=nets, params={'value': value}
        ))
        self.all_nets.update(nets)
    
    def _extract_params(self, param_list: List[str]) -> Dict[str, float]:
        """Extract parameters like W=1u L=0.18u."""
        params = {}
        for param in param_list:
            if '=' in param:
                key, val = param.split('=')
                params[key.lower()] = self._parse_value(val)
        return params
    
    def _parse_value(self, value_str: str) -> float:
        """Parse SPICE value with multipliers (1u, 10n, etc.).

        Also handles single-quoted expressions like '4*2' -> 8.
        """
        # Strip surrounding quotes (SPICE expression syntax)
        value_str = value_str.strip("'\"")

        # Handle multiplication expressions like '4*2' or '8*10u'
        if '*' in value_str:
            result = 1.0
            for part in value_str.split('*'):
                result *= self._parse_value(part.strip())
            return result

        value_str = value_str.lower().replace('meg', 'e6')

        multipliers = {
            'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
            'm': 1e-3, 'k': 1e3, 'g': 1e9, 't': 1e12
        }

        for suffix, mult in multipliers.items():
            if value_str.endswith(suffix):
                return float(value_str[:-1]) * mult

        return float(value_str)
