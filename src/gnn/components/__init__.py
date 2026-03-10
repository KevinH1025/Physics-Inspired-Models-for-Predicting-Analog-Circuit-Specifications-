"""
Reusable GNN components.

This module provides building blocks for constructing GNN architectures:
- layers: MLP and DeepGCNLayer factories
- aggregation: Jumping Knowledge aggregation
- virtual_node: Virtual node module for global information flow
- current_from_voltage: Physics-informed current prediction from voltages
"""

from .layers import build_mlp, create_deepgcn_layer
from .aggregation import JumpingKnowledge
from .virtual_node import VirtualNode
from .current_from_voltage import MOSFETCurrentMLP, compute_mosfet_currents

__all__ = [
    'build_mlp',
    'create_deepgcn_layer',
    'JumpingKnowledge',
    'VirtualNode',
    'MOSFETCurrentMLP',
    'compute_mosfet_currents',
]
