"""
GNN architectures for circuit node voltage/current prediction.

All architectures are automatically registered via the @register_model decorator.
Import this module to register all available architectures.

Virtual Node is a configuration option on DeepGENConv, not a separate model.
Use `use_virtual_node=True` to enable it.
"""

from .base import BaseGNN
from .deepgen import DeepGENConv
from .tower_genconv import TowerGENConv

__all__ = [
    'BaseGNN',
    'DeepGENConv',
    'TowerGENConv',
]
