"""
Graph Neural Network Package for Circuit Node Voltage Prediction.

This package provides GNN architectures for predicting DC operating point
voltages and currents in analog circuits from graph representations.

Subpackages:
    - architectures: GNN model implementations (DeepGENConv, etc.)
    - components: Reusable building blocks (layers, aggregation, virtual_node)

Registry:
    - register_model: Decorator to register new architectures
    - get_model: Factory function to instantiate models by name
    - list_models: List all registered model names

Models:
    - DeepGENConv: Deep GNN with GENConv layers (supports optional Virtual Node)

Virtual Node:
    Virtual Node is a configuration option, not a separate model.
    Enable it with `use_virtual_node=True` in model config.
"""

# Import registry functions
from .registry import (
    MODEL_REGISTRY,
    register_model,
    get_model,
    get_model_class,
    list_models,
)

# Import architectures (this also registers them)
from .architectures import (
    BaseGNN,
    DeepGENConv,
)

# Import components for direct access
from .components import (
    build_mlp,
    create_deepgcn_layer,
    JumpingKnowledge,
    VirtualNode,
)

__all__ = [
    # Registry
    'MODEL_REGISTRY',
    'register_model',
    'get_model',
    'get_model_class',
    'list_models',
    # Base class
    'BaseGNN',
    # Models
    'DeepGENConv',
    # Components
    'build_mlp',
    'create_deepgcn_layer',
    'JumpingKnowledge',
    'VirtualNode',
]
