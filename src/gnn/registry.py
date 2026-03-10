"""
Model registry for GNN architectures.

Provides decorator-based registration and factory functions for model instantiation.
"""

from typing import Dict, Type, List, Any

MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the registry.

    Usage:
        @register_model("my_model")
        class MyModel(nn.Module):
            ...

    Args:
        name: Unique identifier for the model

    Returns:
        Decorator function
    """
    def decorator(cls: Type) -> Type:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> Any:
    """
    Instantiate a model by its registered name.

    Args:
        name: Registered model name
        **kwargs: Arguments to pass to model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name is not registered
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> List[str]:
    """
    List all registered model names.

    Returns:
        List of registered model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_class(name: str) -> Type:
    """
    Get the model class without instantiating.

    Args:
        name: Registered model name

    Returns:
        Model class

    Raises:
        ValueError: If model name is not registered
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name]
