"""
Base class for all GNN architectures.

Provides a common interface and shared functionality for GNN models
used in circuit node voltage/current prediction.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseGNN(nn.Module, ABC):
    """
    Abstract base class for GNN architectures.

    All GNN models should inherit from this class and implement the forward method.

    The forward method must return a dict with:
    - 'node_voltages': Tensor of predicted node voltages
    - 'node_currents' (optional): Tensor of predicted node currents
    """

    def __init__(self):
        super().__init__()
        self._config = {}

    @abstractmethod
    def forward(self, data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            data: PyG Data object with x, edge_index, and optional type_tens/net_type/batch

        Returns:
            Dict with 'node_voltages' and optionally 'node_currents'
        """
        pass

    def get_config(self) -> dict:
        """
        Get model configuration for checkpointing.

        Returns:
            Dict containing model configuration
        """
        return self._config.copy()

    def set_config(self, config: dict):
        """
        Set model configuration (used when loading from checkpoint).

        Args:
            config: Configuration dict
        """
        self._config = config.copy()

    def _get_input_features(self, data) -> torch.Tensor:
        """
        Get input features by concatenating available feature tensors.

        Args:
            data: PyG Data object

        Returns:
            Concatenated feature tensor
        """
        x = data.x
        if hasattr(data, 'type_tens') and data.type_tens is not None:
            x = torch.cat([x, data.type_tens], dim=-1)
        if hasattr(data, 'net_type') and data.net_type is not None:
            x = torch.cat([x, data.net_type], dim=-1)
        return x
