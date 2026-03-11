"""
Layer building utilities for GNN models.

This module provides factory functions for creating common layer types
used in deep GNN architectures.
"""

import torch.nn as nn
import torch_geometric.nn as PyGnn


def build_mlp(
    num_layers: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    norm_type: str = 'layer',
    dropout: float = 0.0,
) -> nn.Module:
    """
    Build an MLP for prediction heads.

    Args:
        num_layers: Number of layers (1 = linear only)
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        norm_type: 'layer' or 'batch' normalization
        dropout: Dropout probability

    Returns:
        nn.Module: MLP as Sequential or Linear
    """
    if num_layers == 1:
        return nn.Linear(input_dim, output_dim)

    def make_norm():
        if norm_type == 'batch':
            return nn.BatchNorm1d(hidden_dim)
        return nn.LayerNorm(hidden_dim)

    layers = []

    # First layer
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(make_norm())
    layers.append(nn.ReLU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Middle layers
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(make_norm())
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Output layer
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


def create_deepgcn_layer(
    hidden_dim: int,
    genconv_num_layers: int = 2,
    norm_type: str = 'layer',
    dropout: float = 0.0,
    edge_dim: int = None,
) -> PyGnn.DeepGCNLayer:
    """
    Create a DeepGCNLayer with GENConv.

    Args:
        hidden_dim: Hidden dimension
        genconv_num_layers: Number of MLP layers in GENConv
        norm_type: 'layer' or 'batch' normalization
        dropout: Dropout probability
        edge_dim: Edge feature dimension (None = no edge features)

    Returns:
        DeepGCNLayer with res+ block
    """
    conv_kwargs = dict(
        aggr='softmax',
        t=1.0,
        learn_t=True,
        num_layers=genconv_num_layers,
    )
    if edge_dim is not None:
        conv_kwargs['edge_dim'] = edge_dim
    conv = PyGnn.GENConv(hidden_dim, hidden_dim, **conv_kwargs)

    if norm_type == 'batch':
        norm = nn.BatchNorm1d(hidden_dim)
    else:
        norm = nn.LayerNorm(hidden_dim)

    act = nn.ReLU()

    return PyGnn.DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)
