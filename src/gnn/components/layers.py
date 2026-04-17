"""
Layer building utilities for GNN models.

This module provides factory functions for creating common layer types
used in deep GNN architectures.
"""

import torch
import torch.nn as nn
import torch_geometric.nn as PyGnn


class GINWithResidualMLP(nn.Module):
    """GINConv with residual connection around the MLP."""

    def __init__(self, hidden_dim, act_type, expansion=2, mlp_depth=2):
        super().__init__()
        # GIN with identity MLP (just passes through)
        self.conv = PyGnn.GINConv(nn.Identity(), train_eps=True)
        # Separate MLP with residual
        if mlp_depth == 3:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * expansion),
                nn.BatchNorm1d(hidden_dim * expansion),
                get_activation(act_type),
                nn.Linear(hidden_dim * expansion, hidden_dim * expansion),
                nn.BatchNorm1d(hidden_dim * expansion),
                get_activation(act_type),
                nn.Linear(hidden_dim * expansion, hidden_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * expansion),
                nn.BatchNorm1d(hidden_dim * expansion),
                get_activation(act_type),
                nn.Linear(hidden_dim * expansion, hidden_dim),
            )

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index)  # sum + (1+eps)*self
        h = self.mlp(h)
        return h


class GATv2WithFFN(nn.Module):
    """GATv2Conv + FFN in one module, so FFN is inside the DeepGCNLayer residual."""

    def __init__(self, hidden_dim, num_heads, dropout, edge_dim, act_type, share_weights=False):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.conv = PyGnn.GATv2Conv(
            hidden_dim, head_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            edge_dim=edge_dim,
            share_weights=share_weights,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            get_activation(act_type),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index, edge_attr)
        h = h + self.ffn(h)  # residual around FFN
        return h


def get_activation(act_type: str = 'relu') -> nn.Module:
    """Return activation module by name. Default: ReLU."""
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'silu':
        return nn.SiLU()
    return nn.ReLU()


def build_mlp(
    num_layers: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    norm_type: str = 'layer',
    dropout: float = 0.0,
    act_type: str = 'relu',
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
    layers.append(get_activation(act_type))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Middle layers
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(make_norm())
        layers.append(get_activation(act_type))
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
    act_type: str = 'relu',
    conv_type: str = 'genconv',
    num_heads: int = 4,
    **kwargs,
) -> PyGnn.DeepGCNLayer:
    """
    Create a DeepGCNLayer with configurable convolution.

    Args:
        hidden_dim: Hidden dimension
        genconv_num_layers: Number of MLP layers in GENConv
        norm_type: 'layer' or 'batch' normalization
        dropout: Dropout probability
        edge_dim: Edge feature dimension (None = no edge features)
        conv_type: 'genconv' or 'gatv2'
        num_heads: Number of attention heads for gatv2

    Returns:
        DeepGCNLayer with res+ block
    """
    if conv_type == 'gin':
        expansion = kwargs.get('mlp_expansion', 2)
        mlp_depth = kwargs.get('mlp_depth', 2)
        conv = GINWithResidualMLP(hidden_dim, act_type, expansion=expansion, mlp_depth=mlp_depth)
    elif conv_type == 'gatv2':
        conv = GATv2WithFFN(
            hidden_dim, num_heads, dropout, edge_dim,
            act_type=act_type,
        )
    else:
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

    act = get_activation(act_type)

    return PyGnn.DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)
