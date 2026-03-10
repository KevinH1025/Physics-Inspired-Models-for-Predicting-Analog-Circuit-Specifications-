"""
GNN-based current prediction from voltage-augmented node features.

Instead of using a simple MLP to predict currents from voltage differences,
this module uses additional GNN message-passing layers to leverage circuit
context (neighboring devices, topology patterns like current mirrors).
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import build_mlp, create_deepgcn_layer


class CurrentGNNBackbone(nn.Module):
    """
    GNN backbone for current prediction from voltage-augmented node features.

    Takes node embeddings augmented with predicted voltages and performs
    additional message passing before a current prediction head. This allows
    the model to learn context-aware current predictions that can capture
    patterns like current mirrors, differential pairs, and load effects.

    Architecture:
        Input [hidden_dim + 1] → Linear → GEN layers → Current head → [1]

    Args:
        input_dim: Input dimension (typically hidden_dim + 1 for voltage)
        hidden_dim: Hidden dimension for GEN layers
        num_layers: Number of GEN message-passing layers
        dropout: Dropout probability
        genconv_num_layers: MLP layers within each GENConv
        norm_type: 'layer' or 'batch' normalization
        current_head_layers: Number of MLP layers in current head
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        genconv_num_layers: int = 2,
        norm_type: str = 'layer',
        current_head_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection from augmented features to hidden_dim
        self.input_linear = nn.Linear(input_dim, hidden_dim)

        # GEN layers for current-specific message passing
        self.layers = nn.ModuleList([
            create_deepgcn_layer(hidden_dim, genconv_num_layers, norm_type, dropout)
            for _ in range(num_layers)
        ])

        # Current prediction head
        self.current_head = build_mlp(
            num_layers=current_head_layers,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            norm_type=norm_type,
            dropout=dropout,
        )

    def forward(
        self,
        x_augmented: torch.Tensor,
        edge_index: torch.Tensor,
        virtual_node=None,
        vn_emb: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ):
        """
        Forward pass for current prediction.

        Args:
            x_augmented: Node features augmented with voltage [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            virtual_node: Optional VirtualNode module for global context
            vn_emb: Optional initial VN embedding [num_graphs, hidden_dim] (carried from backbone)
            batch: Batch assignment [num_nodes] (required if virtual_node is set)
            num_graphs: Number of graphs in batch (required if virtual_node is set)

        Returns:
            Tuple of (current_pred [num_nodes, 1], vn_emb [num_graphs, hidden_dim] or None)
        """
        # Project to hidden dimension
        x = self.input_linear(x_augmented)

        # Message passing layers (with optional virtual node)
        for layer in self.layers:
            if virtual_node is not None and vn_emb is not None:
                x = x + virtual_node.broadcast(vn_emb, batch)

            x = layer(x, edge_index)

            if virtual_node is not None and vn_emb is not None:
                vn_emb, _ = virtual_node(x, vn_emb, batch, num_graphs)

        # Predict currents
        return self.current_head(x), vn_emb

    def get_embeddings(
        self,
        x_augmented: torch.Tensor,
        edge_index: torch.Tensor,
        virtual_node=None,
        vn_emb: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get node embeddings after message passing (before current head).

        Useful for refinement passes where we need intermediate features.

        Args:
            x_augmented: Node features augmented with voltage [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            virtual_node: Optional VirtualNode module
            vn_emb: Optional initial VN embedding [num_graphs, hidden_dim]
            batch: Batch assignment [num_nodes]
            num_graphs: Number of graphs in batch

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Project to hidden dimension
        x = self.input_linear(x_augmented)

        # Message passing layers
        for layer in self.layers:
            if virtual_node is not None and vn_emb is not None:
                x = x + virtual_node.broadcast(vn_emb, batch)

            x = layer(x, edge_index)

            if virtual_node is not None and vn_emb is not None:
                vn_emb, _ = virtual_node(x, vn_emb, batch, num_graphs)

        return x


def propagate_voltages_to_terminals(
    pred_voltages: torch.Tensor,
    edge_index: torch.Tensor,
    num_terminals: torch.Tensor,
    ptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Propagate predicted voltages from NET nodes to TERMINAL nodes.

    In the graph structure, voltages are predicted at all nodes but only
    NET node predictions are meaningful (TERMINALs have simulation values).
    This function copies each NET node's voltage to its connected TERMINALs
    so that the current GNN has voltage information at all nodes.

    The edge_index has bidirectional terminal<->net connections:
    - Terminal indices: [0, num_terminals)
    - Net indices: [num_terminals, num_nodes)

    For each terminal->net edge, we copy: V_terminal = V_net

    Args:
        pred_voltages: Predicted voltages for all nodes [num_nodes]
        edge_index: Bidirectional terminal<->net edges [2, num_edges]
        num_terminals: Number of terminals per graph (int or Tensor)
        ptr: Node boundaries for batched graphs [num_graphs + 1]

    Returns:
        node_voltages: Voltage at each node [num_nodes]
            - NET nodes: their predicted voltage
            - TERMINAL nodes: voltage of their connected net
    """
    device = pred_voltages.device
    num_nodes = len(pred_voltages)

    # Start with predicted voltages (will be overwritten for terminals)
    node_voltages = pred_voltages.clone()

    # Handle batched vs single graph
    if ptr is None or len(ptr) <= 2:
        # Single graph or no batching
        if isinstance(num_terminals, torch.Tensor):
            n_terms = int(num_terminals.item()) if num_terminals.numel() == 1 else int(num_terminals[0].item())
        else:
            n_terms = int(num_terminals)

        src, dst = edge_index

        # Find terminal->net edges (src is terminal, dst is net)
        is_term_to_net = (src < n_terms) & (dst >= n_terms)

        # Copy net voltage to terminal
        terminal_indices = src[is_term_to_net]
        net_indices = dst[is_term_to_net]
        node_voltages[terminal_indices] = pred_voltages[net_indices]

    else:
        # Batched graphs - need to handle offsets
        num_graphs = len(ptr) - 1
        src, dst = edge_index

        # Build batch index for each node
        graph_sizes = ptr[1:] - ptr[:-1]
        batch_idx = torch.repeat_interleave(
            torch.arange(num_graphs, device=device), graph_sizes
        )

        # Local index within each graph
        local_idx = torch.arange(num_nodes, device=device) - ptr[batch_idx]

        # Terminal mask based on num_terminals per graph
        if isinstance(num_terminals, torch.Tensor):
            if num_terminals.numel() == 1:
                # Same num_terminals for all graphs
                terminal_mask = local_idx < num_terminals.item()
            else:
                # Different num_terminals per graph
                terminal_mask = local_idx < num_terminals[batch_idx]
        else:
            terminal_mask = local_idx < num_terminals

        # Find terminal->net edges
        is_term_to_net = terminal_mask[src] & ~terminal_mask[dst]

        # Copy voltages
        node_voltages[src[is_term_to_net]] = pred_voltages[dst[is_term_to_net]]

    return node_voltages
