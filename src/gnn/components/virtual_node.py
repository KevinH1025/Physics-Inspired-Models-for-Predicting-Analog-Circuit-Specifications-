"""
Virtual Node module for graph neural networks.

The virtual node acts as a global information aggregator that:
1. Broadcasts global context to all nodes
2. Aggregates information from all nodes
3. Updates its representation after each message passing layer
"""

import torch
import torch.nn as nn
import torch_geometric.nn as PyGnn
import torch_geometric.utils


class VirtualNode(nn.Module):
    """
    Virtual Node for improved message passing in GNNs.

    The virtual node maintains a learnable embedding per graph that:
    - Gets added to node features before each GNN layer
    - Aggregates node features after each layer (mean or attention pooling)
    - Updates via an MLP

    Args:
        hidden_dim: Hidden dimension
        use_attention_pooling: Use attention-weighted aggregation instead of mean
        learn_temperature: Whether attention temperature is learnable
    """

    def __init__(
        self,
        hidden_dim: int,
        use_attention_pooling: bool = True,
        learn_temperature: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_attention_pooling = use_attention_pooling

        # Learnable initial embedding
        self.embedding = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_uniform_(self.embedding)

        # MLP for updating virtual node
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention pooling
        if use_attention_pooling:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if learn_temperature:
                self.temperature = nn.Parameter(torch.tensor(1.0))
            else:
                self.register_buffer('temperature', torch.tensor(1.0))

    def init_embedding(self, num_graphs: int) -> torch.Tensor:
        """
        Initialize virtual node embeddings for a batch.

        Args:
            num_graphs: Number of graphs in batch

        Returns:
            Virtual node embeddings [num_graphs, hidden_dim]
        """
        return self.embedding.expand(num_graphs, -1)

    def broadcast(self, vn_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Broadcast virtual node embeddings to all nodes.

        Args:
            vn_emb: Virtual node embeddings [num_graphs, hidden_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Per-node virtual node features [num_nodes, hidden_dim]
        """
        return vn_emb[batch]

    def aggregate(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        Aggregate node features to update virtual node.

        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
            num_graphs: Number of graphs

        Returns:
            Aggregated features [num_graphs, hidden_dim]
        """
        if self.use_attention_pooling:
            attn_scores = self.attention_mlp(x)
            temperature = self.temperature.clamp(min=0.1)
            attn_weights = torch_geometric.utils.softmax(attn_scores / temperature, batch)
            return PyGnn.global_add_pool(x * attn_weights, batch, size=num_graphs)
        else:
            return PyGnn.global_mean_pool(x, batch, size=num_graphs)

    def update(self, vn_emb: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """
        Update virtual node embedding with aggregated features.

        Args:
            vn_emb: Current virtual node embedding [num_graphs, hidden_dim]
            aggregated: Aggregated node features [num_graphs, hidden_dim]

        Returns:
            Updated virtual node embedding [num_graphs, hidden_dim]
        """
        return vn_emb + self.mlp(aggregated)

    def forward(
        self,
        x: torch.Tensor,
        vn_emb: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple:
        """
        Full virtual node update step.

        Args:
            x: Node features after GNN layer [num_nodes, hidden_dim]
            vn_emb: Current virtual node embedding [num_graphs, hidden_dim]
            batch: Batch assignment [num_nodes]
            num_graphs: Number of graphs

        Returns:
            Tuple of (updated_vn_emb, broadcast_features)
        """
        aggregated = self.aggregate(x, batch, num_graphs)
        vn_emb = self.update(vn_emb, aggregated)
        broadcast = self.broadcast(vn_emb, batch)
        return vn_emb, broadcast
