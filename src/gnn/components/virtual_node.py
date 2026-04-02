"""
Virtual Node module for graph neural networks.

The virtual node acts as a global information aggregator that:
1. Broadcasts global context to all nodes
2. Aggregates information from all nodes
3. Updates its representation after each message passing layer

Supports two modes:
- 'default': Single learnable VN embedding, broadcast same vector to all nodes
- 'mha': Global multi-head self-attention, each node gets personalized global context
"""

import torch
import torch.nn as nn
import torch_geometric.nn as PyGnn
import torch_geometric.utils
from torch_scatter import scatter
from .layers import get_activation


class VirtualNode(nn.Module):
    """
    Virtual Node for improved message passing in GNNs.

    Args:
        hidden_dim: Hidden dimension
        use_attention_pooling: Use attention-weighted aggregation instead of mean
        learn_temperature: Whether attention temperature is learnable
        gate_broadcast: Use learned sigmoid gate on broadcast/MHA output
        mode: 'default' (single VN embedding) or 'mha' (global self-attention)
        num_heads: Number of attention heads (MHA mode only)
        head_dim: Dimension per head (MHA mode only)
    """

    def __init__(
        self,
        hidden_dim: int,
        use_attention_pooling: bool = True,
        learn_temperature: bool = False,
        gate_broadcast: bool = False,
        mode: str = 'default',
        num_heads: int = 4,
        head_dim: int = 32,
        num_layers: int = 1,
        act_type: str = 'relu',
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_attention_pooling = use_attention_pooling
        self.mode = mode
        self.gate_broadcast = gate_broadcast
        if gate_broadcast:
            self.gate_projs = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_layers)
            ])
            for proj in self.gate_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

        if mode == 'mha':
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.total_dim = num_heads * head_dim
            self._scale = head_dim ** -0.5

            self.W_q = nn.Linear(hidden_dim, self.total_dim, bias=False)
            self.W_k = nn.Linear(hidden_dim, self.total_dim, bias=False)
            self.W_v = nn.Linear(hidden_dim, self.total_dim, bias=False)
            self.W_out = nn.Linear(self.total_dim, hidden_dim, bias=False)
            self.mha_norm = nn.LayerNorm(hidden_dim)
        else:
            # Default mode: single VN embedding
            # Learnable initial embedding
            self.embedding = nn.Parameter(torch.zeros(1, hidden_dim))
            nn.init.xavier_uniform_(self.embedding)

            # MLP for updating virtual node
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                get_activation(act_type),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # Attention pooling
            if use_attention_pooling:
                self.attention_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    get_activation(act_type),
                    nn.Linear(hidden_dim // 2, 1),
                )
                if learn_temperature:
                    self.temperature = nn.Parameter(torch.tensor(1.0))
                else:
                    self.register_buffer('temperature', torch.tensor(1.0))

    def init_embedding(self, num_graphs: int):
        """Initialize virtual node embeddings for a batch. Returns None for MHA mode."""
        if self.mode == 'mha':
            return None
        return self.embedding.expand(num_graphs, -1)

    def broadcast(self, vn_emb: torch.Tensor, batch: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Broadcast virtual node embeddings to all nodes (default mode only)."""
        out = vn_emb[batch]
        if self.gate_broadcast:
            gate = torch.sigmoid(self.gate_projs[layer_idx](out))  # [N, 1]
            out = gate * out
        return out

    def aggregate(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Aggregate node features to update virtual node (default mode only)."""
        if self.use_attention_pooling:
            attn_scores = self.attention_mlp(x)
            temperature = self.temperature.clamp(min=0.1)
            attn_weights = torch_geometric.utils.softmax(attn_scores / temperature, batch)
            return PyGnn.global_add_pool(x * attn_weights, batch, size=num_graphs)
        else:
            return PyGnn.global_mean_pool(x, batch, size=num_graphs)

    def update(self, vn_emb: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """Update virtual node embedding with aggregated features (default mode only)."""
        return vn_emb + self.mlp(aggregated)

    def global_mha(self, x: torch.Tensor, batch: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Global multi-head self-attention within each graph.

        Uses dense batched attention (fixed topology = equal-sized graphs).
        Every node attends to every other node in the same graph.

        Args:
            x: Node features [N, hidden_dim]
            batch: Graph assignment [N]

        Returns:
            MHA output [N, hidden_dim]
        """
        N = x.shape[0]

        # Fixed topology: all graphs have same number of nodes
        _, counts = batch.unique(return_counts=True)
        n_per = counts[0].item()
        B = N // n_per

        # Project and reshape to [B, n_per, H, d]
        Q = self.W_q(x).view(B, n_per, self.num_heads, self.head_dim)
        K = self.W_k(x).view(B, n_per, self.num_heads, self.head_dim)
        V = self.W_v(x).view(B, n_per, self.num_heads, self.head_dim)

        # Transpose to [B, H, n_per, d] for batched matmul
        Q = Q.permute(0, 2, 1, 3)  # [B, H, n_per, d]
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Attention: [B, H, n_per, n_per]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self._scale
        alpha = torch.softmax(scores, dim=-1)

        # Weighted sum: [B, H, n_per, d]
        out = torch.matmul(alpha, V)

        # Reshape back to [N, total_dim]
        out = out.permute(0, 2, 1, 3).reshape(N, self.total_dim)
        out = self.W_out(out)
        out = self.mha_norm(out)

        if self.gate_broadcast:
            gate = torch.sigmoid(self.gate_projs[layer_idx](x))  # [N, 1]
            out = gate * out

        return out

    def forward(
        self,
        x: torch.Tensor,
        vn_emb: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple:
        """
        Full virtual node update step (default mode only).

        Returns:
            Tuple of (updated_vn_emb, broadcast_features)
        """
        aggregated = self.aggregate(x, batch, num_graphs)
        vn_emb = self.update(vn_emb, aggregated)
        broadcast = self.broadcast(vn_emb, batch)
        return vn_emb, broadcast
