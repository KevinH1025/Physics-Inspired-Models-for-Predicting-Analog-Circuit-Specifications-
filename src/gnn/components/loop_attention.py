"""
Loop-aware multi-head attention for circuit GNNs.

Supports two levels:
- 'node': Node-to-node attention between terminal nodes of devices sharing
  a circuit loop (KVL-inspired). Each terminal attends directly to terminals
  of connected devices — no information loss from pooling.
- 'device': Device-level attention (legacy). Terminal embeddings are mean-pooled
  to device embeddings, devices attend, results scattered back to terminals.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import softmax


class LoopAttention(nn.Module):
    """Multi-head attention over circuit loop edges.

    Args:
        hidden_dim: Node embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        fusion: 'add' (direct) or 'gate' (learned scalar gate).
        dropout: Attention weight dropout.
        level: 'node' (terminal-to-terminal) or 'device' (pooled device-level).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        head_dim: int = 32,
        fusion: str = 'add',
        dropout: float = 0.0,
        level: str = 'node',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        self.level = level

        self.W_q = nn.Linear(hidden_dim, self.total_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, self.total_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, self.total_dim, bias=False)
        self.W_out = nn.Linear(self.total_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.fusion = fusion
        if fusion == 'gate':
            self.gate_proj = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)

        self._scale = head_dim ** -0.5

    def _forward_node(
        self,
        x: torch.Tensor,
        device_terminal_map: torch.Tensor,
        loop_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Node-to-node attention between terminal nodes sharing a KVL loop.

        When level='node', loop_edge_index is already pre-computed node-level
        (terminal-to-terminal) edges from _get_loop_data — no expansion needed.
        """
        N = x.shape[0]
        device = x.device

        src, dst = loop_edge_index

        # Q, K, V on node embeddings directly
        Q = self.W_q(x).view(N, self.num_heads, self.head_dim)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim)

        # Sparse attention: Q[dst] · K[src]
        q_dst = Q[dst]  # [E_node, H, d]
        k_src = K[src]  # [E_node, H, d]
        scores = (q_dst * k_src).sum(dim=-1) * self._scale  # [E_node, H]

        # Softmax per destination node, per head
        alpha = softmax(scores, index=dst, num_nodes=N, dim=0)
        alpha = self.attn_drop(alpha)

        # Weighted aggregation of source values → scatter to destination nodes
        v_src = V[src]  # [E_node, H, d]
        weighted_v = v_src * alpha.unsqueeze(-1)  # [E_node, H, d]
        out = torch.zeros(N, self.num_heads, self.head_dim, device=device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_v), weighted_v)

        out = out.reshape(N, self.total_dim)
        out = self.W_out(out)
        out = self.norm(out)

        # Gate fusion
        if self.fusion == 'gate':
            gate = torch.sigmoid(self.gate_proj(x))  # [N, 1]
            out = gate * out

        return out

    def _forward_device(
        self,
        x: torch.Tensor,
        device_terminal_map: torch.Tensor,
        loop_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Device-level attention (legacy). Pools terminals, attends, scatters back."""
        D = device_terminal_map.shape[0]
        device = x.device

        # Pool terminal embeddings → device embeddings
        mask = device_terminal_map >= 0
        safe_idx = device_terminal_map.clamp(min=0)
        term_emb = x[safe_idx]  # [D, max_terms, hidden_dim]
        term_emb = term_emb * mask.unsqueeze(-1).float()
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        device_emb = term_emb.sum(dim=1) / counts  # [D, hidden_dim]

        # Sparse multi-head attention over loop edges
        src, dst = loop_edge_index

        Q = self.W_q(device_emb).view(D, self.num_heads, self.head_dim)
        K = self.W_k(device_emb).view(D, self.num_heads, self.head_dim)
        V = self.W_v(device_emb).view(D, self.num_heads, self.head_dim)

        q_dst = Q[dst]
        k_src = K[src]
        scores = (q_dst * k_src).sum(dim=-1) * self._scale

        alpha = softmax(scores, index=dst, num_nodes=D, dim=0)
        alpha = self.attn_drop(alpha)

        v_src = V[src]
        weighted_v = v_src * alpha.unsqueeze(-1)
        out = torch.zeros(D, self.num_heads, self.head_dim, device=device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_v), weighted_v)

        out = out.reshape(D, self.total_dim)
        out = self.W_out(out)
        out = self.norm(out)

        # Scatter device output back to terminal nodes
        x_update = torch.zeros_like(x)
        for t in range(device_terminal_map.shape[1]):
            col = device_terminal_map[:, t]
            valid = col >= 0
            if valid.any():
                x_update[col[valid]] += out[valid]

        # Gate fusion
        if self.fusion == 'gate':
            gate = torch.sigmoid(self.gate_proj(x))
            x_update = gate * x_update

        return x_update

    def forward(
        self,
        x: torch.Tensor,
        device_terminal_map: torch.Tensor,
        loop_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, hidden_dim] node embeddings.
            device_terminal_map: [D, max_terms] terminal indices per device (-1 = pad).
            loop_edge_index: [2, E] edges — device-level for 'device' mode,
                node-level (terminal-to-terminal) for 'node' mode.

        Returns:
            x_update: [N, hidden_dim] additive update for node embeddings.
        """
        if loop_edge_index.shape[1] == 0:
            return torch.zeros_like(x)

        if self.level == 'node':
            return self._forward_node(x, device_terminal_map, loop_edge_index)
        else:
            return self._forward_device(x, device_terminal_map, loop_edge_index)
