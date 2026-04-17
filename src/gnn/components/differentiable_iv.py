"""
Differentiable I-V model for MOSFETs.

Predicts drain current log10(|ID|) from terminal voltages and a device embedding,
then derives gm and gds via torch.autograd.grad in log-space.

Log-space differentiation avoids the gradient scaling problem of exponentiating
first: d(10^x)/dx = ln(10)*10^x creates a ~1e-6 factor for small currents.
Instead, we differentiate log10(|ID|) directly and reconstruct log10(gm) via:
  log10(gm) = log10(|ID|) + log10(ln(10)) + log10(|d(log10|ID|)/dVGS|)
"""

import torch
import torch.nn as nn

# log10(ln(10)) = log10(2.302585...) ≈ 0.3622
LOG10_LN10 = 0.36221568869946325


class _ResidualBlock(nn.Module):
    """Residual block with SiLU activation (smooth, twice differentiable)."""

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.dropout(self.act(self.linear(x)))


class DifferentiableIVModel(nn.Module):
    """
    Differentiable MOSFET I-V model with log-space derivatives.

    Input: VGS, VDS, VBS (z-score normalized) + device_embedding
    Output: log10(|ID|), log10(gm), log10(gds)

    The MLP predicts log10(|ID|), then gm/gds are derived via autograd
    on the log-space output. SiLU activations ensure smooth derivatives.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        in_dim = 3 + embed_dim  # VGS, VDS, VBS + device embedding

        if residual:
            # Residual architecture: projection → residual blocks → output
            layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            for _ in range(num_layers - 1):
                layers.append(_ResidualBlock(hidden_dim, dropout))
            layers.append(nn.Linear(hidden_dim, 1))
            self.mlp = nn.Sequential(*layers)
        else:
            # Original sequential MLP
            layers = []
            curr = in_dim
            for i in range(num_layers):
                layers.append(nn.Linear(curr, hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0 and i < num_layers - 1:
                    layers.append(nn.Dropout(dropout))
                curr = hidden_dim
            layers.append(nn.Linear(curr, 1))
            self.mlp = nn.Sequential(*layers)

    def forward(self, vgs, vds, vbs, device_emb, training=True):
        """
        Forward pass: predict log10(|ID|), compute log10(gm) and log10(gds) in log-space.

        Args:
            vgs: [M] VGS normalized, requires_grad=True
            vds: [M] VDS normalized, requires_grad=True
            vbs: [M] VBS normalized (body effect context, no autograd)
            device_emb: [M, D] device embedding
            training: If True, create_graph=True for backprop through derivatives.

        Returns:
            dict with:
                'log_abs_id': [M] predicted log10(|ID|)
                'log10_gm': [M] log10(gm) computed in log-space
                'log10_gds': [M] log10(gds) computed in log-space
                'd_log_id_dvgs': [M] d(log10|ID|)/dVGS (raw log-space derivative)
                'd_log_id_dvds': [M] d(log10|ID|)/dVDS (raw log-space derivative)
        """
        # Build input: [M, 3 + embed_dim]
        x = torch.cat([vgs.unsqueeze(-1), vds.unsqueeze(-1), vbs.unsqueeze(-1), device_emb], dim=-1)

        # Predict log10(|ID|)
        log_abs_id = self.mlp(x).squeeze(-1)  # [M]
        log_abs_id = log_abs_id.clamp(-15, 0)  # stability: 1fA to 1A range

        # Differentiate in log-space (well-scaled, ~O(1) derivatives)
        d_log_id_dvgs = torch.autograd.grad(
            log_abs_id.sum(), vgs,
            create_graph=training, retain_graph=True,
        )[0]  # [M]

        d_log_id_dvds = torch.autograd.grad(
            log_abs_id.sum(), vds,
            create_graph=training,
        )[0]  # [M]

        # Reconstruct log10(gm) and log10(gds) entirely in log-space:
        #   gm = |ID| * ln(10) * |d(log10|ID|)/dVGS|
        #   log10(gm) = log10(|ID|) + log10(ln(10)) + log10(|d(log10|ID|)/dVGS|)
        log10_gm = log_abs_id + LOG10_LN10 + torch.log10(d_log_id_dvgs.abs().clamp(min=1e-20))
        log10_gds = log_abs_id + LOG10_LN10 + torch.log10(d_log_id_dvds.abs().clamp(min=1e-20))

        return {
            'log_abs_id': log_abs_id,
            'log10_gm': log10_gm,
            'log10_gds': log10_gds,
            'd_log_id_dvgs': d_log_id_dvgs,
            'd_log_id_dvds': d_log_id_dvds,
        }
