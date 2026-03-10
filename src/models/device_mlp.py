"""
Standalone Device MLP for MOSFET I-V prediction.

This module implements a learnable MOSFET I-V model trained on SPICE simulation data.
The model learns to predict drain-source current (I_ds) from terminal voltages (Vgs, Vds)
and device parameters (W, L).

The trained model can be frozen and used within a GNN to provide physics-informed
gradients for voltage prediction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and ReLU activation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(x + self.net(x)))


class DeviceMLP(nn.Module):
    """
    Standalone MOSFET I-V model trained on SPICE data.

    Learns the mapping: (Vgs, Vds, log10(W), log10(L), is_nmos) -> log10(I_ds)

    This model captures the full BSIM4 transistor behavior including:
    - Strong inversion (saturation, triode)
    - Weak inversion (subthreshold)
    - Channel length modulation
    - Short-channel effects

    Args:
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of hidden layers (default: 3)
        dropout: Dropout probability (default: 0.0)
        use_separate_heads: If True, use separate output heads for NMOS/PMOS
        use_polynomial_features: If True, add Vgs², Vds², Vgs×Vds features
        use_residual: If True, use residual connections in hidden layers
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_separate_heads: bool = False,
        use_polynomial_features: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_separate_heads = use_separate_heads
        self.use_polynomial_features = use_polynomial_features
        self.use_residual = use_residual

        # Input: [Vgs, Vds, log10(W), log10(L), is_nmos] -> 5 features
        # With polynomial: + [Vgs², Vds², Vgs×Vds] -> 8 features
        input_dim = 8 if use_polynomial_features else 5

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        if use_residual:
            # Residual blocks (each block has 2 layers, so num_layers//2 blocks)
            num_blocks = max(1, (num_layers - 1) // 2)
            self.backbone = nn.Sequential(*[
                ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
            ])
        else:
            # Standard MLP layers
            layers = []
            for i in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        if use_separate_heads:
            # Separate heads for NMOS and PMOS
            self.nmos_head = nn.Linear(hidden_dim, 1)
            self.pmos_head = nn.Linear(hidden_dim, 1)
        else:
            # Single output head
            self.output_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to prevent large initial predictions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize output layer(s) with small weights
        if self.use_separate_heads:
            nn.init.normal_(self.nmos_head.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.pmos_head.weight, mean=0.0, std=0.01)
        else:
            nn.init.normal_(self.output_head.weight, mean=0.0, std=0.01)

    def forward(
        self,
        Vgs: torch.Tensor,
        Vds: torch.Tensor,
        log_W: torch.Tensor,
        log_L: torch.Tensor,
        is_nmos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict drain-source current from voltages and device parameters.

        Args:
            Vgs: Gate-source voltage (Volts) [batch]
            Vds: Drain-source voltage (Volts) [batch]
            log_W: log10(W in meters) [batch]
            log_L: log10(L in meters) [batch]
            is_nmos: 1 for NMOS, 0 for PMOS [batch]

        Returns:
            log10(I_ds) - log-scale drain-source current [batch]
        """
        # Stack inputs
        if self.use_polynomial_features:
            # Add polynomial features: Vgs², Vds², Vgs×Vds
            # These capture MOSFET physics: Ids ∝ (Vgs-Vth)² in saturation, Vds² in triode
            x = torch.stack([
                Vgs, Vds, log_W, log_L, is_nmos.float(),
                Vgs ** 2, Vds ** 2, Vgs * Vds
            ], dim=-1)
        else:
            x = torch.stack([Vgs, Vds, log_W, log_L, is_nmos.float()], dim=-1)

        # Forward through input projection and backbone
        x = self.input_proj(x)
        features = self.backbone(x)

        if self.use_separate_heads:
            # Use separate heads based on device type
            nmos_out = self.nmos_head(features).squeeze(-1)
            pmos_out = self.pmos_head(features).squeeze(-1)
            # Select based on is_nmos
            is_nmos_mask = is_nmos.bool()
            out = torch.where(is_nmos_mask, nmos_out, pmos_out)
        else:
            out = self.output_head(features).squeeze(-1)

        return out

    def forward_dict(
        self,
        inputs: dict,
    ) -> torch.Tensor:
        """Forward pass with dict inputs (for standalone training)."""
        return self.forward(
            Vgs=inputs['Vgs'],
            Vds=inputs['Vds'],
            log_W=inputs['log_W'],
            log_L=inputs['log_L'],
            is_nmos=inputs['is_nmos'],
        )


class DeviceMLPWithNormalization(nn.Module):
    """
    Device MLP wrapper that handles input/output normalization.

    Wraps DeviceMLP to handle normalization of inputs and outputs,
    making it easier to integrate with the GNN pipeline.

    Args:
        device_mlp: Base DeviceMLP model
        voltage_mean: Mean for voltage normalization
        voltage_std: Std for voltage normalization
        current_mean: Mean for log10 current output (z-score)
        current_std: Std for log10 current output (z-score)
    """

    def __init__(
        self,
        device_mlp: DeviceMLP,
        voltage_mean: float = 0.0,
        voltage_std: float = 1.0,
        current_mean: float = 0.0,
        current_std: float = 1.0,
    ):
        super().__init__()
        self.device_mlp = device_mlp
        self.register_buffer('voltage_mean', torch.tensor(voltage_mean))
        self.register_buffer('voltage_std', torch.tensor(voltage_std))
        self.register_buffer('current_mean', torch.tensor(current_mean))
        self.register_buffer('current_std', torch.tensor(current_std))

    def forward(
        self,
        Vgs_norm: torch.Tensor,
        Vds_norm: torch.Tensor,
        log_W: torch.Tensor,
        log_L: torch.Tensor,
        is_nmos: torch.Tensor,
        denormalize_voltages: bool = True,
        normalize_current: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with automatic normalization handling.

        Args:
            Vgs_norm: Normalized gate-source voltage
            Vds_norm: Normalized drain-source voltage
            log_W: log10(W in meters) - already normalized
            log_L: log10(L in meters) - already normalized
            is_nmos: Device type flag
            denormalize_voltages: If True, convert voltages from z-score to raw
            normalize_current: If True, convert output to z-score

        Returns:
            Current prediction (normalized if normalize_current=True)
        """
        # Denormalize voltages if needed
        if denormalize_voltages:
            Vgs = Vgs_norm * self.voltage_std + self.voltage_mean
            Vds = Vds_norm * self.voltage_std + self.voltage_mean
        else:
            Vgs = Vgs_norm
            Vds = Vds_norm

        # Predict log10(I_ds)
        log_I_ds = self.device_mlp(Vgs, Vds, log_W, log_L, is_nmos)

        # Normalize output if needed
        if normalize_current:
            log_I_ds_norm = (log_I_ds - self.current_mean) / self.current_std
            return log_I_ds_norm
        else:
            return log_I_ds


def create_device_mlp(
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.0,
    use_separate_heads: bool = False,
    use_polynomial_features: bool = False,
    use_residual: bool = False,
    checkpoint_path: Optional[str] = None,
    freeze: bool = False,
) -> DeviceMLP:
    """
    Factory function to create a DeviceMLP, optionally loading from checkpoint.

    Args:
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        dropout: Dropout probability
        use_separate_heads: Use separate NMOS/PMOS heads
        use_polynomial_features: Add Vgs², Vds², Vgs×Vds features
        use_residual: Use residual connections
        checkpoint_path: Path to checkpoint file (optional)
        freeze: If True, freeze all parameters

    Returns:
        DeviceMLP model
    """
    model = DeviceMLP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_separate_heads=use_separate_heads,
        use_polynomial_features=use_polynomial_features,
        use_residual=use_residual,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded DeviceMLP from {checkpoint_path}")

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("DeviceMLP frozen")

    return model
