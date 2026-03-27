"""
Training visualization utilities.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_voltage_losses: List[float],
    val_voltage_losses: List[float],
    train_maes: List[float],
    val_maes: List[float],
    learning_rates: List[float],
    best_metrics: Dict,
    best_val_loss: float,
    best_epoch: int,
    output_path: Path,
    val_freq: int = 1,
    predict_currents: bool = False,
    train_current_losses: List[float] = None,
    val_current_losses: List[float] = None,
    train_current_maes: List[float] = None,
    val_current_maes: List[float] = None,
    train_ss_gm_losses: List[float] = None,
    val_ss_gm_losses: List[float] = None,
    train_ss_gds_losses: List[float] = None,
    val_ss_gds_losses: List[float] = None,
    train_kcl_losses: List[float] = None,
    val_kcl_losses: List[float] = None,
    train_ac_losses: List[float] = None,
    val_ac_losses: List[float] = None,
    train_ac_component_losses: Dict[str, List[float]] = None,
    val_ac_component_losses: Dict[str, List[float]] = None,
    train_region_losses: List[float] = None,
    val_region_losses: List[float] = None,
    train_gm_physics_losses: List[float] = None,
    val_gm_physics_losses: List[float] = None,
    train_triode_physics_losses: List[float] = None,
    val_triode_physics_losses: List[float] = None,
    train_cutoff_physics_losses: List[float] = None,
    val_cutoff_physics_losses: List[float] = None,
    train_dc_gain_losses: List[float] = None,
    val_dc_gain_losses: List[float] = None,
    max_grad_norms: List[float] = None,
    avg_grad_norms: List[float] = None,
    num_train_batches: int = 0,
    num_val_batches: int = 0,
) -> None:
    """Generate and save training curves plot."""
    num_epochs = len(train_losses)

    # Determine grid size based on what losses are active
    has_region = train_region_losses and any(v > 0 for v in train_region_losses)
    has_ac = train_ac_losses and any(v > 0 for v in train_ac_losses)
    has_kcl = train_kcl_losses and any(v > 0 for v in train_kcl_losses)
    # Count AC component plots (3 separate if per-component available, else 1)
    n_ac_plots = len(train_ac_component_losses) if (has_ac and train_ac_component_losses) else (1 if has_ac else 0)
    n_kcl_ac_plots = (1 if has_kcl else 0) + n_ac_plots + (1 if has_region else 0)
    has_dc_gain = train_dc_gain_losses and any(v > 0 for v in train_dc_gain_losses)
    n_kcl_ac_plots += (1 if has_dc_gain else 0)
    has_kcl_ac = n_kcl_ac_plots > 0
    # Need extra row if KCL+AC plots exceed 4 columns
    kcl_ac_rows = max(1, (n_kcl_ac_plots + 3) // 4) if has_kcl_ac else 0
    has_physics = (train_gm_physics_losses and any(v > 0 for v in train_gm_physics_losses)) or \
                  (train_triode_physics_losses and any(v > 0 for v in train_triode_physics_losses)) or \
                  (train_cutoff_physics_losses and any(v > 0 for v in train_cutoff_physics_losses))
    has_grad_norms = max_grad_norms and len(max_grad_norms) > 0
    nrows = 2 + kcl_ac_rows + (1 if has_physics else 0) + (1 if has_grad_norms else 0)
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 4 * nrows))
    fig.suptitle(f'Training Curves - {num_train_batches} train, {num_val_batches} val, {num_epochs} epochs', fontsize=12)

    train_epochs = list(range(len(train_losses)))
    val_epochs = list(range(0, len(train_losses), val_freq))[:len(val_losses)]

    # Row 1: Total Loss, Voltage Loss, Voltage MAE, LR Schedule
    ax = axes[0, 0]
    ax.semilogy(train_epochs, train_losses, 'b-', label='Train', alpha=0.7)
    ax.semilogy(val_epochs, val_losses, 'r-', label='Val', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(train_epochs, train_voltage_losses, 'b-', label='Train', alpha=0.7)
    ax.semilogy(val_epochs, val_voltage_losses, 'r-', label='Val', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Voltage Loss')
    ax.set_title('Voltage Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(train_epochs, train_maes, 'b-', label='Train', alpha=0.7)
    ax.plot(val_epochs, val_maes, 'r-', label='Val', alpha=0.7)
    ax.axhline(y=best_metrics['val_mae_mv'], color='g', linestyle='--', label=f'Best: {best_metrics["val_mae_mv"]:.1f}mV')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (mV)')
    ax.set_title(f'Voltage MAE (Best: {best_metrics["val_mae_mv"]:.2f}mV @ ep {best_epoch})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 3]
    ax.semilogy(train_epochs, learning_rates, 'g-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    # Row 2: Current Loss, Current MAE, SS Loss, (empty or off)
    if predict_currents and train_current_losses:
        ax = axes[1, 0]
        ax.semilogy(train_epochs, train_current_losses, 'b-', label='Train', alpha=0.7)
        ax.semilogy(val_epochs, val_current_losses, 'r-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Current Loss (MSE)')
        ax.set_title('Current Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if train_current_maes:
            ax.plot(train_epochs, train_current_maes, 'b-', label='Train', alpha=0.7)
        ax.plot(val_epochs, val_current_maes, 'r-', label='Val', alpha=0.7)
        ax.axhline(y=best_metrics['val_current_mae_ua'], color='g', linestyle='--', label=f'Best: {best_metrics["val_current_mae_ua"]:.1f}µA')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (µA)')
        ax.set_title(f'Current MAE (Best: {best_metrics["val_current_mae_ua"]:.1f}µA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    if train_ss_gm_losses and any(v > 0 for v in train_ss_gm_losses):
        ax = axes[1, 2]
        ax.semilogy(train_epochs, train_ss_gm_losses, 'b-', label='Train', alpha=0.7)
        if val_ss_gm_losses:
            ax.semilogy(val_epochs, val_ss_gm_losses, 'r-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('gm Loss (MSE)')
        ax.set_title('SS gm Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')

    if train_ss_gds_losses and any(v > 0 for v in train_ss_gds_losses):
        ax = axes[1, 3]
        ax.semilogy(train_epochs, train_ss_gds_losses, 'b-', label='Train', alpha=0.7)
        if val_ss_gds_losses:
            ax.semilogy(val_epochs, val_ss_gds_losses, 'r-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('gds Loss (MSE)')
        ax.set_title('SS gds Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 3].axis('off')

    # Row 3+: KCL Loss, AC component losses, Region Loss (if present)
    if has_kcl_ac:
        kcl_ac_row = 2
        col = 0

        def _next_ax():
            nonlocal kcl_ac_row, col
            ax = axes[kcl_ac_row, col]
            col += 1
            if col >= 4:
                # Fill remaining cols in current row
                col = 0
                kcl_ac_row += 1
            return ax

        if has_kcl:
            ax = _next_ax()
            ax.semilogy(train_epochs, train_kcl_losses, 'b-', label='Train', alpha=0.7)
            if val_kcl_losses:
                ax.semilogy(val_epochs, val_kcl_losses, 'r-', label='Val', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('KCL Loss')
            ax.set_title('KCL Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        if has_ac:
            # Plot per-component AC losses if available, otherwise combined
            if train_ac_component_losses and len(train_ac_component_losses) > 0:
                comp_names = {'ugbw': 'UGBW', 'pm': 'PM', 'am': 'AM'}
                for comp in train_ac_component_losses:
                    ax = _next_ax()
                    tr_data = train_ac_component_losses[comp]
                    ax.semilogy(train_epochs[:len(tr_data)], tr_data, 'b-', label='Train', alpha=0.7)
                    if val_ac_component_losses and comp in val_ac_component_losses:
                        vl_data = val_ac_component_losses[comp]
                        ax.semilogy(val_epochs[:len(vl_data)], vl_data, 'r-', label='Val', alpha=0.7)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(f'{comp_names.get(comp, comp)} Loss')
                    ax.set_title(f'AC {comp_names.get(comp, comp)} Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            else:
                ax = _next_ax()
                ax.semilogy(train_epochs, train_ac_losses, 'b-', label='Train', alpha=0.7)
                if val_ac_losses:
                    ax.semilogy(val_epochs, val_ac_losses, 'r-', label='Val', alpha=0.7)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('AC Loss (MSE)')
                ax.set_title('AC Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)

        if has_region:
            ax = _next_ax()
            ax.plot(train_epochs, train_region_losses, 'b-', label='Train', alpha=0.7)
            if val_region_losses:
                ax.plot(val_epochs, val_region_losses, 'r-', label='Val', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Region Loss (ordinal MSE)')
            ax.set_title('Region Classification Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        if has_dc_gain:
            ax = _next_ax()
            ax.semilogy(train_epochs, train_dc_gain_losses, 'b-', label='Train', alpha=0.7)
            if val_dc_gain_losses:
                ax.semilogy(val_epochs, val_dc_gain_losses, 'r-', label='Val', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('DC Gain Loss (MSE)')
            ax.set_title('DC Gain Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Turn off unused axes in kcl/ac rows
        for r in range(2, 2 + kcl_ac_rows):
            start_col = col if r == kcl_ac_row else 0
            for c in range(start_col, 4):
                axes[r, c].axis('off')

    # Physics losses row: gm_sat, triode, cutoff
    if has_physics:
        phy_row = 2 + kcl_ac_rows
        col = 0

        if train_gm_physics_losses and any(v > 0 for v in train_gm_physics_losses):
            ax = axes[phy_row, col]
            ax.semilogy(train_epochs, train_gm_physics_losses, 'b-', label='Train', alpha=0.7)
            if val_gm_physics_losses:
                ax.semilogy(val_epochs, val_gm_physics_losses, 'r-', label='Val', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE log10)')
            ax.set_title('Gm Sat Physics Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            col += 1

        if train_triode_physics_losses and any(v > 0 for v in train_triode_physics_losses):
            ax = axes[phy_row, col]
            ax.semilogy(train_epochs, train_triode_physics_losses, 'b-', label='Train', alpha=0.7)
            if val_triode_physics_losses:
                ax.semilogy(val_epochs, val_triode_physics_losses, 'r-', label='Val', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE log10)')
            ax.set_title('Triode Physics Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            col += 1

        if train_cutoff_physics_losses and any(v > 0 for v in train_cutoff_physics_losses):
            ax = axes[phy_row, col]
            ax.semilogy(train_epochs, train_cutoff_physics_losses, 'b-', label='Train', alpha=0.7)
            if val_cutoff_physics_losses:
                ax.semilogy(val_epochs, val_cutoff_physics_losses, 'r-', label='Val', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE log10)')
            ax.set_title('Cutoff Physics Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            col += 1

        for c in range(col, 4):
            axes[phy_row, c].axis('off')

    # Gradient norm row
    if has_grad_norms:
        grad_row = 2 + kcl_ac_rows + (1 if has_physics else 0)
        ax = axes[grad_row, 0]
        ax.semilogy(train_epochs, max_grad_norms, 'r-', label='Max', alpha=0.7)
        ax.semilogy(train_epochs, avg_grad_norms, 'b-', label='Avg', alpha=0.7)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Clip=1.0')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms (before clip)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        for c in range(1, 4):
            axes[grad_row, c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved training curves to {output_path}")
