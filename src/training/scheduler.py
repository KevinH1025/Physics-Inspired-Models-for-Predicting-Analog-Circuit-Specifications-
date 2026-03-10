"""
Learning rate scheduler utilities for GNN training.
"""

import torch
from typing import Optional


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    plateau_factor: float = 0.5,
    plateau_patience: int = 10,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: 'cosine', 'plateau', or 'none'
        epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        plateau_factor: Factor to reduce LR on plateau
        plateau_patience: Patience for plateau scheduler

    Returns:
        Scheduler instance or None
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )
    return None


def apply_warmup(optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int, base_lr: float) -> None:
    """Apply linear warmup to learning rate."""
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor


def step_scheduler(
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scheduler_type: str,
    val_loss: float = None
) -> None:
    """Step the scheduler appropriately based on type."""
    if scheduler is None:
        return
    if scheduler_type == 'plateau':
        scheduler.step(val_loss)
    else:
        scheduler.step()
