"""
Metrics computation utilities for GNN training.

This module provides functions for computing accuracy and MAE metrics
for voltage and current predictions.
"""

from typing import List, Tuple

import numpy as np
import torch


def compute_voltage_accuracy(
    errors_mv: np.ndarray,
    thresholds: List[float] = [80, 50, 20, 10]
) -> Tuple[float, ...]:
    """
    Compute voltage accuracy at different thresholds.

    Args:
        errors_mv: Array of absolute errors in mV
        thresholds: List of threshold values in mV

    Returns:
        Tuple of accuracy percentages for each threshold
    """
    return tuple((errors_mv < t).mean() * 100 for t in thresholds)


def compute_current_accuracy(
    errors_ua: np.ndarray,
    thresholds: List[float] = [50, 20, 5, 2]
) -> Tuple[float, ...]:
    """
    Compute current accuracy at different thresholds.

    Args:
        errors_ua: Array of absolute errors in µA
        thresholds: List of threshold values in µA

    Returns:
        Tuple of accuracy percentages for each threshold
    """
    if len(errors_ua) == 0:
        return tuple(0.0 for _ in thresholds)
    return tuple((errors_ua < t).mean() * 100 for t in thresholds)


def denormalize_voltage(
    pred: torch.Tensor,
    target: torch.Tensor,
    vdc_mean: float,
    vdc_std: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Denormalize voltage predictions to mV.

    Args:
        pred: Normalized predictions
        target: Normalized targets
        vdc_mean: VDC mean for denormalization
        vdc_std: VDC std for denormalization

    Returns:
        Tuple of (pred_mv, target_mv)
    """
    pred_mv = (pred * vdc_std + vdc_mean) * 1000
    target_mv = (target * vdc_std + vdc_mean) * 1000
    return pred_mv, target_mv


def denormalize_current(
    pred: torch.Tensor,
    target: torch.Tensor,
    current_mean: float,
    current_std: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Denormalize current predictions from log z-score to original scale.

    Args:
        pred: Normalized predictions (log z-score)
        target: Normalized targets (log z-score)
        current_mean: Log current mean for denormalization
        current_std: Log current std for denormalization

    Returns:
        Tuple of (pred_original, target_original) in Amps
    """
    pred_orig = torch.pow(10, pred * current_std + current_mean)
    target_orig = torch.pow(10, target * current_std + current_mean)
    return pred_orig, target_orig
