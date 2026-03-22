#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch


_EPS = 1e-6

def solve_scale_shift(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Solve per-image least squares alignment: target ~= scale * pred + shift.
    Inputs are shaped (B, 1, H, W). Returns aligned pred in the same shape.
    """
    if pred.ndim != 4 or target.ndim != 4 or valid_mask.ndim != 4:
        raise ValueError("pred/target/valid_mask must be 4D tensors (B,1,H,W).")

    aligned = pred.clone()
    batch_size = pred.shape[0]

    for b in range(batch_size):
        mask = valid_mask[b, 0]
        if mask.sum().item() < 16:
            continue

        x = pred[b, 0][mask]
        y = target[b, 0][mask]

        # Normal equations for [x, 1] * [scale, shift]^T ~= y
        a00 = torch.sum(x * x)
        a01 = torch.sum(x)
        a11 = torch.sum(torch.ones_like(x))
        b0 = torch.sum(x * y)
        b1 = torch.sum(y)

        det = a00 * a11 - a01 * a01
        if torch.abs(det) < _EPS:
            continue

        scale = (a11 * b0 - a01 * b1) / det
        shift = (-a01 * b0 + a00 * b1) / det
        aligned[b, 0] = scale * pred[b, 0] + shift

    return torch.clamp(aligned, min=1e-3)


def abs_rel_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Compute AbsRel over valid pixels.
    All tensors are (B, 1, H, W).
    """
    if pred.shape != target.shape or pred.shape != valid_mask.shape:
        raise ValueError("pred/target/valid_mask must share the same shape.")

    mask = valid_mask.bool()
    if mask.sum().item() == 0:
        return float("nan")

    p = torch.clamp(pred[mask], min=1e-3)
    t = torch.clamp(target[mask], min=1e-3)
    return (torch.abs(p - t) / t).mean().item()



def to_numpy_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: float(np.asarray(v)) for k, v in metrics.items()}
