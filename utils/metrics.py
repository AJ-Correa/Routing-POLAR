"""Gap metrics vs reference costs (PyVRP .npz scale: round to 3 decimals)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def gap_percent(
    obtained,
    ref,
    decimals: int = 3,
) -> np.ndarray:
    """Gap (%) vs reference; 0 where ``round(obtained, decimals) == round(ref, decimals)``."""
    obtained = np.asarray(obtained, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    agree = np.round(obtained, decimals) == np.round(ref, decimals)
    gap = (obtained - ref) * 100.0 / np.maximum(ref, 1e-30)
    return np.where(agree, 0.0, gap)


def gap_percent_scalar(obtained: float, ref: float, decimals: int = 3) -> float:
    if ref <= 1e-9:
        return float("nan")
    return float(gap_percent(obtained, ref, decimals=decimals))


def gap_percent_torch(
    obtained: Tensor,
    ref: Tensor,
    decimals: int = 3,
) -> Tensor:
    """Element-wise gap (%); 0 where costs agree at ``decimals``."""
    scale = 10.0**decimals
    agree = torch.round(obtained * scale) == torch.round(ref * scale)
    gap = (obtained - ref) * 100.0 / ref.clamp(min=1e-30)
    return torch.where(agree, torch.zeros_like(gap), gap)


def gap_percent_mean_torch(
    obtained: Tensor,
    ref: Tensor,
    decimals: int = 3,
) -> float:
    return float(gap_percent_torch(obtained, ref, decimals=decimals).mean().item())
