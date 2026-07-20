import numpy as np
import pytest
import torch

from utils.metrics import gap_percent, gap_percent_mean_torch, gap_percent_scalar, gap_percent_torch

pytestmark = pytest.mark.unit


def test_gap_percent_zero_when_equal():
    gap = gap_percent(100.0, 100.0)
    assert gap == pytest.approx(0.0)


def test_gap_percent_positive_when_worse():
    gap = gap_percent(110.0, 100.0)
    assert gap == pytest.approx(10.0)


def test_gap_percent_rounding_boundary():
    gap = gap_percent(1.0004, 1.0000, decimals=3)
    assert gap == pytest.approx(0.0)


def test_gap_percent_torch_matches_scalar():
    obtained = torch.tensor([110.0, 100.0])
    ref = torch.tensor([100.0, 100.0])
    batch = gap_percent_torch(obtained, ref)
    assert batch[0].item() == pytest.approx(10.0)
    assert batch[1].item() == pytest.approx(0.0)
    assert gap_percent_scalar(110.0, 100.0) == pytest.approx(10.0)
    assert gap_percent_mean_torch(obtained, ref) == pytest.approx(5.0)
