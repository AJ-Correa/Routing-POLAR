import math

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from utils.functions import (
    batchify,
    clip_grad_norms,
    gather_by_index,
    get_distance,
    load_npz_to_tensordict,
    unbatchify,
)


pytestmark = pytest.mark.unit


def test_gather_by_index_matches_manual():
    src = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    idx = torch.tensor([[0, 2], [1, 0]])
    out = gather_by_index(src, idx, dim=1, squeeze=False)
    assert out.shape == (2, 2, 2)
    assert torch.equal(out[0, 0], src[0, 0])
    assert torch.equal(out[1, 1], src[1, 0])


def test_get_distance_l2():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    y = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    d = get_distance(x, y)
    assert d.tolist() == pytest.approx([5.0, 5.0])



def test_clip_grad_norms_caps_norm():
    p = torch.nn.Parameter(torch.ones(4) * 10.0)
    p.grad = torch.ones_like(p.data) * 3.0
    opt = torch.optim.SGD([p], lr=0.1)
    _, clipped = clip_grad_norms(opt.param_groups, max_norm=1.0)
    assert float(clipped[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(p.grad.norm().item()) == pytest.approx(1.0, abs=1e-5)


def test_load_npz_to_tensordict_shapes(project_root):
    path = f"{project_root}/data/cvrp/test/50.npz"
    if not __import__("os").path.isfile(path):
        pytest.skip("test npz not available")
    td = load_npz_to_tensordict(path)
    assert td.batch_size[0] > 0
    assert "locs" in td.keys()
    assert td["locs"].dim() >= 3
