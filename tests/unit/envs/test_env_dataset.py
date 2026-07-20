import os

import pytest
import torch

from envs.mtvrp import MTVRPEnv

pytestmark = pytest.mark.unit


def test_dataset_loads_npz(project_root, device):
    env = MTVRPEnv(
        generator_params={"variant_preset": "cvrp"},
        data_dir=os.path.join(project_root, "data"),
        test_size=[50],
        test_problem=["cvrp"],
        test_distribution=["uniform"],
        device=device,
    )
    datasets = env.dataset(phase="test")
    assert "50_cvrp_uniform" in datasets
    td = datasets["50_cvrp_uniform"]
    assert td.batch_size[0] > 0
    assert "locs" in td.keys()


def test_reset_has_no_nan(tiny_mtvrp_env, deterministic_td, device):
    td = tiny_mtvrp_env._reset(deterministic_td.to(device), batch_size=[deterministic_td.batch_size[0]])
    for key in ("locs", "demand_linehaul", "demand_backhaul"):
        assert torch.isfinite(td[key]).all()
