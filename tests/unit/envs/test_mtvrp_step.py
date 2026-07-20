import pytest
import torch

from tests.helpers.policy import reset_with_prompt, sample_policy_tour

pytestmark = pytest.mark.unit


def test_step_updates_capacity_and_time(tiny_mtvrp_env, random_model, device):
    td = reset_with_prompt(tiny_mtvrp_env, batch_size=1, device=device)
    tour, _, _ = sample_policy_tour(random_model, tiny_mtvrp_env, td)
    first = int(tour[0].item())
    if first == 0:
        first = int(tour[1].item())

    td = reset_with_prompt(tiny_mtvrp_env, batch_size=1, device=device)
    demand = float(td["demand_linehaul"][0, first].item())
    td.set("action", torch.tensor([first], device=device))
    td = tiny_mtvrp_env._step(td)
    assert td["used_capacity_linehaul"].item() == pytest.approx(demand)
    assert td["current_time"].item() >= 0.0


def test_per_step_reward_is_zero(tiny_mtvrp_env, random_model, device):
    td = reset_with_prompt(tiny_mtvrp_env, batch_size=1, device=device)
    tour, _, _ = sample_policy_tour(random_model, tiny_mtvrp_env, td)
    first = int(tour[0].item())
    if first == 0:
        first = int(tour[1].item())

    td = reset_with_prompt(tiny_mtvrp_env, batch_size=1, device=device)
    td.set("action", torch.tensor([first], device=device))
    td = tiny_mtvrp_env._step(td)
    assert td["reward"].item() == 0.0
