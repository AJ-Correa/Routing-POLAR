import pytest
import torch

from envs.mtvrp import MTVRPEnv
from tests.helpers.instances import manual_tour_cost
from tests.helpers.policy import reset_with_prompt, sample_policy_tour

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("variant", ["cvrp", "ovrp"])
def test_mtvrp_reward_matches_manual_cost(random_model, device, variant):
    env = MTVRPEnv(
        generator_params={"num_loc": 4, "variant_preset": variant},
        device=device,
        check_solution=False,
        seed=0,
    )
    td = reset_with_prompt(env, batch_size=1, device=device)
    tour, reward, _ = sample_policy_tour(random_model, env, td)
    open_route = bool(td["open_route"][0, 0].item())
    actions = tour.tolist()
    expected = manual_tour_cost(td["locs"][0].cpu(), actions, open_route=open_route)
    assert -float(reward) == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_mtvrp_reward_open_excludes_return_leg(random_model, device):
    """Same policy tour: open_route=True must not charge the return-to-depot leg."""
    env_closed = MTVRPEnv(
        generator_params={"num_loc": 4, "variant_preset": "cvrp"},
        device=device,
        check_solution=False,
        seed=0,
    )
    td = reset_with_prompt(env_closed, batch_size=1, device=device)
    tour, _, _ = sample_policy_tour(random_model, env_closed, td)
    actions = tour.unsqueeze(0)

    td_closed = td.clone()
    td_closed["open_route"] = torch.zeros_like(td_closed["open_route"])
    td_open = td.clone()
    td_open["open_route"] = torch.ones_like(td_open["open_route"], dtype=torch.bool)

    closed_r, _ = env_closed.get_reward(td_closed, actions)
    open_r, _ = env_closed.get_reward(td_open, actions)
    # Open drops depot-return legs → higher reward (lower cost), or equal if no return
    assert open_r.item() >= closed_r.item()
    if (actions == 0).any():
        assert open_r.item() > closed_r.item()
