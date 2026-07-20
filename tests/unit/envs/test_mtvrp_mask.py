import pytest
import torch

from envs.mtvrp import MTVRPEnv
from tests.helpers.policy import reset_with_prompt, sample_policy_tour

pytestmark = pytest.mark.unit


VARIANTS = ["cvrp", "ovrp", "vrptw", "vrpb"]


@pytest.fixture(params=VARIANTS)
def variant_env(request, device):
    return MTVRPEnv(
        generator_params={"num_loc": 4, "variant_preset": request.param},
        device=device,
        check_solution=False,
        seed=1,
    )


def test_mask_blocks_visited_nodes(variant_env, random_model, device):
    td = reset_with_prompt(variant_env, batch_size=1, device=device)
    tour, _, _ = sample_policy_tour(random_model, variant_env, td)
    # First customer in the sampled tour must become masked after visiting
    first_cust = int(tour[0].item())
    if first_cust == 0:
        first_cust = int(tour[1].item())
    td = reset_with_prompt(variant_env, batch_size=1, device=device)
    td.set("action", torch.tensor([first_cust], device=device))
    td = variant_env._step(td)
    assert not td["action_mask"][0, first_cust]


def test_policy_tour_completes_episode(tiny_mtvrp_env, random_model, device):
    td = reset_with_prompt(tiny_mtvrp_env, batch_size=1, device=device)
    tour, reward, _ = sample_policy_tour(random_model, tiny_mtvrp_env, td)
    n_customers = td["locs"].shape[-2] - 1
    visited_customers = {int(x) for x in tour.tolist() if int(x) > 0}
    assert len(visited_customers) == n_customers
    assert torch.isfinite(reward)


def test_backhaul_class1_linehaul_before_backhaul_in_policy_tour(random_model, device):
    env = MTVRPEnv(
        generator_params={"num_loc": 4, "variant_preset": "vrpb"},
        device=device,
        check_solution=False,
        seed=2,
    )
    td = reset_with_prompt(env, batch_size=1, device=device)
    # At depot start, only linehaul customers should be feasible
    mask = env.get_action_mask(td)
    linehaul = (td["demand_linehaul"][0] > 0).nonzero(as_tuple=False).view(-1)
    backhaul = (td["demand_backhaul"][0] > 0).nonzero(as_tuple=False).view(-1)
    if linehaul.numel() and backhaul.numel():
        assert mask[0, linehaul].any()

    tour, _, _ = sample_policy_tour(random_model, env, td)
    seq = [int(x) for x in tour.tolist() if int(x) > 0]
    linehaul_set = set(linehaul.tolist())
    backhaul_set = set(backhaul.tolist())
    first_backhaul_pos = next(
        (i for i, n in enumerate(seq) if n in backhaul_set), len(seq)
    )
    assert all(n in linehaul_set or n not in backhaul_set for n in seq[:first_backhaul_pos])


def test_tw_mask_blocks_late_arrival(random_model, device):
    env = MTVRPEnv(
        generator_params={"num_loc": 4, "variant_preset": "vrptw"},
        device=device,
        check_solution=False,
        seed=3,
    )
    td = reset_with_prompt(env, batch_size=1, device=device)
    # Push clock past all late windows so no customer is reachable
    late_max = td["time_windows"][0, :, 1].max()
    td.set("current_time", (late_max + 1.0).view(1, 1))
    td.set("current_node", torch.zeros(1, dtype=torch.long, device=device))
    mask = env.get_action_mask(td)
    assert not mask[0, 1:].any()
