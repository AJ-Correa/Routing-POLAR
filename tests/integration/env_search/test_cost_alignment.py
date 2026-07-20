from itertools import product

import pytest

from envs.mtvrp import MTVRPEnv
from search.search import Search
from tests.helpers.instances import actions_to_search_tour
from tests.helpers.policy import reset_with_prompt, sample_policy_tour

pytestmark = pytest.mark.integration

# (O, TW, L, B) -> variant_preset
_VARIANT = {
    (False, False, False, False): "cvrp",
    (True, False, False, False): "ovrp",
    (False, False, False, True): "vrpb",
    (False, False, True, False): "vrpl",
    (False, True, False, False): "vrptw",
    (True, True, False, False): "ovrptw",
    (True, False, False, True): "ovrpb",
    (True, False, True, False): "ovrpl",
    (False, False, True, True): "vrpbl",
    (False, True, False, True): "vrpbtw",
    (False, True, True, False): "vrpltw",
    (True, False, True, True): "ovrpbl",
    (True, True, False, True): "ovrpbtw",
    (True, True, True, False): "ovrpltw",
    (False, True, True, True): "vrpbltw",
    (True, True, True, True): "ovrpbltw",
}


@pytest.mark.parametrize(
    "open_route,has_tw,has_limit,has_backhaul",
    list(product([False, True], repeat=4)),
    ids=[_VARIANT[k] for k in product([False, True], repeat=4)],
)
def test_search_tour_cost_matches_env(
    random_model, device, open_route, has_tw, has_limit, has_backhaul
):
    variant = _VARIANT[(open_route, has_tw, has_limit, has_backhaul)]
    env = MTVRPEnv(
        generator_params={"num_loc": 4, "variant_preset": variant},
        device=device,
        check_solution=False,
        seed=0,
    )
    td = reset_with_prompt(env, batch_size=1, device=device)
    tour, reward, _ = sample_policy_tour(random_model, env, td)
    env_cost = -float(reward)

    open_flag = bool(td["open_route"][0, 0].item())
    dist_lim = float(td["distance_limit"][0, 0].item())
    search = Search(
        locs=td["locs"][0].detach().cpu().numpy(),
        demands_linehauls=td["demand_linehaul"][0].detach().cpu().numpy(),
        demands_backhauls=td["demand_backhaul"][0].detach().cpu().numpy(),
        distance_limit=dist_lim if dist_lim < 1e8 else 1e9,
        open_route=open_flag,
        time_windows=td["time_windows"][0].detach().cpu().numpy(),
        service_times=td["service_time"][0].detach().cpu().numpy(),
        nb_granular=5,
    )
    search_cost = search.tour_cost(actions_to_search_tour(tour.tolist()))
    assert search_cost != float("inf"), f"Search infeasible for {variant}"
    assert env_cost == pytest.approx(search_cost, rel=1e-3, abs=1e-4)
