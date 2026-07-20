import numpy as np
import pytest
import torch

from search.search import Search
from tests.helpers.instances import actions_to_search_tour
from tests.helpers.policy import reset_with_prompt, sample_policy_tour

pytestmark = pytest.mark.unit


def _search_from_td(td, open_route=None):
    open_flag = (
        bool(td["open_route"][0, 0].item()) if open_route is None else open_route
    )
    dist_lim = float(td["distance_limit"][0, 0].item())
    return Search(
        locs=td["locs"][0].detach().cpu().numpy(),
        demands_linehauls=td["demand_linehaul"][0].detach().cpu().numpy(),
        demands_backhauls=td["demand_backhaul"][0].detach().cpu().numpy(),
        distance_limit=dist_lim if dist_lim < 1e8 else 1e9,
        open_route=open_flag,
        time_windows=td["time_windows"][0].detach().cpu().numpy(),
        service_times=td["service_time"][0].detach().cpu().numpy(),
        nb_granular=5,
    )


def test_search_tour_cost_finite(tiny_mtvrp_env, random_model, device):
    td = reset_with_prompt(tiny_mtvrp_env, batch_size=1, device=device)
    tour, _, _ = sample_policy_tour(random_model, tiny_mtvrp_env, td)
    search = _search_from_td(td)
    cost = search.tour_cost(actions_to_search_tour(tour.tolist()))
    assert np.isfinite(cost)
    assert cost > 0


def test_search_tour_cost_deterministic(tiny_mtvrp_env, random_model, device):
    td = reset_with_prompt(tiny_mtvrp_env, batch_size=1, device=device)
    tour, _, _ = sample_policy_tour(random_model, tiny_mtvrp_env, td)
    search = _search_from_td(td)
    seq = actions_to_search_tour(tour.tolist())
    assert search.tour_cost(seq) == pytest.approx(search.tour_cost(seq))


@pytest.mark.slow
def test_search_ls_improves_or_equal(tiny_mtvrp_env, random_model, device):
    td = reset_with_prompt(tiny_mtvrp_env, batch_size=1, device=device)
    tour, _, _ = sample_policy_tour(random_model, tiny_mtvrp_env, td)
    search = _search_from_td(td)
    seq = actions_to_search_tour(tour.tolist())
    before = search.tour_cost(seq)
    improved = search.build_solution(seq, seed=0)
    after = search.tour_cost(improved)
    assert after <= before + 1e-6
