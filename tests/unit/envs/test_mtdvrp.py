import pytest
import torch

from envs.mtdvrp.env import get_starting_points
from tests.helpers.instances import build_mtd3_c5

pytestmark = [pytest.mark.unit, pytest.mark.mtdvrp]


def test_mtdvrp_select_start_nodes(tiny_mtdvrp_env, device):
    td = tiny_mtdvrp_env.generator(batch_size=1).to(device)
    num_starts, selected, _ = tiny_mtdvrp_env.select_start_nodes(td)
    n_depots = int(td["num_depots"][0].item())
    n_customers = td["locs"].shape[-2] - n_depots
    assert num_starts == n_depots * n_customers


def test_get_starting_points_tracks_depot_segments(device):
    actions = torch.tensor([[0, 3, 4, 1, 5, 2]], device=device)
    num_depots = torch.tensor([[3]], device=device)
    starts = get_starting_points(actions, num_depots)
    assert starts[0, 0].item() == 0
    assert starts[0, 3].item() == 1


def test_mtdvrp_first_step_only_depots(tiny_mtdvrp_env, device):
    td = build_mtd3_c5(device=device)
    td = tiny_mtdvrp_env._reset(td, batch_size=[1])
    mask = td["action_mask"]
    n_depots = int(td["num_depots"][0].item())
    assert mask[0, :n_depots].all()
    assert not mask[0, n_depots:].any()
