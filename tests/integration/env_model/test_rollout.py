import pytest
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from utils.functions import batchify

pytestmark = pytest.mark.integration


def test_rollout_completes_and_feasible(random_model, tiny_mtvrp_env, deterministic_td, device):
    td = deterministic_td.to(device)
    tiny_mtvrp_env.to(device)
    random_model.eval()
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        out = random_model(td, tiny_mtvrp_env, gate_alpha=1.0)

    batch = td.batch_size[0]
    num_starts = td["locs"].shape[-2] - 1
    assert out["reward"].numel() == batch * num_starts
    assert torch.isfinite(out["reward"]).all()

    # Flat [B * S, T] -> [S, B, T] matching trainer convention
    tours = out["tours"].view(num_starts, batch, -1)
    reward = out["reward"].view(num_starts, batch)
    assert tours.shape[:2] == (num_starts, batch)
    assert reward.shape == (num_starts, batch)


def test_rollout_reward_matches_env_get_reward(random_model, tiny_mtvrp_env, deterministic_td, device):
    td = deterministic_td.to(device)
    tiny_mtvrp_env.to(device)
    random_model.eval()
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        out = random_model(td, tiny_mtvrp_env, gate_alpha=1.0)

    batch = td.batch_size[0]
    num_starts = out["tours"].shape[0] // batch
    td_exp = batchify(td, num_starts)
    rew2, _ = tiny_mtvrp_env.get_reward(td_exp, out["tours"])
    assert torch.allclose(
        out["reward"].reshape(-1).float(),
        rew2.reshape(-1).float(),
        rtol=1e-5,
        atol=1e-6,
    )
