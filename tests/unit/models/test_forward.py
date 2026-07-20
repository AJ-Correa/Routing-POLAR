import pytest
import torch

pytestmark = pytest.mark.unit


def test_forward_output_keys(random_model, tiny_mtvrp_env, deterministic_td, device):
    td = deterministic_td.to(device)
    tiny_mtvrp_env.to(device)
    random_model.eval()
    with torch.no_grad():
        out = random_model(td, tiny_mtvrp_env, gate_alpha=1.0)
    assert "reward" in out
    assert "log_likelihood" in out
    assert "tours" in out
    assert out["log_likelihood"].dim() == 2


def test_set_loss_mode_po_affects_starts(tiny_mtvrp_env, deterministic_td):
    tiny_mtvrp_env.set_loss_mode("po")
    n_po, _, _ = tiny_mtvrp_env.select_start_nodes(deterministic_td, po_B=2)
    tiny_mtvrp_env.set_loss_mode("rl")
    n_rl, _, _ = tiny_mtvrp_env.select_start_nodes(deterministic_td)
    assert n_po == 2
    assert n_rl == deterministic_td["locs"].shape[-2] - 1
