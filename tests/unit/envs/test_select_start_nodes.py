import pytest
import torch

pytestmark = pytest.mark.unit


def test_select_start_nodes_default_count(tiny_mtvrp_env, deterministic_td):
    num_starts, selected, greedy_mask = tiny_mtvrp_env.select_start_nodes(deterministic_td)
    n_customers = deterministic_td["locs"].shape[-2] - 1
    assert num_starts == n_customers
    assert selected.numel() == n_customers * deterministic_td.batch_size[0]
    assert not greedy_mask.any()


def test_select_start_nodes_po_b_subset(tiny_mtvrp_env, deterministic_td):
    tiny_mtvrp_env.set_loss_mode("po")
    num_starts, _, _ = tiny_mtvrp_env.select_start_nodes(deterministic_td, po_B=2)
    assert num_starts == 2


def test_select_start_nodes_with_greedy(tiny_mtvrp_env, deterministic_td):
    num_starts, selected, greedy_mask = tiny_mtvrp_env.select_start_nodes(
        deterministic_td, with_greedy=True
    )
    n_customers = deterministic_td["locs"].shape[-2] - 1
    assert num_starts == n_customers + 1
    assert greedy_mask.sum().item() == deterministic_td.batch_size[0]
