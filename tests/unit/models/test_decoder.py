import pytest
import torch
import torch.nn.functional as F

from models.decoder import VRP_Decoder
from models.helpers import PrecomputedCache, reshape_by_heads
from models.model import VRPModel
from utils.functions import batchify, gather_by_index

pytestmark = pytest.mark.unit


def _decoder_forward(decoder, td, cache, num_starts, gate_alpha=1.0):
    logprobs, mask, _ = decoder(td, cache, num_starts, gate_alpha=gate_alpha)
    return logprobs, mask


def test_decoder_masks_infeasible_logits(model_params, deterministic_td, tiny_mtvrp_env, device):
    decoder = VRP_Decoder(**model_params).to(device)
    embed_dim = model_params["embedding_dim"]
    n = deterministic_td["locs"].shape[-2]
    b = deterministic_td.batch_size[0]
    node_embed = torch.randn(b, n, embed_dim, device=device)
    node_coords = deterministic_td["locs"]

    k = reshape_by_heads(decoder.Wk(node_embed), head_num=model_params["head_num"])
    v = reshape_by_heads(decoder.Wv(node_embed), head_num=model_params["head_num"])
    cache = PrecomputedCache(node_embed, k, v, node_embed.transpose(1, 2), node_coords)

    td = tiny_mtvrp_env._reset(deterministic_td, batch_size=[b])
    td = batchify(td, 2)
    logprobs, mask = _decoder_forward(decoder, td, cache, num_starts=2)
    assert torch.isneginf(logprobs[~mask]).all()
    assert torch.isfinite(logprobs[mask]).all()


def test_greedy_never_selects_masked_action(model_params, deterministic_td, tiny_mtvrp_env, device):
    decoder = VRP_Decoder(**model_params).to(device)
    embed_dim = model_params["embedding_dim"]
    n = deterministic_td["locs"].shape[-2]
    b = deterministic_td.batch_size[0]
    node_embed = torch.randn(b, n, embed_dim, device=device)
    k = reshape_by_heads(decoder.Wk(node_embed), head_num=model_params["head_num"])
    v = reshape_by_heads(decoder.Wv(node_embed), head_num=model_params["head_num"])
    cache = PrecomputedCache(node_embed, k, v, node_embed.transpose(1, 2), deterministic_td["locs"])

    td = tiny_mtvrp_env._reset(deterministic_td, batch_size=[b])
    td = batchify(td, 2)
    logprobs, mask = _decoder_forward(decoder, td, cache, num_starts=2)
    selected = VRPModel.greedy(logprobs, mask)
    assert mask.gather(1, selected.unsqueeze(1)).all()


def test_gate_alpha_zero_matches_baseline(model_params, deterministic_td, tiny_mtvrp_env, device):
    params = {**model_params, "use_gate": True}
    decoder = VRP_Decoder(**params).to(device)
    embed_dim = model_params["embedding_dim"]
    n = deterministic_td["locs"].shape[-2]
    b = deterministic_td.batch_size[0]
    node_embed = torch.randn(b, n, embed_dim, device=device)
    k = reshape_by_heads(decoder.Wk(node_embed), head_num=model_params["head_num"])
    v = reshape_by_heads(decoder.Wv(node_embed), head_num=model_params["head_num"])
    cache = PrecomputedCache(node_embed, k, v, node_embed.transpose(1, 2), deterministic_td["locs"])
    td = batchify(tiny_mtvrp_env._reset(deterministic_td, batch_size=[b]), 2)

    lp0, _ = _decoder_forward(decoder, td, cache, num_starts=2, gate_alpha=0.0)
    lp1, _ = _decoder_forward(decoder, td, cache, num_starts=2, gate_alpha=1.0)
    assert not torch.allclose(lp0, lp1)
