import pytest
import torch

from models.encoder import VRP_Encoder
from models.encoder_ple import VRP_Encoder as VRP_Encoder_PLE

pytestmark = pytest.mark.unit


def test_encoder_output_shape(model_params, deterministic_td, device):
    encoder = VRP_Encoder(**model_params).to(device)
    out, coords = encoder(deterministic_td.to(device))
    b, n = deterministic_td.batch_size[0], deterministic_td["locs"].shape[-2]
    assert out.shape == (b, n, model_params["embedding_dim"])
    assert coords.shape == deterministic_td["locs"].shape


def test_encoder_nan_free(model_params, deterministic_td, device):
    encoder = VRP_Encoder(**model_params).to(device)
    out, _ = encoder(deterministic_td.to(device))
    assert torch.isfinite(out).all()


def test_ple_expert_count(model_params, deterministic_td, device):
    params = {**model_params, "use_ple": True, "K": 3}
    dim = params["embedding_dim"]
    encoder = VRP_Encoder_PLE(**params).to(device)
    b = deterministic_td.batch_size[0]
    prompt = torch.zeros(b, params["p_num"], dim, device=device)
    out, _ = encoder(deterministic_td.to(device), prompt=prompt)
    assert out.shape[-1] == dim
    assert len(encoder.ple_layers) == params["encoder_layer_num"]
