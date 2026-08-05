import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.mtvrp import MTVRPEnv
from envs.mtdvrp.env import MTVRPEnv as MTDVRPEnv
from models.model import VRPModel
from utils.functions import get_torch_device


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def seed():
    torch.manual_seed(0)
    np.random.seed(0)
    return 0


@pytest.fixture
def device():
    """Same policy as run.py: CUDA if available, otherwise CPU."""
    dev = str(get_torch_device())
    torch.set_default_device(torch.device(dev))
    return dev


@pytest.fixture
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def model_params():
    dim = 128
    return {
        "embedding_dim": dim,
        "encoder_layer_num": 2,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 512,
        "ffd": "siglu",
        "norm_type": "rms",
        "p_num": 6,
        "logit_clipping": 10,
        "K": 2,
        "use_gate": True,
        "use_ple": False,
        "sqrt_embedding_dim": dim**0.5,
    }


@pytest.fixture
def tiny_mtvrp_env(device):
    return MTVRPEnv(
        generator_params={"num_loc": 4, "variant_preset": "cvrp"},
        device=device,
        check_solution=False,
        seed=0,
    )


@pytest.fixture
def tiny_mtdvrp_env(device):
    return MTDVRPEnv(
        generator_params={"num_loc": 5, "num_depots": 3, "variant_preset": "mdcvrp"},
        device=device,
        check_solution=False,
        seed=0,
    )


@pytest.fixture
def deterministic_td(tiny_mtvrp_env, seed, device):
    td = tiny_mtvrp_env.generator(batch_size=2).to(device)
    p_s_tag = td["p_s_tag"].clone() if "p_s_tag" in td.keys() else None
    td = tiny_mtvrp_env.reset(td)
    if p_s_tag is not None:
        td["p_s_tag"] = p_s_tag
    return td


@pytest.fixture
def model_args(model_params, device):
    logger = logging.getLogger("test")
    return SimpleNamespace(
        model_params=model_params,
        trainer_params={"po_alpha": 0.07, "po_B": 2, "loss_function": "po"},
        log=logger.info,
        device=device,
    )


@pytest.fixture
def random_model(model_args, device):
    model = VRPModel(model_args)
    model.to(device)
    model.eval()
    return model
