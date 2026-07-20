import argparse
import os

import pytest
import yaml

pytestmark = pytest.mark.unit


REQUIRED_KEYS = (
    "model_params",
    "trainer_params",
    "optimizer_params",
    "env",
    "tester_params",
)


def test_config_yaml_has_required_keys(project_root):
    path = os.path.join(project_root, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in REQUIRED_KEYS:
        assert key in cfg, f"missing {key}"


def test_config_model_feature_flags(project_root):
    with open(os.path.join(project_root, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    mp = cfg["model_params"]
    assert "use_gate" in mp
    assert "use_film" in mp
    assert "use_rope" in mp
    assert "K" in mp


def test_run_load_config_applies_batch_size(project_root, monkeypatch):
    monkeypatch.chdir(project_root)
    from run import load_config

    args = argparse.Namespace(batch_size=8, n_size=50)
    load_config(args)
    assert args.model_params["embedding_dim"] == 128
    assert args.env["generator_params"]["num_loc"] == 50
