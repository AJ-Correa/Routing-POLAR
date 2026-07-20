"""Shared helpers for policy-sampled feasible tours."""

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type


def reset_with_prompt(env, batch_size=1, device="cpu"):
    """Generate, reset, and restore p_s_tag (dropped by reset)."""
    td = env.generator(batch_size=batch_size).to(device)
    p_s_tag = td["p_s_tag"].clone() if "p_s_tag" in td.keys() else None
    td = env.reset(td)
    if p_s_tag is not None:
        td["p_s_tag"] = p_s_tag
    return td


def sample_policy_tour(model, env, td):
    """Greedy/POMO rollout; return best tour, its reward, and full model out."""
    model.eval()
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        out = model(td, env, gate_alpha=1.0)
    idx = int(out["reward"].argmax())
    return out["tours"][idx], out["reward"][idx], out
