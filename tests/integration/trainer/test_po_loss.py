from types import SimpleNamespace

import pytest
import torch

from trainer import VRPTrainer

pytestmark = pytest.mark.integration


class _POStub:
    def __init__(self, alpha=0.07):
        self.args = SimpleNamespace(trainer_params={"po_alpha": alpha})


def test_compute_po_loss_prefers_better_solution():
    stub = _POStub(alpha=0.1)
    reward = torch.tensor([[-10.0, -5.0, -8.0]])
    log_likelihood = torch.tensor([[-1.0, -1.0, -1.0]])
    loss = VRPTrainer._compute_po_loss(stub, reward, log_likelihood)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_compute_po_loss_all_equal_rewards():
    stub = _POStub(alpha=0.1)
    reward = torch.tensor([[-5.0, -5.0]])
    log_likelihood = torch.tensor([[-1.0, -2.0]])
    loss = VRPTrainer._compute_po_loss(stub, reward, log_likelihood)
    assert loss.item() == pytest.approx(0.0)
