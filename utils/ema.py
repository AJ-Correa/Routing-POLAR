import torch
import copy
from typing import Optional

class ModelEMA:
    """
    Exponential Moving Average of model weights for improved generalization.
    
    The EMA model is a smoothed version of the training model that often
    generalizes better at test time.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        decay: float = 0.999,
        warmup_steps: int = 0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (0.999 is typical, higher = slower update)
            warmup_steps: Number of steps before starting EMA (useful to let model stabilize)
            device: Device for EMA model (defaults to same as source model)
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Create EMA model as a deep copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        if device is not None:
            self.ema_model.to(device)
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Update EMA weights after each training step."""
        self.step_count += 1
        
        # Skip update during warmup
        if self.step_count <= self.warmup_steps:
            # During warmup, just copy weights directly
            for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.copy_(p.data)
            return
        
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            # EMA update: ema_p = decay * ema_p + (1 - decay) * p
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        
        # Also update buffers (e.g., BatchNorm running stats)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)
    
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay,
            'step_count': self.step_count,
            'warmup_steps': self.warmup_steps,
        }
    
    def load_state_dict(self, state_dict):
        """Restore EMA state from checkpoint."""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.decay = state_dict['decay']
        self.step_count = state_dict['step_count']
        self.warmup_steps = state_dict['warmup_steps']