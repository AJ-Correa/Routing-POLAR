import math
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

from models.helpers import *


class RoPE2D(nn.Module):
    """2D Rotary Positional Embeddings for spatial coordinates."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.base = base
        self.dim_per_axis = head_dim // 2

        # Precompute inverse frequencies for each axis
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _compute_rotation(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute cos and sin rotation matrices from 2D coordinates."""
        batch, seq_len, _ = coords.shape

        # Scale coordinates to reasonable range for rotation
        x = coords[..., 0:1] * 2 * math.pi  # (batch, seq_len, 1)
        y = coords[..., 1:2] * 2 * math.pi  # (batch, seq_len, 1)

        # Compute rotation angles for each frequency
        freqs = self.inv_freq.to(coords.device)

        # Compute angles: position * frequency
        x_angles = x * freqs.view(1, 1, -1)
        y_angles = y * freqs.view(1, 1, -1)

        # Interleave x and y angles to fill head_dim
        x_cos_sin = torch.cat([x_angles, x_angles], dim=-1)
        y_cos_sin = torch.cat([y_angles, y_angles], dim=-1)

        # Combine x and y rotations
        angles = torch.cat([x_cos_sin, y_cos_sin], dim=-1)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(
        self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply 2D rotary embeddings to queries and keys."""
        batch, num_heads, seq_len, head_dim = q.shape

        # Expand for multi-head: (batch, 1, seq_len, head_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Apply rotation: x * cos + rotate_half(x) * sin
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot


class PromptNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Extended to 6 constraints: O, TW, L, B, MB, MD
        input_dim = 6
        output_dim = args.model_params["embedding_dim"]
        self.logit_clipping = args.model_params["logit_clipping"]
        self.p_num = args.model_params.get("p_num", 6)

        layer1 = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.uniform_(layer1.weight)
        self.model = nn.Sequential(
            layer1,
            nn.LayerNorm(output_dim),
            linear_layer(output_dim, output_dim),
            nn.ReLU(),
            linear_layer(output_dim, output_dim // 8),  # task embedding
            nn.LayerNorm(output_dim // 8),
            linear_layer(output_dim // 8, self.p_num * output_dim),
        )

    def forward(self, td):
        # Extract indices 1:7 for O, TW, L, B, MB, MD
        prompt_input = td["p_s_tag"][:, 1:7]
        return {
            "prompt": self.model(prompt_input).view(td.batch_size[0], self.p_num, -1)
        }


class FiLMGenerator(nn.Module):
    """FiLM generator for constraint-conditioned embeddings, following Corrêa et al. (2026)."""

    def __init__(self, num_constraints: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        output_dim = embedding_dim * 2

        self.film_net = nn.Sequential(
            nn.Linear(num_constraints, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim),
        )

        # Initialize to identity: gamma=1, beta=0
        self._init_identity()

    def _init_identity(self):
        """Initialize with gamma=1, beta=0."""
        with torch.no_grad():
            last_layer = self.film_net[-1]
            nn.init.zeros_(last_layer.weight)
            last_layer.bias.data[: self.embedding_dim] = 1.0
            last_layer.bias.data[self.embedding_dim :] = 0.0

    def forward(self, constraint_flags: Tensor) -> tuple:
        """Generate FiLM parameters: gamma for scaling, beta for shifting."""
        params = self.film_net(constraint_flags)
        # Split into gamma (scale) and beta (shift)
        gamma = params[:, : self.embedding_dim].unsqueeze(1)  # (batch, 1, embed_dim)
        beta = params[:, self.embedding_dim :].unsqueeze(1)  # (batch, 1, embed_dim)
        return gamma, beta


class AddAndNorm(nn.Module):
    """Residual connection with normalization."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm_type = model_params["norm_type"]

        if self.norm_type == "instance":
            self.norm = nn.InstanceNorm1d(
                embedding_dim, affine=True, track_running_stats=False
            )
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif self.norm_type == "rms":
            self.norm = RMSNorm(embedding_dim)
        elif self.norm_type == "none":
            self.norm = None
        else:
            raise NotImplementedError(f"Unknown norm_type: {self.norm_type}")

    def forward(self, input1, input2):
        added = input1 + input2
        if self.norm is None:
            return added
        if self.norm_type == "instance":
            out = self.norm(added.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.norm(added)
        return out


class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()
        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError
        self.multiple_of = multiple_of * model_parallel_size
        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size, out_features=inner_size, bias=False
        )
        self.l2 = nn.Linear(
            in_features=hidden_size, out_features=inner_size, bias=False
        )
        self.l3 = nn.Linear(
            in_features=inner_size, out_features=hidden_size, bias=False
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)


class FeedForward(nn.Module):
    """Standard feed-forward layer."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight
