import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict
from dataclasses import dataclass, fields
from torch.nn.functional import scaled_dot_product_attention

from utils.functions import batchify


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor
    node_coords: Tensor = None  # (batch, seq_len, 2) for RoPE-2D
    log_d_nn: Tensor = None     # (batch, N, N) for TSNR distance bias

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


def linear_layer(input_dim, output_dim, std=1e-2, bias=True):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim, bias=bias)
    nn.init.normal_(linear.weight, std=std)
    nn.init.zeros_(linear.bias)
    return linear


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(
    q,
    k,
    v,
    ninf_mask=None,
    sparse=False,
    attn_weight=None,
    use_efficient=True,
):
    """Multi-head attention with optional memory-efficient implementation."""
    batch_s, head_num, n, key_dim = q.shape
    input_s = k.size(2)

    # Sparse attention variants
    if sparse == "topk":
        score = torch.matmul(q, k.transpose(2, 3))
        score_scaled = score * (key_dim**-0.5)
        if ninf_mask is not None:
            score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(
                batch_s, head_num, n, input_s
            )

        k_ = n // 2
        mask = torch.zeros(
            batch_s, head_num, n, n, device=score.device, requires_grad=False
        )
        mask.scatter_(-1, torch.topk(score_scaled, k=k_, dim=-1, largest=True)[1], 1.0)
        attn = torch.where(
            mask > 0, score_scaled, torch.full_like(score_scaled, float("-inf"))
        )
        attn = attn.softmax(dim=-1)
        out = (attn @ v) * attn_weight

        out_transposed = out.transpose(1, 2)
        return out_transposed.reshape(batch_s, n, head_num * key_dim)

    elif sparse == "relu":
        score = torch.matmul(q, k.transpose(2, 3))
        score_scaled = score * (key_dim**-0.5)
        if ninf_mask is not None:
            score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(
                batch_s, head_num, n, input_s
            )
        weights = torch.relu(score_scaled) ** 2
        out = torch.matmul(weights, v)
        out_transposed = out.transpose(1, 2)
        return out_transposed.reshape(batch_s, n, head_num * key_dim)

    # Standard attention
    else:
        if use_efficient:
            if ninf_mask is not None:
                attn_mask = ninf_mask[:, None, :, :].expand(
                    batch_s, head_num, n, input_s
                )
            else:
                attn_mask = None

            out = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            out_transposed = out.transpose(1, 2)
            return out_transposed.reshape(batch_s, n, head_num * key_dim)
        else:
            score = torch.matmul(q, k.transpose(2, 3))
            score_scaled = score * (key_dim**-0.5)
            if ninf_mask is not None:
                score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(
                    batch_s, head_num, n, input_s
                )
            weights = torch.softmax(score_scaled, dim=-1)
            out = torch.matmul(weights, v)
            out_transposed = out.transpose(1, 2)
            return out_transposed.reshape(batch_s, n, head_num * key_dim)