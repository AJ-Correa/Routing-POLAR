from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from utils.functions import batchify


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor
    node_coords: Tensor = None

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, (Tensor, TensorDict)):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


class ParallelGatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()
        multiple_of = inner_size_multiple_of
        if mlp_activation == "gelu":
            self.act = F.gelu
        elif mlp_activation == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError
        self.multiple_of = multiple_of * model_parallel_size
        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )
        self.l1 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)


class AddAndNorm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = RMSNorm(embedding_dim)

    def forward(self, input1, input2):
        return self.norm(input2)


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)


def multi_head_attention(q, k, v, ninf_mask=None, use_efficient=True):
    batch_s, head_num, n, key_dim = q.shape
    input_s = k.size(2)

    if use_efficient:
        if ninf_mask is not None:
            attn_mask = ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
        else:
            attn_mask = None
        out = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out_transposed = out.transpose(1, 2)
        return out_transposed.reshape(batch_s, n, head_num * key_dim)

    score = torch.matmul(q, k.transpose(2, 3)) * (key_dim**-0.5)
    if ninf_mask is not None:
        score = score + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    weights = torch.softmax(score, dim=-1)
    out = torch.matmul(weights, v).transpose(1, 2)
    return out.reshape(batch_s, n, head_num * key_dim)