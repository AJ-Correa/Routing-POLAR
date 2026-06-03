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


class AddAndNorm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1, input2):
        added = input1 + input2
        return self.norm(added.transpose(1, 2)).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


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
