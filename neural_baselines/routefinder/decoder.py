import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.functions import gather_by_index, unbatchify

from .layers import multi_head_attention, reshape_by_heads


class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.Wq_last = nn.Linear(embedding_dim + 5, head_num * qkv_dim, bias=False)

    def forward(self, td, cache, num_starts, reld_alpha=1.0):
        td = unbatchify(td, num_starts)

        cur_node = td["current_node"]
        cur_node_embedding = gather_by_index(
            cache.node_embeddings, cur_node, squeeze=False
        )

        remaining_linehaul = td["vehicle_capacity"] - td["used_capacity_linehaul"]
        remaining_backhaul = td["vehicle_capacity"] - td["used_capacity_backhaul"]
        state_embedding = torch.cat(
            [
                remaining_linehaul,
                remaining_backhaul,
                td["current_time"],
                td["current_route_length"],
                td["open_route"],
            ],
            dim=-1,
        )
        context_embedding = torch.cat([cur_node_embedding, state_embedding], dim=-1)

        glimpse_k = cache.glimpse_key
        glimpse_v = cache.glimpse_val
        logit_k = cache.logit_key

        glimpse_q = reshape_by_heads(
            self.Wq_last(context_embedding), head_num=self.model_params["head_num"]
        )
        mask = td["action_mask"]

        out_concat = multi_head_attention(glimpse_q, glimpse_k, glimpse_v, mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, logit_k)
        score_scaled = score / self.model_params["sqrt_embedding_dim"]

        logits = rearrange(score_scaled, "b s l -> (s b) l", s=num_starts)
        mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        logits = torch.tanh(logits) * self.model_params["logit_clipping"]
        logits[~mask] = float("-inf")
        return F.log_softmax(logits, dim=-1), mask, cache
