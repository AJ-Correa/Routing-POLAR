import torch
import random
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from models.layers import *
from models.helpers import *
from utils.functions import unbatchify, gather_by_index

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

        self.use_ccl = model_params.get("use_ccl", False)
        if self.use_ccl:
            self.rgcr = RGCR(embedding_dim)
            self.tsnr = TSNR(embedding_dim, head_num, qkv_dim, norm_type=model_params.get("norm_type", "rms"))
            self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

            self.ccl_prob_train = model_params.get("ccl_prob_train", 1.0)
            self.ccl_prob_test = model_params.get("ccl_prob_test", 1.0)
        else:
            self.Wq_last = nn.Linear(embedding_dim + 5, head_num * qkv_dim, bias=False)

        # ReLD decoder
        self.use_reld = model_params.get("use_reld", False)
        if self.use_reld:
            if not self.use_ccl:
                self.attr_mapping = nn.Linear(5, embedding_dim, bias=False)
            self.decoder_ffn = FeedForward(**model_params)

    def forward(self, td, cache, num_starts, reld_alpha=1.0, ccl_active=None):
        # Split batch dimension for multi-start decoding
        td = unbatchify(td, num_starts)
        
        # Get embedding of current node
        cur_node = td["current_node"]
        cur_node_embedding = gather_by_index(cache.node_embeddings, cur_node, squeeze=False)
        
        # Determine Context
        if self.use_ccl:
            # --- Extract Node-Specific Attributes ---
            # Coordinates (c^x, c^y)
            cur_locs = gather_by_index(td["locs"], cur_node, dim=2) 
            cur_cx, cur_cy = cur_locs[..., 0:1], cur_locs[..., 1:2]
            
            # Demands (linehaul, backhaul)
            cur_dl = gather_by_index(td["demand_linehaul"], cur_node, dim=2).unsqueeze(-1)
            cur_db = gather_by_index(td["demand_backhaul"], cur_node, dim=2).unsqueeze(-1)

            # Time Windows (early, late, service)
            cur_tw = gather_by_index(td["time_windows"], cur_node, dim=2)
            cur_tw = torch.nan_to_num(cur_tw, nan=0.0, posinf=0.0, neginf=0.0)
            cur_te, cur_tl = cur_tw[..., 0:1], cur_tw[..., 1:2]
            cur_ts = gather_by_index(td["service_time"], cur_node, dim=2).unsqueeze(-1)
            cur_ts = torch.nan_to_num(cur_ts, nan=0.0, posinf=0.0, neginf=0.0)

            # --- Extract State-Specific Attributes ---
            # c_{i,j}: remaining vehicle capacity
            remaining_linehaul = td["vehicle_capacity"] - td["used_capacity_linehaul"]
            remaining_backhaul = td["vehicle_capacity"] - td["used_capacity_backhaul"]

            # d'_{i,j} & d_{i,j}: total traveled & remaining distance
            cur_route_len = td["current_route_length"]
            rem_dist = td["distance_limit"] - cur_route_len
            rem_dist = torch.nan_to_num(rem_dist, nan=0.0, posinf=0.0, neginf=0.0)
            
            # t_{i,j}: current time
            cur_time = td["current_time"]

            # --- Build the Constraint Sets (Eq. 4) ---
            c_B = torch.cat([cur_dl, cur_db, remaining_linehaul, remaining_backhaul], dim=-1)
            c_L = torch.cat([cur_cx, cur_cy, rem_dist], dim=-1)
            c_O = torch.cat([cur_cx, cur_cy, cur_route_len], dim=-1)
            c_TW = torch.cat([cur_te, cur_tl, cur_ts, cur_time], dim=-1)

            # 1. RGCR: Context Reformulation
            context_state = self.rgcr(cur_node_embedding, c_B, c_L, c_O, c_TW)
            context_embedding = self.rgcr.proj_final(context_state)

            if ccl_active is None:
                prob = self.ccl_prob_train if self.training else self.ccl_prob_test
                ccl_active = random.random() < prob

            # 2. TSNR: Update Global Node Embeddings BEFORE calculating logits -> Shape: [B, N, D]
            if ccl_active:
                new_node_embed = self.tsnr(
                    H=cache.node_embeddings,
                    C=context_embedding,
                    coords=cache.node_coords,
                    cur_nodes=cur_node,
                    log_d_nn=cache.log_d_nn
                )

                # --- DYNAMICALLY RECOMPUTE KEYS/VALUES FOR CURRENT STEP ---
                glimpse_k = reshape_by_heads(self.Wk(new_node_embed), head_num=self.model_params["head_num"])
                glimpse_v = reshape_by_heads(self.Wv(new_node_embed), head_num=self.model_params["head_num"])
                logit_k = new_node_embed.transpose(1, 2)

                cache = PrecomputedCache(new_node_embed, glimpse_k, glimpse_v, logit_k, cache.node_coords, cache.log_d_nn)
            else:
                glimpse_k = cache.glimpse_key
                glimpse_v = cache.glimpse_val
                logit_k = cache.logit_key
        else:
            # Standard Static State Embedding (Fallback)
            remaining_linehaul = td["vehicle_capacity"] - td["used_capacity_linehaul"]
            remaining_backhaul = td["vehicle_capacity"] - td["used_capacity_backhaul"]
            state_embedding = torch.cat([
                remaining_linehaul, remaining_backhaul, td["current_time"], 
                td["current_route_length"], td["open_route"]
            ], dim=-1)
            
            context_embedding = torch.cat([cur_node_embedding, state_embedding], dim=-1)

            # Use cached static keys
            new_node_embed = None
            glimpse_k = cache.glimpse_key
            glimpse_v = cache.glimpse_val
            logit_k = cache.logit_key
        
        # Compute query vectors
        glimpse_q = reshape_by_heads(
            self.Wq_last(context_embedding), head_num=self.model_params["head_num"]
        )
        mask = td["action_mask"]

        # Multi-head attention with cached keys and values
        out_concat = multi_head_attention(glimpse_q, glimpse_k, glimpse_v, mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # ReLD: add residual connections and FFN
        if self.use_reld:
            # We set reld_alpha to zero in the first epoch of training to stabilize convergence
            if self.use_ccl:
                reld_contrib = cur_node_embedding + context_state[:, :, self.model_params["embedding_dim"]:]
            else:
                reld_contrib = cur_node_embedding + self.attr_mapping(
                state_embedding.clone()
                )
            mh_atten_out = mh_atten_out + (reld_alpha * reld_contrib)
            ffn_out = self.decoder_ffn(mh_atten_out)
            mh_atten_out = mh_atten_out + (reld_alpha * ffn_out)

        # Compute logits with single-head attention
        score = torch.matmul(mh_atten_out, logit_k)
        score_scaled = score / self.model_params["sqrt_embedding_dim"]

        # Rearrange for multi-start: (batch, num_starts, nodes) -> (batch * num_starts, nodes)
        logits = rearrange(score_scaled, "b s l -> (s b) l", s=num_starts)
        mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        # Clip logits and apply mask
        logits = torch.tanh(logits) * self.model_params["logit_clipping"]

        """if self.use_reld:
            cur_locs = gather_by_index(td["locs"], td["current_node"], dim=2).unsqueeze(2)  # [B, S, 1, 2]
            all_locs = td["locs"]  # [B, S, N, 2]
            
            # cdist computes distance between each start's current pos and all N nodes
            distance = torch.cdist(cur_locs, all_locs).squeeze(2)  # [B, S, N]
            logdis = -1.0 * torch.nan_to_num(torch.log(distance), nan=0.0, posinf=0.0, neginf=0.0)
            
            # Rearrange logdis to match logits shape
            logdis_flat = rearrange(logdis, "b s l -> (s b) l", s=num_starts)
            
            logits = logits + logdis_flat"""

        logits[~mask] = float("-inf")
        return F.log_softmax(logits, dim=-1), mask, cache
        
