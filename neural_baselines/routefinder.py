import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Union
from dataclasses import dataclass, fields
from tensordict import TensorDict
from torch import Tensor
import random
import math

from utils.functions import batchify, gather_by_index, unbatchify, unbatchify_and_gather
from torch.nn.functional import scaled_dot_product_attention


class VRPModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_mode = "rl"
        self.encoder = VRP_Encoder(**args.model_params)
        self.decoder = VRP_Decoder(**args.model_params)
        self.encoded_nodes = None  # (batch, N + M, EMBEDDING_DIM)
        self.encoded_coords = None  # (batch, N + M, 2)
        self.now_p_type = None

    @staticmethod
    def greedy(logprobs, mask=None):
        """Select the action with the highest probability."""
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert not (~mask).gather(1, selected.unsqueeze(-1)).data.any(), (
                "infeasible action selected"
            )
        return selected

    @staticmethod
    def sampling(logprobs, log, mask=None):
        """Sample an action with a multinomial distribution given to the log probabilities."""
        probs = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)
        if mask is not None:
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                log("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
            assert not (~mask).gather(1, selected.unsqueeze(-1)).data.any(), (
                "infeasible action selected"
            )
        return selected

    def set_loss_mode(self, mode: str):
        """Set loss mode to RL or PO."""
        self.loss_mode = mode

    def forward(self, td, env, reld_alpha=1.0, with_greedy=False):
        """Main forward pass: encode -> decode -> compute reward.

        When ``with_greedy=True`` (used during PO+LS training) the environment
        appends an extra start at depot node 0.  All normal POMO starts are
        decoded by sampling; the depot-0 start is decoded greedily.  This
        happens inside a single forward pass — no second encode, no extra call.
        """
        args = self.args

        # Encode nodes to get embeddings
        node_embed, node_coords = self.encoder(td)

        # (valid only within the same forward/backward step)
        self.encoded_nodes = node_embed
        self.encoded_coords = node_coords

        # Select POMO start nodes for multi-start decoding
        if self.training and self.loss_mode == "po":
            try:
                po_B = args.trainer_params.get("po_B", None)
            except Exception:
                po_B = None
        else:
            po_B = None
        num_starts, start_actions, greedy_mask = env.select_start_nodes(
            td, po_B=po_B, with_greedy=with_greedy
        )
        start_actions = start_actions.to(td.device)

        # Expand batch for multi-start
        greedy_mask = greedy_mask.to(td.device).bool()
        td = batchify(td, num_starts)

        # Handle greedy mask size mismatch
        if greedy_mask.numel() != td.batch_size[0] * num_starts:
            batch = td.batch_size[0]
            if greedy_mask.numel() == num_starts:
                greedy_mask = greedy_mask.repeat_interleave(batch)
            else:
                greedy_mask = greedy_mask.view(-1)[: (num_starts * batch)].to(
                    torch.bool
                )

        # Initialize tracking lists
        logprobs_list = [
            torch.zeros_like(start_actions, dtype=torch.float32, device=td.device)
        ]
        actions_list = [start_actions]

        # First step: depot/customer selection
        td.set("action", start_actions)
        td = env.step(td)["next"]

        # Multi-depot: handle second start action if present
        pomo_customer_starts = (
            env.get_pomo_customer_starts()
            if hasattr(env, "get_pomo_customer_starts")
            else None
        )
        if pomo_customer_starts is not None:
            pomo_customer_starts = pomo_customer_starts.to(td.device)
            logprobs_list.append(
                torch.zeros_like(
                    pomo_customer_starts, dtype=torch.float32, device=td.device
                )
            )
            actions_list.append(pomo_customer_starts)
            td.set("action", pomo_customer_starts)
            td = env.step(td)["next"]

        # Prepare decoder cache for efficient attention
        decoder_k = reshape_by_heads(
            self.decoder.Wk(node_embed), head_num=args.model_params["head_num"]
        )
        decoder_v = reshape_by_heads(
            self.decoder.Wv(node_embed), head_num=args.model_params["head_num"]
        )
        decoder_single_head_k = node_embed.transpose(1, 2)

        cache = PrecomputedCache(
            node_embed, decoder_k, decoder_v, decoder_single_head_k, node_coords
        )

        # Autoregressive decoding loop
        step = 0
        while not td["done"].all():
            logprobs, mask, cache = self.decoder(td, cache, num_starts, reld_alpha=reld_alpha)
            if self.training:
                # Sample for all starts; greedy for the depot-0 slot
                if greedy_mask.any():
                    select_sample = VRPModel.sampling(logprobs, self.args.log, mask)
                    select_greedy = VRPModel.greedy(logprobs, mask)
                    select = torch.where(greedy_mask, select_greedy, select_sample)
                else:
                    select = VRPModel.sampling(logprobs, self.args.log, mask)
            else:
                select = VRPModel.greedy(logprobs, mask)
            logprobs = gather_by_index(logprobs, select, dim=1)
            td.set("action", select)
            actions_list.append(select)
            logprobs_list.append(logprobs)
            td = env.step(td)["next"]
            step += 1

        # Compute final reward and return
        logprobs = torch.stack(logprobs_list, 1)
        actions = torch.stack(actions_list, 1)
        rew, tours = env.get_reward(td, actions)
        td.set("reward", rew)
        assert (logprobs > -1000).data.all(), (
            "Logprobs should not be -inf, check sampling procedure!"
        )
        return {
            "reward": td["reward"],
            "log_likelihood": logprobs,
            "tours": tours,
            "ccl_active_steps": [],
        }

class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]
        self.embedding_depot = nn.Linear(3, embedding_dim)  # locs, distance_limit
        self.embedding_node = nn.Linear(7, embedding_dim)

        model_params_copy = model_params.copy()
        model_params_copy["use_sparse"] = False
        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params_copy) for _ in range(encoder_layer_num)]
        )

    def forward(self, td):
        # Get number of depots (default to 1 for single-depot problems)
        if "num_depots" in td.keys():
            num_depots = td["num_depots"][0].item()  # Scalar, same for all batch items
        else:
            num_depots = 1

        # Extract depot features: coords + distance_limit
        # Depots are at indices 0 to num_depots-1
        depot_feats = torch.cat(
            [
                td["locs"][:, :num_depots, :],  # First M nodes are depots
                td["distance_limit"][..., None].expand(
                    -1, num_depots, -1
                ),  # Same limit for all depots
            ],
            -1,
        )
        # Extract customer features: demands, TW, service_time, coords
        # Customers are at indices num_depots onwards
        node_feats = torch.cat(
            (
                td["demand_linehaul"][..., num_depots:, None],
                td["demand_backhaul"][..., num_depots:, None],
                td["time_windows"][..., num_depots:, :],
                td["service_time"][..., num_depots:, None],
                td["locs"][:, num_depots:, :],
            ),
            -1,
        )  # (batch, N, 7)

        # Handle NaN values
        depot_feats = torch.nan_to_num(depot_feats, nan=0.0, posinf=0.0, neginf=0.0)
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
        bs, n, _7 = node_feats.shape  # n = number of customers

        # Embed depot and customer nodes separately
        global_embeddings = self.embedding_depot(depot_feats)  # [batch, M, embed_dim]
        cust_embeddings = self.embedding_node(node_feats)  # [batch, N, embed_dim]

        # Concatenate depot and customer embeddings
        out = torch.cat(
            (global_embeddings, cust_embeddings), -2
        )  # [batch, M + N, embed_dim]

        # Process through transformer layers
        for layer in self.layers:
            out = layer(out)
        return out, td["locs"] # (batch, M + N, embedding), (batch, M + N, 2)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndNorm(**model_params)
        self.add_n_normalization_2 = AddAndNorm(**model_params)

        self.feed_forward = ParallelGatedMLP()

    def forward(self, input1):
        normed = self.add_n_normalization_1(None, input1)

        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(normed), head_num=head_num)
        k = reshape_by_heads(self.Wk(normed), head_num=head_num)
        v = reshape_by_heads(self.Wv(normed), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        input2 = input1 + multi_head_out
        normed2 = self.add_n_normalization_2(None, input2)
        ff_out = self.feed_forward(normed2)
        out3 = input2 + ff_out
        
        return out3


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

        # ReLD decoder
        self.use_reld = model_params.get("use_reld", False)
        if self.use_reld:
            self.attr_mapping = nn.Linear(5, embedding_dim, bias=False)
            self.decoder_ffn = FeedForward(**model_params)

    def forward(self, td, cache, num_starts, reld_alpha=1.0):
        # Split batch dimension for multi-start decoding
        td = unbatchify(td, num_starts)
        
        # Get embedding of current node
        cur_node = td["current_node"]
        cur_node_embedding = gather_by_index(cache.node_embeddings, cur_node, squeeze=False)
        
        # Standard Static State Embedding (Fallback)
        remaining_linehaul = td["vehicle_capacity"] - td["used_capacity_linehaul"]
        remaining_backhaul = td["vehicle_capacity"] - td["used_capacity_backhaul"]
        state_embedding = torch.cat([
            remaining_linehaul, remaining_backhaul, td["current_time"], 
            td["current_route_length"], td["open_route"]
        ], dim=-1)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], dim=-1)

        # Use cached static keys
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
            reld_contrib = cur_node_embedding + self.attr_mapping(state_embedding.clone())
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

        if self.use_reld:
            cur_locs = gather_by_index(td["locs"], td["current_node"], dim=2).unsqueeze(2)  # [B, S, 1, 2]
            all_locs = td["locs"]  # [B, S, N, 2]
            
            # cdist computes distance between each start's current pos and all N nodes
            distance = torch.cdist(cur_locs, all_locs).squeeze(2)  # [B, S, N]
            logdis = -1.0 * torch.nan_to_num(torch.log(distance), nan=0.0, posinf=0.0, neginf=0.0)
            
            # Rearrange logdis to match logits shape
            logdis_flat = rearrange(logdis, "b s l -> (s b) l", s=num_starts)
            
            logits = logits + logdis_flat

        logits[~mask] = float("-inf")
        return F.log_softmax(logits, dim=-1), mask, cache


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor
    node_coords: Tensor = None  # (batch, seq_len, 2) for RoPE-2D

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


class AddAndNorm(nn.Module):
    """Residual connection with normalization."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm_type = model_params["norm_type"]

        self.norm = RMSNorm(embedding_dim)

    def forward(self, input1, input2):
        out = self.norm(input2)

        return out

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
    use_efficient=True,
):
    """Multi-head attention with optional memory-efficient implementation."""
    batch_s, head_num, n, key_dim = q.shape
    input_s = k.size(2)

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

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight

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
