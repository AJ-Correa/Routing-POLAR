import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from einops import rearrange
from typing import Tuple, Union
from dataclasses import dataclass, fields
from tensordict import TensorDict
from torch import Tensor
import numpy as np
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
        self.aux_loss = 0

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
        node_embed, node_coords, moe_loss = self.encoder(td)
        self.aux_loss = moe_loss

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
            logprobs, mask, cache, moe_loss = self.decoder(td, cache, num_starts, reld_alpha=reld_alpha)
            self.aux_loss += moe_loss
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
        moe_loss = 0

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
            out, loss = layer(out)
            moe_loss = moe_loss + loss
        return out, td["locs"], moe_loss # (batch, M + N, embedding), (batch, M + N, 2)


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

        self.feed_forward = MoE(input_size=embedding_dim, output_size=embedding_dim, num_experts=4,
                                   hidden_size=self.model_params['ff_hidden_dim'], k=2, T=1.0, noisy_gating=True,
                                   routing_level='node', routing_method='input_choice', moe_model="MLP")

    def forward(self, input1):
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v, use_efficient=False)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2, moe_loss = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3, moe_loss


class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq_last = nn.Linear(embedding_dim + 5, head_num * qkv_dim, bias=False)

        self.multi_head_combine = MoE(input_size=head_num * qkv_dim, output_size=embedding_dim, num_experts=4,
                                   hidden_size=self.model_params['ff_hidden_dim'], k=2, T=1.0, noisy_gating=True,
                                   routing_level='node', routing_method='input_choice', moe_model="Linear")

        # ReLD decoder
        self.use_reld = model_params.get("use_reld", False)
        if self.use_reld:
            self.attr_mapping = nn.Linear(5, embedding_dim, bias=False)
            self.decoder_ffn = FeedForward(**model_params)

    def forward(self, td, cache, num_starts, reld_alpha=1.0):
        moe_loss = 0

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
        out_concat = multi_head_attention(glimpse_q, glimpse_k, glimpse_v, mask, use_efficient=False)
        mh_atten_out, moe_loss = self.multi_head_combine(out_concat)

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
        return F.log_softmax(logits, dim=-1), mask, cache, moe_loss


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

        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1, input2):
        added = input1 + input2
        out = self.norm(added.transpose(1, 2)).transpose(1, 2)

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

class SparseDispatcher(object):
    """
        Helper for implementing a mixture of experts.
        The purpose of this class is to create input minibatches for the
        experts and to combine the results of the experts to form a unified
        output tensor.
        There are two functions:
        dispatch - take an input Tensor and create input Tensors for each expert.
        combine - take output Tensors from each expert and form a combined output
          Tensor.  Outputs from different experts for the same batch element are
          summed together, weighted by the provided "gates".
        The class is initialized with a "gates" Tensor, which specifies which
        batch elements go to which experts, and the weights to use when combining
        the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
        The inputs and outputs are all two-dimensional [batch, depth].
        Caller is responsible for collapsing additional dimensions prior to
        calling this class and reshaping the output to the original shape.
        See common_layers.reshape_like().
        Example use:
        gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
        inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
        experts: a list of length `num_experts` containing sub-networks.
        dispatcher = SparseDispatcher(num_experts, gates)
        expert_inputs = dispatcher.dispatch(inputs)
        expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
        outputs = dispatcher.combine(expert_outputs)
        The preceding code sets the output for a particular example b to:
        output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
        This class takes advantage of sparsity in the gate matrix by including in the
        `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, routing_level="node"):
        """
            Create a SparseDispatcher.
        """

        self._gates = gates
        self._num_experts = num_experts
        self._routing_level = routing_level

        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # drop indices: corresponds to which expert
        _, self._expert_index = sorted_experts.split(1, dim=1)

        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]

        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()

        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]

        # get the corresponding weights
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
            Create one input Tensor for each expert.
            The `Tensor` for a expert `i` contains the slices of `inp` corresponding
            to the batch elements `b` where `gates[b, i] > 0`.
            Args:
              inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
            Returns:
              a list of `num_experts` `Tensor`s with shapes
                `[expert_batch_size_i, <extra_input_dims>]`.
        """
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
            Sum together the expert output, weighted by the gates.
            The slice corresponding to a particular batch element `b` is computed
            as the sum over all experts `i` of the expert output, weighted by the
            corresponding gate values.  If `multiply_by_gates` is set to False, the
            gate values are ignored.
            Args:
              expert_out: a list of `num_experts` `Tensor`s, each with shape
                `[expert_batch_size_i, <extra_output_dims>]`.
              multiply_by_gates: a boolean
            Returns:
              a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # stitched = torch.cat(expert_out, 0).exp()
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if self._routing_level == "node":
                stitched = stitched.mul(self._nonzero_gates)
            elif self._routing_level == "instance":
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))

        if self._routing_level == "node":
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)
        elif self._routing_level == "instance":
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)

        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps

        # back to log space
        # return combined.log()
        return combined

    def expert_to_gates(self):
        """
            Gate values corresponding to the examples in the per-expert `Tensor`s.
            Returns:
              a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
                  and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MoE(nn.Module):
    """
        Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
        Args:
        input_size: integer - size of the input
        output_size: integer - size of the output
        num_experts: an integer - number of experts
        hidden_size: an integer - hidden size of the experts
        k: an integer - how many experts to use for each batch element
        T: float - temperature to control the entropy of probability distribution
        noisy_gating: boolean - only used for input_choice routing method
        routing_level: string - ["node", "instance", "problem"]
        routing_method: string - ["input_choice", "expert_choice", "soft_moe"]
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size=None, k=1, T=1.0, noisy_gating=True, routing_level="node", routing_method="input_choice", moe_model="MLP"):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.routing_level = routing_level
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.T = T

        # instantiate experts and gating/routing network
        if moe_model == "MLP":
            self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for _ in range(self.num_experts)])
        elif moe_model == "Linear":
            self.experts = nn.ModuleList([nn.Linear(self.input_size, self.output_size) for _ in range(self.num_experts)])
        else:
            raise NotImplementedError

        if routing_method == "soft_moe":
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts * k), requires_grad=True)
        else:
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        if not (routing_level in ["node", "instance"] and routing_method == "input_choice"):
            # Refer to: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
            torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """
            The squared coefficient of variation of a sample.
            Useful as a loss to encourage a positive distribution to be more uniform.
            Epsilons added for numerical stability.
            Returns 0 for an empty Tensor.
            Args:
            x: a `Tensor`.
            Returns:
            a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """
            Compute the "true load per expert", given the gates.
            The load is the number of data/instances for which the corresponding gate is >0.
            Args:
            gates: a `Tensor` of shape [batch_size, n]
            Returns:
            a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
            Helper function to NoisyTopKGating.
            Computes the probability that value is in top k, given different random noise.

            "This gives us a way of backpropagating from a loss that balances the number
            of times each expert is in the top k experts per example."

            In the case of no noise, pass in None for noise_stddev, and the result will
            not be differentiable.
            Args:
            clean_values: a `Tensor` of shape [batch, n].
            noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
              normally distributed noise with standard deviation noise_stddev.
            noise_stddev: a `Tensor` of shape [batch, n], or None
            noisy_top_values: a `Tensor` of shape [batch, m].
               "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
            Returns:
            a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)  # (batch_size, 1)
        is_in = torch.gt(noisy_values, threshold_if_in)  # (batch_size, num_experts)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)

        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
            Noisy top-k gating.
              See paper: https://arxiv.org/abs/1701.06538.
              Args:
                x: input Tensor with shape [batch_size, input_size]
                train: a boolean - we only add noise at training time.
                noise_epsilon: a float
              Returns:
                gates: a Tensor with shape [batch_size, num_experts]
                load: a Tensor with shape [num_experts]
        """
        # for instance-level routing: [batch_size, problem, input_size] -> [batch_size, input_size]
        x = x.mean(1) if self.routing_level == "instance" else x

        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits = top_logits[..., :self.k]
        top_k_indices = top_indices[..., :self.k]
        top_k_gates = self.softmax(top_k_logits / self.T)

        zeros = torch.zeros_like(logits, requires_grad=True)  # (batch_size, num_experts)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)  # non-topk elements will be 0

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def problem_top_k_gating(self, x, prob_emb=None):
        """
            Specialize for problem-level routing without noise, since no need for load balancing.
            x: [batch_size, -1, input_size]
        """
        # for problem-level routing: [batch_size, problem, input_size] -> [1, input_size]
        x = x.mean(0).mean(0).unsqueeze(0) + prob_emb if prob_emb is not None else x.mean(0).mean(0).unsqueeze(0)
        # x = x.max(0)[0].max(0)[0].unsqueeze(0) + prob_emb if prob_emb is not None else x.mean(0).mean(0).unsqueeze(0)
        logits = x @ self.w_gate

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        # top_indices = torch.multinomial(self.softmax(logits / self.T), self.k)  # sample
        # top_logits = logits[:, top_indices.squeeze(0)]
        selected_logits = top_logits[..., :self.k].squeeze(0)  # [1, k] -> [k]
        selected_indices = top_indices[..., :self.k].squeeze(0)
        selected_gates = self.softmax(selected_logits / self.T)
        # print(selected_indices)

        return selected_indices, selected_gates

    def expert_gating(self, x, k):
        """
            Expert Routing, ref to "Mixture-of-Experts with Expert Choice Routing" in NeurIPS 2022.
            Pros: Perfect load balancing.
            Cons: Some tokens may not pass any expert.
                input: (batch_size, ..., input_size)
            return:
                G - probability matrix: (num_experts, topk);
                I - index matrix: (num_experts, topk);
                P - One-hot version of I: (num_experts, topk, batch_size).
        """
        x = x.mean(1) if self.routing_level == "instance" else x

        logits = x @ self.w_gate
        S = torch.softmax(logits, dim=0)
        G, I = torch.topk(S.T, k=k, dim=-1)
        P = nn.functional.one_hot(I, num_classes=x.size(0))

        return G, I, P

    def forward(self, x, loss_coef=1e-3, prob_emb=None):
        """
            Args:
            x: tensor shape [batch_size, problem, input_size]
            train: a boolean scalar.
            loss_coef: a scalar - multiplier on load-balancing losses

            Returns:
            y: a tensor with shape [batch_size, output_size].
            extra_training_loss: a scalar. This should be added into the overall
            training loss of the model. The backpropagation of this loss encourages
            all experts to be approximately equally used across a batch (only used for node/instance-level routing).
        """
        assert self.routing_level in ["node", "instance", "problem"], "Unsupported Routing Level!"
        output_shape = list(x.size()[:-1]) + [self.output_size]
        if self.routing_level in ["instance", "problem"]:
            # the input must have the shape of [batch_size, -1, input_size]
            assert x.dim() == 3
        elif self.routing_level == "node":
            # the input must have the shape of [batch_size, input_size]
            x = x.reshape(-1, self.input_size) if x.dim() != 2 else x

        if self.routing_level == "problem":
            """
                1. Problem-Level Routing: Batches of instances are routed to different experts.
            """
            if self.routing_method != "random":
                expert_ids, gates = self.problem_top_k_gating(x, prob_emb)
            else:
                expert_ids = torch.randperm(self.num_experts)[:self.k]
                gates = torch.Tensor([1./self.k for _ in range(self.k)])
            expert_outputs = []
            for i in expert_ids.tolist():
                expert_outputs.append(self.experts[i](x).unsqueeze(0))
            expert_outputs = torch.cat(expert_outputs, 0)  # [k, batch_size, -1, output_size]
            y = (expert_outputs * gates.reshape(-1, 1, 1, 1)).sum(0)  # [batch_size, -1, output_size]
            return y, 0
        elif self.routing_level in ["node", "instance"]:
            """
                2. Node-Level Routing: Tokens are routed to different experts.
                3. Instance-Level Routing: Instances (containing many nodes) are routed to different experts.
            """
            if self.routing_method == "input_choice":
                """
                    - (a) Input Choice: Each node chooses TopK experts, auxiliary losses required for load balancing.
                                        Refer to "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" in ICLR 2017.
                """
                gates, load = self.noisy_top_k_gating(x, self.training)
                # calculate importance loss
                importance = gates.sum(0)
                loss = self.cv_squared(importance) + self.cv_squared(load)
                loss *= loss_coef
                dispatcher = SparseDispatcher(self.num_experts, gates, routing_level=self.routing_level)
                expert_inputs = dispatcher.dispatch(x)
                # gates = dispatcher.expert_to_gates()
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]  # [(batch_size*, num_experts), ...]
                y = dispatcher.combine(expert_outputs)
                # an example of loss scales: reinforce_loss: tensor(-9.3630) aug_loss: tensor(0.0001)
                return y.reshape(output_shape), loss
            elif self.routing_method == "expert_choice":
                """
                    - (b) Expert Choice: Each expert chooses TopK instances, explicitly ensuring the load balancing. 
                                         However, some tokens may (1) not be chosen, or (2) be chosen many times.
                                         Refer to "Mixture-of-Experts with Expert Choice Routing" in NeurIPS 2022.
                """
                assert self.input_size == self.output_size
                # Here we view self.k as the capacity factor, and calculate k as: k = num_tokens * capacity_factor / num_experts
                k = int(x.size(0) * self.k / self.num_experts)
                expert_inputs_shape = [self.num_experts, k] + list(x.size()[1:])
                G, I, P = self.expert_gating(x, k)  # [num_experts, topk]
                expert_inputs = P.float() @ x.reshape(P.size(-1), -1)  # [num_experts, topk, -1]
                expert_inputs = expert_inputs.reshape(expert_inputs_shape)  # [num_experts, topk, ..., input_size]
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
                expert_outputs = torch.stack(expert_outputs, 0)  # [num_experts, topk, ..., output_size]
                output = torch.einsum('ijl,ij,ijd->ld', P, G, expert_outputs.reshape(self.num_experts, k, -1))
                return output.reshape(output_shape), 0
            elif self.routing_method == "soft_moe":
                """
                    - (c) Soft MoE: performs an implicit soft assignment by passing convex combinations of all input tokens to each expert.
                                    Refer to "From Sparse to Soft Mixtures of Experts" in arxiv 2308.00951.
                """
                x_ = x.mean(1) if self.routing_level == "instance" else x
                logits = x_ @ self.w_gate  # [batch_size, num_experts * topk]
                expert_inputs_shape = [logits.size(-1)] + list(x.size()[1:])
                expert_inputs = torch.softmax(logits, dim=0).T @ x.reshape(logits.size(0), -1)  # [num_experts * topk, -1]
                expert_inputs = expert_inputs.reshape(expert_inputs_shape)
                expert_outputs = [self.experts[i](expert_inputs[i*self.k: (i+1)*self.k]) for i in range(self.num_experts)]
                expert_outputs = torch.stack(expert_outputs, 0).reshape(self.num_experts * self.k, -1)  # [num_experts, topk, ..., output_size] -> [num_experts * topk, -1]
                output = torch.softmax(logits, dim=-1) @ expert_outputs  # [batch_size, -1]
                return output.reshape(output_shape), 0
            elif self.routing_method == "random":
                """
                    - (d) random heuristic.
                """
                zeros = torch.zeros((x.size(0), self.num_experts), requires_grad=True)
                random_indices = torch.argsort(torch.rand(x.size(0), self.num_experts), dim=-1)[:, :self.k]
                gates = zeros.scatter(-1, random_indices, 1./self.k)
                dispatcher = SparseDispatcher(self.num_experts, gates, routing_level=self.routing_level)
                expert_inputs = dispatcher.dispatch(x)
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]  # [(batch_size*, num_experts), ...]
                y = dispatcher.combine(expert_outputs)
                return y.reshape(output_shape), 0
            else:
                raise NotImplementedError
