import torch
import torch.nn as nn

from models.layers import *
from models.helpers import *

class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]
        self.embedding_depot = nn.Linear(3, embedding_dim)  # locs, distance_limit
        self.embedding_node = nn.Linear(7, embedding_dim)
        self.p_num = self.model_params["p_num"]

        # FiLM conditioning for constraints
        self.use_film = model_params.get("use_film", False)
        if self.use_film:
            self.film_generator = FiLMGenerator(
                num_constraints=6, embedding_dim=embedding_dim
            )

        # RoPE-2D for spatial coordinates
        self.use_rope = model_params.get("use_rope", False)
        if self.use_rope:
            self.rope = RoPE2D(head_dim=model_params["qkv_dim"])

        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num)]
        )

        model_params_copy = model_params.copy()
        model_params_copy["use_sparse"] = False
        self.layers2 = nn.ModuleList(
            [EncoderLayer(**model_params_copy) for _ in range(encoder_layer_num)]
        )
        self.layers1combine = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num)]
        )
        self.layers2combine = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim)
                for _ in range(encoder_layer_num - 1)
            ]
        )

    def forward(self, td, prompt):
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

        # Store coordinates for RoPE
        coords = td["locs"]  # (batch, N + M, 2)

        # Apply FiLM: modulate embeddings based on constraint flags
        if self.use_film:
            constraint_flags = td["p_s_tag"][:, 1:7]
            gamma, beta = self.film_generator(constraint_flags)
            cust_embeddings = gamma * cust_embeddings + beta

        # Concatenate depot and customer embeddings
        out = torch.cat(
            (global_embeddings, cust_embeddings), -2
        )  # [batch, M + N, embed_dim]
        out2 = out
        num_nodes = num_depots + n  # Total nodes = M depots + N customers

        # Compute RoPE rotation matrices if enabled
        rope_cos, rope_sin = None, None
        rope_module = None
        if self.use_rope:
            rope_cos, rope_sin = self.rope._compute_rotation(coords)
            rope_module = self.rope

        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            if i == 0:
                out2 = torch.cat(
                    (out2, prompt), dim=1
                )  # [batch, M + N + p_num, embed_dim]
            out = layer(
                out,
                coords=coords,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_module=rope_module,
            )  # Sparse branch: with RoPE
            out2 = self.layers2[i](
                out2,
                coords=coords,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_module=rope_module,
            )  # Global branch: with RoPE
            # Combine: num_nodes = M depots + N customers
            out = out + self.layers1combine[i](out2[:, :num_nodes])
            if i != len(self.layers) - 1:
                out2_ = out2[:, :num_nodes] + self.layers2combine[i](out)
                out2_ = torch.cat((out2_, out2[:, -self.p_num :]), dim=1)
                out2 = out2_

        # Return embeddings for M + N nodes (depots + customers)
        return out[
            :, :num_nodes
        ], coords  # (batch, M + N, embedding), (batch, M + N, 2)


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

        if model_params["ffd"] == "ffd":
            self.feed_forward = FeedForward(**model_params)
        elif model_params["ffd"] == "siglu":
            assert embedding_dim == 128
            self.feed_forward = ParallelGatedMLP()
        else:
            raise NotImplementedError
        self.attn_weight = None
        if self.model_params["use_sparse"] == "topk":
            self.attn_weight = nn.Parameter(
                torch.tensor([0.2], dtype=torch.float, requires_grad=True)
            )

    def forward(
        self, input1, coords=None, rope_cos=None, rope_sin=None, rope_module=None
    ):
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        # Apply RoPE-2D (only to positions that have coordinates)
        if rope_module is not None and rope_cos is not None:
            # rope_cos/sin have shape [batch, n_coords, head_dim]
            # q/k may have more tokens (prompt tokens) than coords
            n_coords = rope_cos.size(1)
            n_tokens = q.size(2)

            if n_tokens > n_coords:
                # Split: node tokens get RoPE, prompt tokens don't
                q_nodes, q_prompt = q[:, :, :n_coords], q[:, :, n_coords:]
                k_nodes, k_prompt = k[:, :, :n_coords], k[:, :, n_coords:]
                q_nodes, k_nodes = rope_module(q_nodes, k_nodes, rope_cos, rope_sin)
                q = torch.cat([q_nodes, q_prompt], dim=2)
                k = torch.cat([k_nodes, k_prompt], dim=2)
            else:
                q, k = rope_module(q, k, rope_cos, rope_sin)

        attn_weight = None
        if self.model_params["use_sparse"] == "topk":
            attn_weight = self.attn_weight

        out_concat = multi_head_attention(
            q,
            k,
            v,
            sparse=self.model_params["use_sparse"],
            attn_weight=attn_weight,
        )
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3