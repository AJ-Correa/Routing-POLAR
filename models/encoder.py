import torch
import torch.nn as nn

from models.helpers import multi_head_attention, reshape_by_heads
from models.layers import FiLMGenerator, ParallelGatedMLP, RMSNorm, RoPE2D


class PreNorm(nn.Module):
    """Pre-LN wrapper used by the RouteFinder encoder (norm only; residual outside)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = RMSNorm(embedding_dim)

    def forward(self, _unused, x):
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        # Depth-aware residual scale (GPT-2 style): keeps Pre-LN + SiGLU stable at large L
        num_layers = max(1, int(model_params.get("encoder_layer_num", 1)))
        self.residual_scale = (2 * num_layers) ** -0.5
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_n_normalization_1 = PreNorm(embedding_dim)
        self.add_n_normalization_2 = PreNorm(embedding_dim)
        self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim)

    def forward(self, input1, coords=None, rope_cos=None, rope_sin=None, rope_module=None):
        normed = self.add_n_normalization_1(None, input1)
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(normed), head_num=head_num)
        k = reshape_by_heads(self.Wk(normed), head_num=head_num)
        v = reshape_by_heads(self.Wv(normed), head_num=head_num)

        # RoPE-2D on node tokens only (prompt tokens, if any, stay unrotated)
        if rope_module is not None and rope_cos is not None:
            n_coords = rope_cos.size(1)
            n_tokens = q.size(2)
            if n_tokens > n_coords:
                q_nodes, q_prompt = q[:, :, :n_coords], q[:, :, n_coords:]
                k_nodes, k_prompt = k[:, :, :n_coords], k[:, :, n_coords:]
                q_nodes, k_nodes = rope_module(q_nodes, k_nodes, rope_cos, rope_sin)
                q = torch.cat([q_nodes, q_prompt], dim=2)
                k = torch.cat([k_nodes, k_prompt], dim=2)
            else:
                q, k = rope_module(q, k, rope_cos, rope_sin)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        input2 = input1 + multi_head_out * self.residual_scale
        normed2 = self.add_n_normalization_2(None, input2)
        ff_out = self.feed_forward(normed2)
        return input2 + ff_out * self.residual_scale


class VRP_Encoder(nn.Module):
    """RouteFinder single-stream encoder (dense attention, optional FiLM / RoPE)."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]
        self.embedding_depot = nn.Linear(3, embedding_dim)
        self.embedding_node = nn.Linear(7, embedding_dim)

        self.use_film = model_params.get("use_film", False)
        if self.use_film:
            self.film_generator = FiLMGenerator(
                num_constraints=6, embedding_dim=embedding_dim
            )

        self.use_rope = model_params.get("use_rope", False)
        if self.use_rope:
            self.rope = RoPE2D(head_dim=model_params["qkv_dim"])

        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num)]
        )

    def _embed(self, td):
        if "num_depots" in td.keys():
            num_depots = td["num_depots"][0].item()
        else:
            num_depots = 1

        depot_feats = torch.cat(
            [
                td["locs"][:, :num_depots, :],
                td["distance_limit"][..., None].expand(-1, num_depots, -1),
            ],
            -1,
        )
        node_feats = torch.cat(
            (
                td["demand_linehaul"][..., num_depots:, None],
                td["demand_backhaul"][..., num_depots:, None],
                td["time_windows"][..., num_depots:, :],
                td["service_time"][..., num_depots:, None],
                td["locs"][:, num_depots:, :],
            ),
            -1,
        )
        depot_feats = torch.nan_to_num(depot_feats, nan=0.0, posinf=0.0, neginf=0.0)
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)

        global_embeddings = self.embedding_depot(depot_feats)
        cust_embeddings = self.embedding_node(node_feats)
        out = torch.cat((global_embeddings, cust_embeddings), -2)

        if self.use_film:
            gamma, beta = self.film_generator(td["p_s_tag"][:, 1:7])
            out = gamma * out + beta

        return out, td["locs"]

    def forward(self, td):
        out, coords = self._embed(td)
        rope_cos = rope_sin = rope_module = None
        if self.use_rope:
            rope_cos, rope_sin = self.rope._compute_rotation(coords)
            rope_module = self.rope
        for layer in self.layers:
            out = layer(
                out,
                coords=coords,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_module=rope_module,
            )
        return out, coords
