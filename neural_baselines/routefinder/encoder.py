import torch
import torch.nn as nn

from .layers import AddAndNorm, ParallelGatedMLP, multi_head_attention, reshape_by_heads


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
        self.add_n_normalization_1 = AddAndNorm(**model_params)
        self.add_n_normalization_2 = AddAndNorm(**model_params)
        self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim)

    def forward(self, input1):
        normed = self.add_n_normalization_1(None, input1)
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(normed), head_num=head_num)
        k = reshape_by_heads(self.Wk(normed), head_num=head_num)
        v = reshape_by_heads(self.Wv(normed), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        input2 = input1 + multi_head_out * self.residual_scale
        normed2 = self.add_n_normalization_2(None, input2)
        ff_out = self.feed_forward(normed2)
        return input2 + ff_out * self.residual_scale


class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]
        self.embedding_depot = nn.Linear(3, embedding_dim)
        self.embedding_node = nn.Linear(7, embedding_dim)

        model_params_copy = model_params.copy()
        model_params_copy["use_sparse"] = False
        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params_copy) for _ in range(encoder_layer_num)]
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
        return out, td["locs"]

    def forward(self, td):
        out, coords = self._embed(td)
        for layer in self.layers:
            out = layer(out)
        return out, coords
