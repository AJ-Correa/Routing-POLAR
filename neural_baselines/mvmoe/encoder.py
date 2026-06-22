import torch
import torch.nn as nn

from .layers import AddAndNorm, multi_head_attention, reshape_by_heads
from .moe import MoE


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
        self.feed_forward = MoE(
            input_size=embedding_dim,
            output_size=embedding_dim,
            num_experts=4,
            hidden_size=self.model_params["ff_hidden_dim"],
            k=2,
            T=1.0,
            noisy_gating=True,
            routing_level="node",
            routing_method="input_choice",
            moe_model="MLP",
        )

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
        losses = []
        out, coords = self._embed(td)
        for layer in self.layers:
            out, loss = layer(out)
            if isinstance(loss, torch.Tensor):
                losses.append(loss)
        return out, coords, sum(losses) if losses else 0
