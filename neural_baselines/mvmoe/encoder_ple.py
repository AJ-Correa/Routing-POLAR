import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderLayer


class GlobalExpert(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        mp = model_params.copy()
        mp["use_sparse"] = False
        self.layer = EncoderLayer(**mp)

    def forward(self, x):
        return self.layer(x)


class PLELayer(nn.Module):
    def __init__(self, model_params, num_task_groups=5):
        super().__init__()
        self.num_task_groups = num_task_groups
        self.embed_dim = model_params["embedding_dim"]

        self.shared_expert = GlobalExpert(**model_params)
        self.task_experts = nn.ModuleList(
            [GlobalExpert(**model_params) for _ in range(num_task_groups)]
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, num_task_groups + 1),
        )
        _init = torch.logit(torch.tensor(0.1))
        self.alpha = nn.Parameter(_init.clone())

    def forward(self, x, gate_context):
        moe_loss = 0
        shared_out, loss = self.shared_expert(x)
        moe_loss = moe_loss + loss

        shared_summary = shared_out.mean(dim=1)
        gate_input = torch.cat([gate_context, shared_summary], dim=-1)
        gate_weights = F.softmax(self.gate_proj(gate_input), dim=-1)

        task_weights = gate_weights[:, : self.num_task_groups]
        shared_weight = gate_weights[:, self.num_task_groups :]

        task_outs = []
        for expert in self.task_experts:
            t_out, t_loss = expert(shared_out)
            task_outs.append(t_out)
            moe_loss = moe_loss + t_loss

        task_stack = torch.stack(task_outs, dim=0)
        task_out = torch.einsum("k b n d, b k -> b n d", task_stack, task_weights)

        alpha = torch.sigmoid(self.alpha)
        shared_contrib = shared_out * shared_weight.unsqueeze(-1).unsqueeze(-1)
        final_task = task_out + alpha * shared_contrib
        return shared_out, final_task, moe_loss


class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        encoder_layer_num = model_params["encoder_layer_num"]
        num_task_groups = model_params.get("ple_num_task_groups", 5)

        self.embedding_depot = nn.Linear(3, embedding_dim)
        self.embedding_node = nn.Linear(7, embedding_dim)

        self.ple_layers = nn.ModuleList(
            [
                PLELayer(model_params, num_task_groups=num_task_groups)
                for _ in range(encoder_layer_num)
            ]
        )
        self.final_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
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
        moe_loss = 0
        out, coords = self._embed(td)
        gate_context = out.mean(dim=1)

        shared_x = out
        task_x = out
        for ple_layer in self.ple_layers:
            shared_x, task_x, loss = ple_layer(shared_x, gate_context)
            moe_loss = moe_loss + loss

        out = self.final_fusion(torch.cat([shared_x, task_x], dim=-1))
        return out, coords, moe_loss
