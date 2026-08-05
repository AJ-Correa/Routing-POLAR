import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AddAndNorm, FeedForward, multi_head_attention, reshape_by_heads


class PLEEncoderLayer(nn.Module):
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
        self.feed_forward = FeedForward(**model_params)

    def forward(self, input1):
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v, use_efficient=False)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        return self.add_n_normalization_2(out1, out2)


class GlobalExpert(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        mp = model_params.copy()
        mp["use_sparse"] = False
        self.layer = PLEEncoderLayer(**mp)

    def forward(self, x):
        return self.layer(x)


class PLELayer(nn.Module):
    def __init__(self, model_params, num_task_groups=3):
        super().__init__()
        self.num_task_groups = num_task_groups
        self.embed_dim = model_params["embedding_dim"]

        self.shared_expert = GlobalExpert(**model_params)
        self.task_experts = nn.ModuleList(
            [GlobalExpert(**model_params) for _ in range(num_task_groups)]
        )
        self.prompt_depth_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.task_gate_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, num_task_groups + 1),
        )
        _init = torch.logit(torch.tensor(0.1))
        self.alpha = nn.Parameter(_init.clone())

    def forward(self, shared_in, task_in, prompt_embedding, num_nodes):
        shared_out = self.shared_expert(shared_in)

        depth_prompt = self.prompt_depth_proj(prompt_embedding)
        shared_summary = shared_out.mean(dim=1)
        gate_input = torch.cat([depth_prompt, shared_summary], dim=-1)
        gate_weights = F.softmax(self.task_gate_proj(gate_input), dim=-1)

        task_weights = gate_weights[:, : self.num_task_groups]
        shared_weight = gate_weights[:, self.num_task_groups :]

        task_outs = [expert(task_in) for expert in self.task_experts]
        task_stack = torch.stack(task_outs, dim=0)
        task_stack_nodes = task_stack[:, :, :num_nodes, :]
        task_out_nodes = torch.einsum(
            "k b n d, b k -> b n d", task_stack_nodes, task_weights
        )

        alpha = torch.sigmoid(self.alpha)
        shared_contrib = shared_out * shared_weight.unsqueeze(-1)
        final_task_nodes = task_out_nodes + alpha * shared_contrib

        if task_in.size(1) > num_nodes:
            final_task = torch.cat([final_task_nodes, task_in[:, num_nodes:]], dim=1)
        else:
            final_task = final_task_nodes

        return shared_out, final_task


class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        encoder_layer_num = model_params["encoder_layer_num"]
        self.p_num = model_params.get("p_num", 6)
        num_task_groups = int(model_params.get("ple_num_task_groups", 3))

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
        return out, td["locs"], num_depots + node_feats.size(1)

    def forward(self, td, prompt):
        out, coords, num_nodes = self._embed(td)
        prompt_for_gate = prompt.mean(dim=1) if prompt.dim() == 3 else prompt

        shared_x = out
        task_x = out
        for i, ple_layer in enumerate(self.ple_layers):
            if i == 0:
                task_x = torch.cat([task_x, prompt], dim=1)

            shared_x, task_x = ple_layer(
                shared_x, task_x, prompt_for_gate, num_nodes
            )

        out = self.final_fusion(torch.cat([shared_x, task_x[:, :num_nodes]], dim=-1))
        return out, coords
