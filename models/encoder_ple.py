import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import EncoderLayer
from models.layers import FiLMGenerator, RMSNorm, RoPE2D


class GlobalExpert(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.layer = EncoderLayer(**model_params)

    def forward(self, x, coords=None, rope_cos=None, rope_sin=None, rope_module=None):
        return self.layer(
            x,
            coords=coords,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            rope_module=rope_module,
        )


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
        # Pre-norm experts: gated blending shrinks activations unless pathways
        # are re-normalized explicitly.
        self.shared_norm = RMSNorm(self.embed_dim)
        self.task_norm = RMSNorm(self.embed_dim)

    def forward(
        self,
        shared_in,
        task_in,
        prompt_embedding,
        num_nodes,
        coords=None,
        rope_cos=None,
        rope_sin=None,
        rope_module=None,
    ):
        shared_out = self.shared_norm(
            self.shared_expert(
                shared_in,
                coords=coords,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_module=rope_module,
            )
        )

        depth_prompt = self.prompt_depth_proj(prompt_embedding)
        shared_summary = shared_out.mean(dim=1)
        gate_input = torch.cat([depth_prompt, shared_summary], dim=-1)
        gate_weights = F.softmax(self.task_gate_proj(gate_input), dim=-1)

        task_weights = gate_weights[:, : self.num_task_groups]
        shared_weight = gate_weights[:, self.num_task_groups :]
        task_weights = task_weights / task_weights.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)

        task_outs = [
            expert(
                task_in,
                coords=coords,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_module=rope_module,
            )
            for expert in self.task_experts
        ]
        task_stack = torch.stack(task_outs, dim=0)
        task_stack_nodes = task_stack[:, :, :num_nodes, :]
        task_out_nodes = torch.einsum(
            "k b n d, b k -> b n d", task_stack_nodes, task_weights
        )

        alpha = torch.sigmoid(self.alpha)
        shared_contrib = shared_out * shared_weight.unsqueeze(1)
        final_task_nodes = self.task_norm(task_out_nodes + alpha * shared_contrib)

        if task_in.size(1) > num_nodes:
            final_task = torch.cat([final_task_nodes, task_in[:, num_nodes:]], dim=1)
        else:
            final_task = final_task_nodes

        return shared_out, final_task


class VRP_Encoder(nn.Module):
    """RouteFinder PLE encoder: shared + K task experts (global/dense only)."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        encoder_layer_num = model_params["encoder_layer_num"]
        self.p_num = model_params.get("p_num", 6)
        num_task_groups = int(
            model_params.get("K", model_params.get("ple_num_task_groups", 3))
        )

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

        self.ple_layers = nn.ModuleList(
            [
                PLELayer(model_params, num_task_groups=num_task_groups)
                for _ in range(encoder_layer_num)
            ]
        )
        self.final_fusion = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)
        self.output_norm = RMSNorm(embedding_dim)

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

        if self.use_film:
            gamma, beta = self.film_generator(td["p_s_tag"][:, 1:7])
            cust_embeddings = gamma * cust_embeddings + beta

        out = torch.cat((global_embeddings, cust_embeddings), -2)
        return out, td["locs"], num_depots + node_feats.size(1)

    def forward(self, td, prompt):
        out, coords, num_nodes = self._embed(td)
        prompt_for_gate = prompt.mean(dim=1) if prompt.dim() == 3 else prompt

        rope_cos = rope_sin = rope_module = None
        if self.use_rope:
            rope_cos, rope_sin = self.rope._compute_rotation(coords)
            rope_module = self.rope

        shared_x = out
        task_x = out
        for i, ple_layer in enumerate(self.ple_layers):
            if i == 0:
                task_x = torch.cat([task_x, prompt], dim=1)

            shared_x, task_x = ple_layer(
                shared_x,
                task_x,
                prompt_for_gate,
                num_nodes,
                coords=coords,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_module=rope_module,
            )

        out = self.output_norm(
            self.final_fusion(torch.cat([shared_x, task_x[:, :num_nodes]], dim=-1))
        )
        return out, coords
