import torch
import torch.nn as nn

from models.layers import *
from models.helpers import *

class DualBranchExpert(nn.Module):
    """
    Your existing encoder layer, packaged as an expert.
    Each expert has its own sparse branch, global branch, and combination layers.
    """
    def __init__(self, model_params, use_sparse=True):
        super().__init__()
        self.model_params = model_params
        self.use_sparse = use_sparse
        self.p_num = model_params["p_num"]
        
        # Sparse branch
        mp_sparse = model_params.copy()
        mp_sparse["use_sparse"] = model_params.get("use_sparse", "topk") if use_sparse else False
        self.sparse_layer = EncoderLayer(**mp_sparse)
        
        # Global branch  
        mp_global = model_params.copy()
        mp_global["use_sparse"] = False
        self.global_layer = EncoderLayer(**mp_global)
        
        # Combination layers (same as your current code)
        embed_dim = model_params["embedding_dim"]
        self.combine1 = nn.Linear(embed_dim, embed_dim)  # global -> sparse
        self.combine2 = nn.Linear(embed_dim, embed_dim)  # sparse -> global
        
    def forward(self, x_sparse, x_global, num_nodes, coords, rope_cos, rope_sin, rope_module):
        # x_sparse: (B, N, D)  (N is num_nodes)
        # x_global: (B, N + p_num, D)
        
        out_sparse = self.sparse_layer(x_sparse, coords=coords, rope_cos=rope_cos, 
                                       rope_sin=rope_sin, rope_module=rope_module)
        
        out_global = self.global_layer(x_global, coords=coords, rope_cos=rope_cos,
                                       rope_sin=rope_sin, rope_module=rope_module)
        
        # Combine: same logic
        out_sparse = out_sparse + self.combine1(out_global[:, :num_nodes])
        out_global_res = out_global[:, :num_nodes] + self.combine2(out_sparse)
        
        out_global = torch.cat([out_global_res, out_global[:, num_nodes:]], dim=1)
        return out_sparse, out_global

class PLELayer(nn.Module):
    """
    Progressive Layered Extraction layer.
    Maintains separate shared and task-specific pathways with progressive separation.
    """
    def __init__(self, model_params, num_shared_experts=2, num_task_groups=5):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_task_groups = num_task_groups
        self.embed_dim = model_params["embedding_dim"]
        
        # Shared experts: visible to all tasks
        self.shared_experts = nn.ModuleList([
            DualBranchExpert(model_params, use_sparse=True) 
            for _ in range(num_shared_experts)
        ])
        
        # Task-specific experts: each group has its own private experts
        self.task_experts = nn.ModuleList([
            DualBranchExpert(model_params, use_sparse=False)  # global-only for task-specific
            for _ in range(num_task_groups)
        ])
        
        # Shared gate: which shared experts to use (conditioned on prompt)
        self.shared_gate = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_shared_experts)
        )
        
        # Task-specific gate: which task experts + shared experts to use
        self.task_gate = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_task_groups + num_shared_experts)
        )
        
    def forward(self, shared_sparse_in, shared_global_in, task_sparse_in, task_global_in, num_nodes, prompt_embedding, task_group_ids, coords, rope_cos, rope_sin, rope_module):
        # --- 1. Shared Experts ---
        shared_outputs_sparse = []
        shared_outputs_global = []
        for expert in self.shared_experts:
            s, g = expert(shared_sparse_in, shared_global_in, num_nodes, coords, rope_cos, rope_sin, rope_module)
            shared_outputs_sparse.append(s)
            shared_outputs_global.append(g)
        
        shared_stack_sparse = torch.stack(shared_outputs_sparse, dim=0) # (num_shared_experts, B, N, D)
        shared_stack_global = torch.stack(shared_outputs_global, dim=0)
        
        shared_gate_logits = self.shared_gate(prompt_embedding)
        shared_gate_weights = F.softmax(shared_gate_logits, dim=-1)  # (B, K_s)
        
        shared_out_sparse = torch.einsum('k b n d, b k -> b n d', shared_stack_sparse, shared_gate_weights)
        shared_out_global = torch.einsum('k b n d, b k -> b n d', shared_stack_global, shared_gate_weights)
        
        # --- 2. Task-Specific Experts ---
        task_gate_logits = self.task_gate(prompt_embedding)  # (B, K_t + K_s)
        task_gate_weights = F.softmax(task_gate_logits, dim=-1)  # (B, K_t + K_s)
        
        task_specific_weights = task_gate_weights[:, :self.num_task_groups]   # (B, K_t)
        task_shared_weights = task_gate_weights[:, self.num_task_groups:]       # (B, K_s)
        
        task_outputs_sparse = []
        task_outputs_global = []
        for i, expert in enumerate(self.task_experts):
            s, g = expert(shared_out_sparse, task_global_in, num_nodes, coords, rope_cos, rope_sin, rope_module)
            task_outputs_sparse.append(s)
            task_outputs_global.append(g)
        
        task_stack_sparse = torch.stack(task_outputs_sparse, dim=0)
        task_stack_global = torch.stack(task_outputs_global, dim=0)
        
        task_out_sparse = torch.einsum('k b n d, b k -> b n d', task_stack_sparse, task_specific_weights)
        task_out_global = torch.einsum('k b n d, b k -> b n d', task_stack_global, task_specific_weights)
        
        shared_contrib_sparse = torch.einsum('k b n d, b k -> b n d', shared_stack_sparse, task_shared_weights)
        shared_contrib_global = torch.einsum('k b n d, b k -> b n d', shared_stack_global, task_shared_weights)
        
        final_task_sparse = task_out_sparse + 0.1 * shared_contrib_sparse
        final_task_global = task_out_global + 0.1 * shared_contrib_global
        
        return shared_out_sparse, shared_out_global, final_task_sparse, final_task_global

class VRP_Encoder(nn.Module):
    """
    Full encoder using PLE layers instead of fixed dual-branch.
    Replaces your VRP_Encoder's layers/layers2/layers1combine.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]
        self.num_task_groups = 5 # 0: Default, 1: TW, 2: O, 3: L, 4: B
        
        self.embedding_depot = nn.Linear(3, embedding_dim)
        self.embedding_node = nn.Linear(7, embedding_dim)
        self.p_num = model_params["p_num"]
        
        self.use_film = model_params.get("use_film", False)
        if self.use_film:
            self.film_generator = FiLMGenerator(
                num_constraints=6, embedding_dim=embedding_dim
            )
        
        self.use_rope = model_params.get("use_rope", False)
        if self.use_rope:
            self.rope = RoPE2D(head_dim=model_params["qkv_dim"])
        
        # PLE layers
        self.ple_layers = nn.ModuleList([
            PLELayer(model_params, num_shared_experts=2, num_task_groups=self.num_task_groups)
            for _ in range(encoder_layer_num)
        ])
        
        # Final combination: shared + task-specific outputs
        self.final_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, td, prompt):
        num_depots = td["num_depots"][0].item() if "num_depots" in td.keys() else 1
        
        depot_feats = torch.cat([
            td["locs"][:, :num_depots, :],
            td["distance_limit"][..., None].expand(-1, num_depots, -1),
        ], -1)
        
        node_feats = torch.cat((
            td["demand_linehaul"][..., num_depots:, None],
            td["demand_backhaul"][..., num_depots:, None],
            td["time_windows"][..., num_depots:, :],
            td["service_time"][..., num_depots:, None],
            td["locs"][:, num_depots:, :],
        ), -1)
        
        depot_feats = torch.nan_to_num(depot_feats, nan=0.0, posinf=0.0, neginf=0.0)
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
        bs, n, _ = node_feats.shape
        
        global_emb = self.embedding_depot(depot_feats)
        cust_emb = self.embedding_node(node_feats)
        
        coords = td["locs"]
        
        if self.use_film:
            constraint_flags = td["p_s_tag"][:, 1:7]
            gamma, beta = self.film_generator(constraint_flags)
            cust_emb = gamma * cust_emb + beta
        
        out = torch.cat((global_emb, cust_emb), -2)
        num_nodes = num_depots + n
        
        constraint_flags = td["p_s_tag"][:, 1:7]
        has_pd = (constraint_flags[:, 3:5].sum(dim=-1) > 0).long() # B or MB
        has_l = (constraint_flags[:, 2] > 0).long()
        has_tw = (constraint_flags[:, 1] > 0).long()
        has_open = (constraint_flags[:, 0] > 0).long()
        
        task_group_ids = torch.zeros(bs, device=out.device, dtype=torch.long)
        task_group_ids = torch.where(has_open > 0, torch.tensor(2, device=out.device), task_group_ids)
        task_group_ids = torch.where(has_l > 0, torch.tensor(3, device=out.device), task_group_ids)
        task_group_ids = torch.where(has_pd > 0, torch.tensor(4, device=out.device), task_group_ids)
        task_group_ids = torch.where(has_tw > 0, torch.tensor(1, device=out.device), task_group_ids)
        
        rope_cos, rope_sin = None, None
        rope_module = None
        if self.use_rope:
            rope_cos, rope_sin = self.rope._compute_rotation(coords)
            rope_module = self.rope
        
        shared_sparse_in = out
        shared_global_in = out
        task_sparse_in = out
        task_global_in = out
        
        prompt_for_gate = prompt.mean(dim=1) if prompt.dim() == 3 else prompt
        
        for i, ple_layer in enumerate(self.ple_layers):
            if i == 0:
                shared_global_in = torch.cat([shared_global_in, prompt], dim=1)
                task_global_in = torch.cat([task_global_in, prompt], dim=1)
            
            s_sparse, s_global, t_sparse, t_global = ple_layer(
                shared_sparse_in, shared_global_in, task_sparse_in, task_global_in, num_nodes,
                prompt_for_gate, task_group_ids, coords, rope_cos, rope_sin, rope_module
            )
            
            shared_sparse_in = s_sparse
            shared_global_in = s_global
            task_sparse_in = t_sparse
            task_global_in = t_global
        
        final_repr = self.final_fusion(
            torch.cat([shared_sparse_in[:, :num_nodes], task_sparse_in[:, :num_nodes]], dim=-1)
        )
        
        return final_repr, coords


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

        # QK-Norm for training stability
        self.use_qk_norm = model_params.get("use_qk_norm", False)
        if self.use_qk_norm:
            norm_type = self.model_params.get("norm_type", "rms")
            self.q_norm = RMSNorm(qkv_dim) if norm_type == "rms" else nn.LayerNorm(qkv_dim)
            self.k_norm = RMSNorm(qkv_dim) if norm_type == "rms" else nn.LayerNorm(qkv_dim)

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

        # Apply QK-Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

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
        