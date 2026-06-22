import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
import concurrent.futures
import multiprocessing as mp
from search import _ls_instance_iterated
from search.vrplib_helpers import vrplib_round_func_from_id

from utils.functions import batchify, gather_by_index

from models.encoder import VRP_Encoder
from models.encoder_ple import VRP_Encoder as VRP_Encoder_PLE
from models.decoder import VRP_Decoder
from models.layers import *
from models.helpers import *


class VRPModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_mode = "rl"
        if self.args.model_params.get("use_ple", False):
            self.encoder = VRP_Encoder_PLE(**args.model_params)
        else:
            self.encoder = VRP_Encoder(**args.model_params)
        self.decoder = VRP_Decoder(**args.model_params)
        self.encoded_nodes = None  # (batch, N + M, EMBEDDING_DIM)
        self.encoded_coords = None  # (batch, N + M, 2)
        self.now_p_type = None
        self.prompt_net = PromptNet(args)

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
        # Generate task-specific prompts from constraint flags
        p_out = self.prompt_net(td)
        prompt = p_out["prompt"]
        # Encode nodes to get embeddings
        node_embed, node_coords = self.encoder(td, prompt)

        # Cache encoder output so route_forward can reuse it (sync trainer_2 path)
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
            node_embed,
            decoder_k,
            decoder_v,
            decoder_single_head_k,
            node_coords,
        )

        # Autoregressive decoding loop
        step = 0
        while not td["done"].all():
            if self.decoder.use_ccl:
                prob = (
                    self.decoder.ccl_prob_train
                    if self.training
                    else self.decoder.ccl_prob_test
                )
                use_ccl_this_step = random.random() < prob
            else:
                use_ccl_this_step = None

            logprobs, mask, cache = self.decoder(
                td,
                cache,
                num_starts,
                reld_alpha=reld_alpha,
                ccl_active=use_ccl_this_step,
            )

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
        out = {
            "reward": td["reward"],
            "log_likelihood": logprobs,
            "tours": tours,
        }

        return out

    def route_forward(
        self,
        td,
        env,
        tours,
        tour_lengths,
        num_starts,
        node_embed=None,
        node_coords=None,
        reld_alpha=1.0,
    ):
        """Compute log-likelihoods for LS-improved tours.

        When node_embed and node_coords are provided (sync trainer_2 path), the
        encoder is skipped — saving one full pass through the 6 dual-branch
        transformer layers.  When they are None (async trainer / profiler paths),
        the encoder runs as normal.

        Parameters
        ----------
        node_embed  : (batch, N+1, embedding_dim) or None
        node_coords : (batch, N+1, 2) or None
        """
        if tours.dim() != 2:
            raise ValueError("tours must be 2D: [batch, steps]")
        if tour_lengths.dim() != 1 or tour_lengths.size(0) != tours.size(0):
            raise ValueError("tour_lengths must be 1D with same batch as tours")

        # Run encoder only when embeddings are not pre-supplied
        if node_embed is None or node_coords is None:
            p_out = self.prompt_net(td)
            prompt = p_out["prompt"]
            node_embed, node_coords = self.encoder(td, prompt)

        td = batchify(td, num_starts)

        # Prepare decoder cache (3 linear projections)
        decoder_k = reshape_by_heads(
            self.decoder.Wk(node_embed), head_num=self.args.model_params["head_num"]
        )
        decoder_v = reshape_by_heads(
            self.decoder.Wv(node_embed), head_num=self.args.model_params["head_num"]
        )
        decoder_single_head_k = node_embed.transpose(1, 2)

        cache = PrecomputedCache(
            node_embed,
            decoder_k,
            decoder_v,
            decoder_single_head_k,
            node_coords,
        )

        # Replay tour steps and compute log-probs
        actions_list = []
        logprobs_list = []

        step = 0

        while not td["done"].all():
            logprobs, _, cache = self.decoder(
                td, cache, num_starts, reld_alpha=reld_alpha
            )
            action = tours[:, step]
            logprobs = gather_by_index(logprobs, action.unsqueeze(1), dim=1)

            td.set("action", action)
            actions_list.append(action)
            logprobs_list.append(logprobs)
            td = env.step(td)["next"]
            step += 1

        logprobs = torch.stack(logprobs_list, dim=1)
        actions = torch.stack(actions_list, dim=1)
        reward, tours_out = env.get_reward(td, actions)
        assert (logprobs > -1000).data.all(), (
            "Logprobs should not be -inf, check sampling procedure!"
        )

        return {"reward": reward, "log_likelihood": logprobs, "tours": tours_out}

    def route_forward_ccl(
        self,
        td,
        env,
        ls_tours,
        ls_tour_lengths,
        node_embed=None,
        node_coords=None,
        reld_alpha=1.0,
    ):
        """Compute log-likelihoods for LS-improved tours alongside N sampled solutions.

        Samples N solutions using POMO to build robust context in TSNR, while
        teacher-forcing the LS-improved tour at start 0. ONLY returns the
        reward and log-likelihood for the LS-improved tour.
        """
        if ls_tours.dim() != 2:
            raise ValueError("ls_tours must be 2D: [batch, steps]")

        args = self.args
        batch_size = td.batch_size[0]

        # Encode if embeddings not supplied
        if node_embed is None or node_coords is None:
            p_out = self.prompt_net(td)
            prompt = p_out["prompt"]
            node_embed, node_coords = self.encoder(td, prompt)

        # Select POMO starts (same logic as forward)
        po_B = args.trainer_params.get("po_B", None)
        num_starts, start_actions, _ = env.select_start_nodes(
            td, po_B=po_B, with_greedy=True
        )
        start_actions = start_actions.to(td.device)

        # Expand batch for multi-start decoding
        td = batchify(td, num_starts)

        # Track global step index
        step = 0
        start_actions[:batch_size] = ls_tours[:, step]

        # First step: depot/customer selection
        logprobs_list = [
            torch.zeros_like(start_actions, dtype=torch.float32, device=td.device)
        ]
        actions_list = [start_actions]
        td.set("action", start_actions)
        td = env.step(td)["next"]
        step += 1

        # Multi-depot second start
        pomo_customer_starts = (
            env.get_pomo_customer_starts()
            if hasattr(env, "get_pomo_customer_starts")
            else None
        )
        if pomo_customer_starts is not None:
            pomo_customer_starts = pomo_customer_starts.to(td.device)
            # Override start 0 with the LS tour's second action (customer start)
            pomo_customer_starts[:batch_size] = ls_tours[:, step]
            logprobs_list.append(
                torch.zeros_like(
                    pomo_customer_starts, dtype=torch.float32, device=td.device
                )
            )
            actions_list.append(pomo_customer_starts)
            td.set("action", pomo_customer_starts)
            td = env.step(td)["next"]
            step += 1

        # Prepare decoder cache
        decoder_k = reshape_by_heads(
            self.decoder.Wk(node_embed), head_num=args.model_params["head_num"]
        )
        decoder_v = reshape_by_heads(
            self.decoder.Wv(node_embed), head_num=args.model_params["head_num"]
        )
        decoder_single_head_k = node_embed.transpose(1, 2)

        cache = PrecomputedCache(
            node_embed,
            decoder_k,
            decoder_v,
            decoder_single_head_k,
            node_coords,
        )

        # Autoregressive decoding
        while not td["done"].all():
            if self.decoder.use_ccl:
                prob = (
                    self.decoder.ccl_prob_train
                    if self.training
                    else self.decoder.ccl_prob_test
                )
                use_ccl_this_step = random.random() < prob
            else:
                use_ccl_this_step = None

            logprobs, mask, cache = self.decoder(
                td,
                cache,
                num_starts,
                reld_alpha=reld_alpha,
                ccl_active=use_ccl_this_step,
            )

            # Sample all starts
            select = VRPModel.sampling(logprobs, self.args.log, mask)

            # Teacher-force start 0 (LS tour)
            within_tour = step < ls_tour_lengths
            if within_tour.any():
                ls_actions = ls_tours[:, step]
                select[:batch_size] = torch.where(
                    within_tour, ls_actions, select[:batch_size]
                )

            logprobs = gather_by_index(logprobs, select, dim=1)
            td.set("action", select)
            actions_list.append(select)
            logprobs_list.append(logprobs)
            td = env.step(td)["next"]
            step += 1

        logprobs = torch.stack(logprobs_list, 1)
        actions = torch.stack(actions_list, 1)
        reward, tours = env.get_reward(td, actions)

        assert (logprobs > -1000).data.all(), (
            "Logprobs should not be -inf, check sampling procedure!"
        )

        # We successfully sampled N solutions for context and evaluated the LS tour.
        # Now, extract AND RETURN ONLY the LS-improved tour results (the first batch_size elements)
        ls_reward = reward[:batch_size]
        ls_logprobs = logprobs[:batch_size]

        return {
            "reward": ls_reward,
            "log_likelihood": ls_logprobs,
            "tours": ls_tours,
        }

    @torch.inference_mode()
    def iterative_refinement(
        self,
        td_orig,
        env,
        ls_nb_granular: int = 40,
        num_iters: int = 5000,
        stop_condition: str = "iterations",
        num_seconds: float | None = None,
        dmax: int = 30,
        dmin: int = 15,
        gamma: int = 30,
        eta_min: float = 0.01,
    ):
        args = self.args
        input_batch_size = td_orig.batch_size[0]
        num_augment = int(args.tester_params.get("num_augment", 1))
        if num_augment > 1 and input_batch_size % num_augment == 0:
            batch_size = input_batch_size // num_augment
            td_orig = td_orig[:batch_size]
        else:
            batch_size = input_batch_size
        device = td_orig.device
        po_B = args.trainer_params.get("po_B", None)
        neural_start = time.perf_counter()

        # ═════════════════════════════════════════════════════════════════
        # ONCE: encode + build static decoder cache (never changes)
        # ═════════════════════════════════════════════════════════════════
        p_out = self.prompt_net(td_orig)
        node_embed, node_coords = self.encoder(td_orig, p_out["prompt"])

        decoder_k = reshape_by_heads(
            self.decoder.Wk(node_embed), head_num=args.model_params["head_num"]
        )
        decoder_v = reshape_by_heads(
            self.decoder.Wv(node_embed), head_num=args.model_params["head_num"]
        )
        decoder_shk = node_embed.transpose(1, 2)

        static_cache = PrecomputedCache(
            node_embed,
            decoder_k,
            decoder_v,
            decoder_shk,
            node_coords,
        )

        # ═════════════════════════════════════════════════════════════════
        # ONCE: extract CPU data for LS (never changes)
        # ═════════════════════════════════════════════════════════════════
        td_cpu = td_orig.cpu()
        locs_np = td_cpu["locs"].numpy()
        dlin_np = td_cpu["demand_linehaul"].numpy()
        dbac_np = td_cpu["demand_backhaul"].numpy()
        dlim_np = td_cpu["distance_limit"].numpy()
        open_np = td_cpu["open_route"].numpy()
        tw_np = td_cpu["time_windows"].numpy()
        svc_np = td_cpu["service_time"].numpy()
        workers = min(batch_size, os.cpu_count() or 1)

        if "num_depots" in td_cpu.keys():
            nd_raw = td_cpu["num_depots"].numpy()
            num_depots_np = nd_raw[:, 0] if nd_raw.ndim == 2 else nd_raw
        else:
            num_depots_np = np.ones(batch_size, dtype=np.int64)

        # Instance-specific mixed-backhaul flags from p_s_tag[:, 5].
        if "p_s_tag" in td_cpu.keys():
            mixed_backhaul_flags = td_cpu["p_s_tag"][:, 5].numpy().astype(bool)
        else:
            mixed_backhaul_flags = np.zeros(batch_size, dtype=bool)

        best_reward = None
        best_tours = None

        # ── Decode ────────────────────────────────────────────────────
        td = td_orig.clone()

        num_starts, start_actions, greedy_mask = env.select_start_nodes(
            td, po_B=po_B, with_greedy=False
        )
        start_actions = start_actions.to(device)

        td_dec = batchify(td, num_starts)
        actions_list = [start_actions]
        logprobs_list = [
            torch.zeros_like(start_actions, dtype=torch.float32, device=device)
        ]
        td_dec.set("action", start_actions)
        td_dec = env.step(td_dec)["next"]

        pomo_cust = (
            env.get_pomo_customer_starts()
            if hasattr(env, "get_pomo_customer_starts")
            else None
        )
        if pomo_cust is not None:
            pomo_cust = pomo_cust.to(device)
            actions_list.append(pomo_cust)
            logprobs_list.append(
                torch.zeros_like(pomo_cust, dtype=torch.float32, device=device)
            )
            td_dec.set("action", pomo_cust)
            td_dec = env.step(td_dec)["next"]

        # ── REUSE static cache instead of rebuilding ─────────────────
        cache = static_cache

        # Autoregressive decode
        while not td_dec["done"].all():
            use_ccl_step = (
                random.random() < self.decoder.ccl_prob_test
                if self.decoder.use_ccl
                else None
            )
            logprobs, mask, cache = self.decoder(
                td_dec,
                cache,
                num_starts,
                ccl_active=use_ccl_step,
            )
            select = VRPModel.greedy(logprobs, mask)
            actions_list.append(select)
            logprobs_list.append(gather_by_index(logprobs, select, dim=1))
            td_dec.set("action", select)
            td_dec = env.step(td_dec)["next"]

        actions = torch.stack(actions_list, dim=1)
        reward_all, tours_all = env.get_reward(td_dec, actions)

        # Best across POMO starts
        reward_2d = reward_all.view(num_starts, batch_size)
        tours_3d = tours_all.view(num_starts, batch_size, -1)
        best_start = reward_2d.argmax(dim=0)
        batch_idx = torch.arange(batch_size, device=device)
        reward_iter = reward_2d[best_start, batch_idx]
        tours_iter = tours_3d[best_start, batch_idx]

        # Update global best
        if best_reward is None:
            best_reward = reward_iter
            best_tours = tours_iter
        else:
            improved_m = reward_iter > best_reward
            best_reward = torch.where(improved_m, reward_iter, best_reward)

            T_best = best_tours.size(1)
            T_iter = tours_iter.size(1)
            if T_best < T_iter:
                best_tours = torch.nn.functional.pad(
                    best_tours, (0, T_iter - T_best), value=0
                )
            elif T_iter < T_best:
                tours_iter = torch.nn.functional.pad(
                    tours_iter, (0, T_best - T_iter), value=0
                )

            best_tours = torch.where(
                improved_m.unsqueeze(-1).expand_as(best_tours),
                tours_iter,
                best_tours,
            )

        ils_time_limit = None
        if stop_condition == "time":
            budget = float(num_seconds) if num_seconds is not None else 0.0
            ils_time_limit = max(0.0, budget - (time.perf_counter() - neural_start))

        # ── Search ──────────────────────────────────
        best_np = best_tours.cpu().numpy()  # ← only this moves per iteration

        ls_costs = np.empty(batch_size, dtype=np.float32)
        ls_tours_lst = [None] * batch_size
        futures_map = {}

        use_vrplib = "vrplib_coords" in td_cpu.keys()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        ) as pool:
            for i in range(batch_size):
                inst = (
                    locs_np[i],
                    dlin_np[i],
                    dbac_np[i],
                    float(dlim_np[i, 0])
                    if dlim_np.ndim == 2
                    else float(dlim_np[i]),
                    bool(open_np[i, 0])
                    if open_np.ndim == 2
                    else bool(open_np[i]),
                    tw_np[i],
                    svc_np[i],
                    int(num_depots_np[i]),
                )
                seed = (i * 100003) & 0xFFFFFFFF

                vrplib_opts = None
                if use_vrplib:
                    cap = td_cpu["vrplib_capacity"]
                    cap_i = int(cap[i, 0]) if cap.ndim > 1 else int(cap[i])
                    if "vrplib_round_func_id" in td_cpu.keys():
                        rid = int(td_cpu["vrplib_round_func_id"][i].reshape(-1)[0].item())
                        round_func = vrplib_round_func_from_id(rid)
                    else:
                        round_func = "round"
                    opts = {
                        "coords": td_cpu["vrplib_coords"][i].numpy(),
                        "demands": td_cpu["vrplib_demands"][i].numpy(),
                        "capacity": cap_i,
                        "round_func": round_func,
                    }
                    if "vrplib_edge_weight" in td_cpu.keys():
                        opts["edge_weight"] = td_cpu["vrplib_edge_weight"][i].numpy()
                    vrplib_opts = opts

                futures_map[
                    pool.submit(
                        _ls_instance_iterated,
                        inst,
                        best_np[i],
                        ls_nb_granular,
                        seed,
                        num_iters=num_iters,
                        time_limit=ils_time_limit,
                        dmax=dmax,
                        dmin=dmin,
                        gamma=gamma,
                        eta_min=eta_min,
                        vrplib_options=vrplib_opts,
                        mixed_backhaul=bool(mixed_backhaul_flags[i]),
                    )
                ] = i

            for fut in concurrent.futures.as_completed(futures_map):
                i = futures_map[fut]
                ls_costs[i], ls_tours_lst[i] = fut.result()

        # Update best with LS results
        ls_reward = torch.tensor(-ls_costs, dtype=torch.float32, device=device)
        if use_vrplib:
            # LS costs are already in CVRPLIB integer units (metric='vrplib').
            best_reward = ls_reward
        else:
            best_reward = torch.maximum(best_reward, ls_reward)

        return best_reward
