import concurrent.futures
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn

from search import _ls_instance_iterated
from search.vrplib_helpers import vrplib_round_func_from_id
from utils.functions import batchify, gather_by_index

from models.decoder import VRP_Decoder
from models.encoder import VRP_Encoder
from models.encoder_ple import VRP_Encoder as VRP_Encoder_PLE
from models.helpers import PrecomputedCache, reshape_by_heads
from models.layers import PromptNet


class VRPModel(nn.Module):
    """RouteFinder-based main model (single-stream encoder + optional PLE / PGB)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_mode = "rl"
        if self.args.model_params.get("use_ple", False):
            self.encoder = VRP_Encoder_PLE(**args.model_params)
            self.prompt_net = PromptNet(args)
        else:
            self.encoder = VRP_Encoder(**args.model_params)
            self.prompt_net = None
        self.decoder = VRP_Decoder(**args.model_params)
        self.encoded_nodes = None
        self.encoded_coords = None
        self.now_p_type = None

    @staticmethod
    def greedy(logprobs, mask=None):
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert not (~mask).gather(1, selected.unsqueeze(-1)).data.any(), (
                "infeasible action selected"
            )
        return selected

    @staticmethod
    def sampling(logprobs, log, mask=None):
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
        self.loss_mode = mode

    def _encode(self, td):
        if self.prompt_net is not None:
            prompt = self.prompt_net(td)["prompt"]
            return self.encoder(td, prompt)
        return self.encoder(td)

    def forward(self, td, env, with_greedy=False, gate_alpha=1.0):
        """Encode -> multi-start decode -> reward.

        When ``with_greedy=True`` (PO+LS), the env appends a depot-0 start that
        is decoded greedily; other POMO starts are sampled.
        """
        args = self.args
        node_embed, node_coords = self._encode(td)
        self.encoded_nodes = node_embed
        self.encoded_coords = node_coords

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

        greedy_mask = greedy_mask.to(td.device).bool()
        td = batchify(td, num_starts)

        if greedy_mask.numel() != td.batch_size[0] * num_starts:
            batch = td.batch_size[0]
            if greedy_mask.numel() == num_starts:
                greedy_mask = greedy_mask.repeat_interleave(batch)
            else:
                greedy_mask = greedy_mask.view(-1)[: (num_starts * batch)].to(
                    torch.bool
                )

        logprobs_list = [
            torch.zeros_like(start_actions, dtype=torch.float32, device=td.device)
        ]
        actions_list = [start_actions]

        td.set("action", start_actions)
        td = env.step(td)["next"]

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

        while not td["done"].all():
            logprobs, mask, cache = self.decoder(
                td, cache, num_starts, gate_alpha=gate_alpha
            )
            if self.training:
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
        }

    def route_forward(
        self,
        td,
        env,
        tours,
        tour_lengths,
        num_starts,
        node_embed=None,
        node_coords=None,
    ):
        """Compute log-likelihoods for given tours (LS / PO sync path)."""
        if tours.dim() != 2:
            raise ValueError("tours must be 2D: [batch, steps]")
        if tour_lengths.dim() != 1 or tour_lengths.size(0) != tours.size(0):
            raise ValueError("tour_lengths must be 1D with same batch as tours")

        if node_embed is None or node_coords is None:
            node_embed, node_coords = self._encode(td)

        td = batchify(td, num_starts)

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

        actions_list = []
        logprobs_list = []
        step = 0

        while not td["done"].all():
            logprobs, _, cache = self.decoder(td, cache, num_starts)
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

        node_embed, node_coords = self._encode(td_orig)

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

        if "p_s_tag" in td_cpu.keys():
            mixed_backhaul_flags = td_cpu["p_s_tag"][:, 5].numpy().astype(bool)
        else:
            mixed_backhaul_flags = np.zeros(batch_size, dtype=bool)

        best_reward = None
        best_tours = None

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

        cache = static_cache
        while not td_dec["done"].all():
            logprobs, mask, cache = self.decoder(td_dec, cache, num_starts)
            select = VRPModel.greedy(logprobs, mask)
            actions_list.append(select)
            logprobs_list.append(gather_by_index(logprobs, select, dim=1))
            td_dec.set("action", select)
            td_dec = env.step(td_dec)["next"]

        actions = torch.stack(actions_list, dim=1)
        reward_all, tours_all = env.get_reward(td_dec, actions)

        reward_2d = reward_all.view(num_starts, batch_size)
        tours_3d = tours_all.view(num_starts, batch_size, -1)
        best_start = reward_2d.argmax(dim=0)
        batch_idx = torch.arange(batch_size, device=device)
        reward_iter = reward_2d[best_start, batch_idx]
        tours_iter = tours_3d[best_start, batch_idx]

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

        best_np = best_tours.cpu().numpy()
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
                    float(dlim_np[i, 0]) if dlim_np.ndim == 2 else float(dlim_np[i]),
                    bool(open_np[i, 0]) if open_np.ndim == 2 else bool(open_np[i]),
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
                        rid = int(
                            td_cpu["vrplib_round_func_id"][i].reshape(-1)[0].item()
                        )
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

        ls_reward = torch.tensor(-ls_costs, dtype=torch.float32, device=device)
        if use_vrplib:
            best_reward = ls_reward
        else:
            best_reward = torch.maximum(best_reward, ls_reward)

        return best_reward
