import torch
import torch.nn as nn

from utils.functions import batchify, gather_by_index

from models.layers import PromptNet

from .decoder import VRP_Decoder
from .encoder import VRP_Encoder
from .encoder_ple import VRP_Encoder as VRP_Encoder_PLE
from .layers import PrecomputedCache, reshape_by_heads


class VRPModel(nn.Module):
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

    def forward(self, td, env, reld_alpha=1.0, with_greedy=False):
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
                td, cache, num_starts, reld_alpha=reld_alpha
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
        reld_alpha=1.0,
    ):
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
