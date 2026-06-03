"""
Fine-tuning module for VRP models on new variant sets.

Supports 'mb' (8 mixed backhaul), 'md' (16 multi-depot), and 'both' variants.
"""

import os
import time
import gc
import concurrent.futures
import multiprocessing as mp

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pandas as pd
import wandb
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

from utils.functions import TimeEstimator
from utils.functions import (
    batchify,
    gather_by_index,
    load_npz_to_tensordict,
    clip_grad_norms,
)
from search import Search, POLAR_SCALER
from models.model import VRPModel
from envs.transformer import StateAugmentation
from tester import VRPTester, metric2str


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Generator settings for each tuning variant
VARIANT_GEN_SETTINGS = {
    "mb": {
        "backhaul_ratio": 0.3,
        "sample_backhaul_class": False,
        "backhaul_class": 2,
        "variant_preset": "mixed_backhaul",
    },
    "md": {
        "backhaul_ratio": 0.2,
        "sample_backhaul_class": False,
        "backhaul_class": 1,
        "num_depots": 3,
        "variant_preset": "all",
    },
    "both": {
        "backhaul_ratio": 0.3,
        "sample_backhaul_class": False,
        "backhaul_class": 2,
        "num_depots": 3,
        "variant_preset": "mixed_backhaul",
    },
}


def get_test_variant_folders(data_dir, variant_present):
    """Get test data folders for specified variant type."""
    all_dirs = [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and os.path.exists(os.path.join(data_dir, d, "test"))
    ]

    if variant_present == "mb":
        return [
            d for d in all_dirs if "mb" in d.lower() and not d.lower().startswith("md")
        ]
    elif variant_present == "md":
        return [
            d for d in all_dirs if d.lower().startswith("md") and "mb" not in d.lower()
        ]
    elif variant_present == "both":
        return [d for d in all_dirs if d.lower().startswith("md") and "mb" in d.lower()]
    else:
        raise ValueError(f"Unknown variant_present: {variant_present}")


class VRPTuner:
    """Fine-tunes pretrained VRP models on unseen variants."""

    def __init__(self, args):
        clear_gpu()
        self.args = args
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        self._build_core_components()
        self._configure_training_tools()
        self._load_checkpoint()
        self._setup_distributed_training()
        self._prepare_evaluation_artifacts()

        self.time_estimator = TimeEstimator()
        self.console = Console()
        self._ls_executor = None

    def _build_core_components(self):
        """Initialize model and environment for fine-tuning."""
        args = self.args

        self.model = VRPModel(args)

        tuner_params = args.tuner_params
        self.variant_present = tuner_params.get("variant_present", "mb")

        # Select environment based on variant type
        if self.variant_present == "mb":
            from envs.mtvrp.env import MTVRPEnv

            gen_params = {
                **args.env.get("generator_params", {}),
                **VARIANT_GEN_SETTINGS[self.variant_present],
                "subsample": True,
            }
            self.env = MTVRPEnv(
                generator_params=gen_params, data_dir=args.env.get("data_dir", "./data")
            )
        else:
            from envs.mtdvrp.env import MTVRPEnv as MDEnv

            gen_params = {
                **args.env.get("generator_params", {}),
                **VARIANT_GEN_SETTINGS[self.variant_present],
                "subsample": True,
            }
            self.env = MDEnv(
                generator_params=gen_params,
                data_dir=args.env.get("data_dir", "./data"),
            )

        self.use_amp = bool(tuner_params.get("use_amp", True))

        # AMP dtype: BF16 for Ampere+, otherwise FP16
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            self.supports_bf16 = capability[0] >= 8
            self.amp_dtype = torch.bfloat16 if self.supports_bf16 else torch.float16
        else:
            self.supports_bf16 = False
            self.amp_dtype = torch.float16

        self.loss_function = tuner_params.get("loss_function", "po")
        if hasattr(self.env, "set_loss_mode"):
            self.env.set_loss_mode(self.loss_function)
        if hasattr(self.model, "set_loss_mode"):
            self.model.set_loss_mode(self.loss_function)

    def _configure_training_tools(self):
        """Setup optimizer, scheduler, and AMP gradient scaler."""
        args = self.args
        tuner_params = args.tuner_params

        # GradScaler only needed for FP16; BF16 shares FP32 exponent range
        use_scaler = self.use_amp and not self.supports_bf16
        self.scaler = GradScaler(enabled=use_scaler)
        self.use_scaler = use_scaler

        # Load optimizer params from config
        opt_params = args.tuner_optimizer_params
        lr = float(opt_params["optimizer"].get("lr", 1e-4))
        weight_decay = float(opt_params["optimizer"].get("weight_decay", 1e-6))
        betas = (
            float(opt_params["optimizer"].get("beta1", 0.9)),
            float(opt_params["optimizer"].get("beta2", 0.999)),
        )
        milestones = opt_params["scheduler"].get("milestones", [8])
        gamma = opt_params["scheduler"].get("gamma", 0.1)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=gamma
        )
        args.log(
            f"Tuner optimizer: lr={lr}, weight_decay={weight_decay}, milestones={milestones}"
        )

    def _load_checkpoint(self):
        """Load pretrained model from checkpoint."""
        args = self.args
        self.start_epoch = 1

        model_load = args.trainer_params.get("model_load", {})
        if not model_load.get("enable", False):
            args.log("Warning: No checkpoint loaded. Starting from scratch.")
            return

        checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
        checkpoint = torch.load(
            checkpoint_fullname,
            map_location="cuda",
            weights_only=False,
        )

        model_state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(model_state_dict, strict=True)

        args.log(f"Pretrained model loaded from {checkpoint_fullname}")

    def _setup_distributed_training(self):
        """Wrap model with DDP for distributed training."""
        args = self.args
        if not args.ddp:
            return

        dist.barrier()
        self.model = DistributedDataParallel(self.model)
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        args.log(f"Using DDP, current device: {torch.cuda.current_device()}")

    def _prepare_evaluation_artifacts(self):
        """Create dataloaders and tester for evaluation."""
        args = self.args

        self.augmentation = StateAugmentation()
        self.test_dataloader = self._create_variant_dataloaders()
        self.tester = VRPTester(self.model, self.env, self.augmentation, args)

    def _create_variant_dataloaders(self):
        """Create dataloaders for variant test sets."""
        args = self.args
        data_dir = args.env.get("data_dir", "./data")
        sizes = [args.n_size]

        variant_folders = get_test_variant_folders(data_dir, self.variant_present)
        args.log(
            f"Found {len(variant_folders)} variant folders for {self.variant_present}: {variant_folders}"
        )

        dataloaders = {}
        for variant in variant_folders:
            for size in sizes:
                test_file = os.path.join(data_dir, variant, "test", f"{size}.npz")
                sol_file = os.path.join(
                    data_dir, variant, "test", f"{size}_sol_pyvrp.npz"
                )

                if not os.path.exists(test_file):
                    continue

                td = load_npz_to_tensordict(test_file).to("cpu")

                if os.path.exists(sol_file):
                    sol_data = np.load(sol_file)
                    if "costs" in sol_data:
                        costs = sol_data["costs"]
                        # Convert negative rewards to positive costs if needed
                        td["opt_cost"] = torch.tensor(
                            np.abs(costs), dtype=torch.float32
                        )
                else:
                    td["opt_cost"] = torch.ones(td.batch_size[0])

                td = self._build_p_s_tag(td, variant, size)
                td = self._add_missing_fields(td)

                name = f"{size}_{variant}_uniform"
                dataloaders[name] = td

        result = {}
        size_bs = {50: 500, 100: 500, 200: 125, 300: 100}

        for name, td in dataloaders.items():
            size = int(name.split("_")[0])
            batch_size = size_bs.get(size, 500)

            class TDDataset:
                def __init__(self, td, batch_size):
                    self.td = td
                    self.batch_size = batch_size

                def __len__(self):
                    return (
                        self.td.batch_size[0] + self.batch_size - 1
                    ) // self.batch_size

                def __iter__(self):
                    for i in range(0, self.td.batch_size[0], self.batch_size):
                        yield self.td[
                            i : min(i + self.batch_size, self.td.batch_size[0])
                        ]

            result[name] = TDDataset(td, batch_size)

        return result

    def _add_missing_fields(self, td):
        """Add missing fields to tensordict with defaults."""
        batch = td.batch_size[0]
        n_locs = td["locs"].shape[1]
        device = td["locs"].device

        # Demand linehaul
        if "demand_linehaul" in td.keys():
            if td["demand_linehaul"].shape[1] == n_locs - 1:
                td["demand_linehaul"] = torch.cat(
                    [torch.zeros(batch, 1, device=device), td["demand_linehaul"]], dim=1
                )
        else:
            td["demand_linehaul"] = torch.zeros(batch, n_locs, device=device)

        # Demand backhaul
        if "demand_backhaul" not in td.keys():
            td["demand_backhaul"] = torch.zeros(batch, n_locs, device=device)
        elif td["demand_backhaul"].shape[1] == n_locs - 1:
            td["demand_backhaul"] = torch.cat(
                [torch.zeros(batch, 1, device=device), td["demand_backhaul"]], dim=1
            )

        # Time windows
        if "time_windows" not in td.keys():
            tw = torch.zeros(batch, n_locs, 2, device=device)
            tw[..., 1] = float("inf")
            td["time_windows"] = tw

        # Service time
        if "service_time" not in td.keys():
            td["service_time"] = torch.zeros(batch, n_locs, device=device)
        elif td["service_time"].shape[1] == n_locs - 1:
            td["service_time"] = torch.cat(
                [torch.zeros(batch, 1, device=device), td["service_time"]], dim=1
            )

        # Open route
        if "open_route" not in td.keys():
            td["open_route"] = torch.zeros(batch, 1, dtype=torch.bool, device=device)

        # Distance limit
        if "distance_limit" not in td.keys():
            td["distance_limit"] = torch.full((batch, 1), float("inf"), device=device)

        # Vehicle capacity
        if "vehicle_capacity" not in td.keys():
            td["vehicle_capacity"] = torch.ones(batch, 1, device=device)

        # Capacity original
        if "capacity_original" not in td.keys():
            td["capacity_original"] = torch.ones(batch, 1, device=device)

        # Speed
        if "speed" not in td.keys():
            td["speed"] = torch.ones(batch, 1, device=device)

        return td

    def _build_p_s_tag(self, td, variant_name, size):
        """Build p_s_tag tensor from variant name."""
        batch_size = td.batch_size[0]
        variant_lower = variant_name.lower()

        has_open = "ovrp" in variant_lower or (
            variant_lower.startswith("o") and "vrp" in variant_lower
        )
        # For md variants, check if 'o' comes after 'md'
        if variant_lower.startswith("md"):
            has_open = "mdo" in variant_lower or "mdovrp" in variant_lower
        has_tw = "tw" in variant_lower
        has_limit = "l" in variant_lower and (
            "ltw" in variant_lower
            or "bl" in variant_lower
            or variant_lower.endswith("l")
        )
        has_mixed_backhaul = "mb" in variant_lower
        has_backhaul = "b" in variant_lower and not has_mixed_backhaul
        has_multi_depot = variant_lower.startswith("md")

        p_s_tag = torch.zeros((batch_size, 8), dtype=torch.float32)
        p_s_tag[:, 0] = float(not has_open)  # C
        p_s_tag[:, 1] = float(has_open)  # O
        p_s_tag[:, 2] = float(has_tw)  # TW
        p_s_tag[:, 3] = float(has_limit)  # L
        p_s_tag[:, 4] = float(has_backhaul)  # B
        p_s_tag[:, 5] = float(has_mixed_backhaul)  # MB
        p_s_tag[:, 6] = float(has_multi_depot)  # MD
        p_s_tag[:, 7] = size / 2000.0  # size

        td["p_s_tag"] = p_s_tag
        return td

    def _start_ls_executor(self):
        """Create the process pool for parallel local search (CPU-bound)."""
        if self._ls_executor is not None:
            return
        worker_count = min(32, os.cpu_count() or 1)
        self._ls_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count, mp_context=mp.get_context("spawn")
        )
        self.args.log(f"Local search process pool started with {worker_count} workers.")

    def _stop_ls_executor(self):
        """Shutdown the process pool."""
        if self._ls_executor is not None:
            self._ls_executor.shutdown(wait=True)
            self._ls_executor = None
            self.args.log("Local search process pool stopped.")

    def _run_local_search(self, td, reward, tours, num_instances):
        """Run local search synchronously on a batch. Blocks until all results are ready.

        Always improves the single best solution per instance (ls_B is fixed at 1).

        Args:
            td: TensorDict with instance data (on GPU).
            reward: (pomo, batch) reward tensor.
            tours: (pomo, batch, seq_len) tour tensor.
            num_instances: how many instances from the batch to improve.

        Returns:
            list of improvement dicts, each with keys:
                batch_idx, pomo_idx, tour (list[int]), cost (float)
        """
        batch_size = reward.size(1)
        if num_instances <= 0:
            return []
        num_instances = min(num_instances, batch_size)
        candidate_batches = torch.randperm(batch_size)[:num_instances]
        num_sols = 1  # Always improve only the single best solution per instance

        # Move data to CPU/numpy for the LS workers
        locs = td["locs"].detach().cpu().numpy()
        demands_linehaul = td["demand_linehaul"].detach().cpu().numpy()
        demands_backhaul = td["demand_backhaul"].detach().cpu().numpy()
        distance_limit = td["distance_limit"].detach().cpu().numpy()
        open_route = td["open_route"].detach().cpu().numpy()
        time_windows = td["time_windows"].detach().cpu().numpy()
        service_time = td["service_time"].detach().cpu().numpy()
        mixed_backhaul = td["p_s_tag"].detach().cpu().numpy()[:, 5]
        if "num_depots" in td.keys():
            num_depots = td["num_depots"].detach().cpu().numpy()
        else:
            num_depots = None

        reward_np = reward.detach().cpu()
        tours_np = tours.detach().cpu().numpy()

        instance_jobs = []
        for batch_idx in candidate_batches.tolist():
            rewards_batch = reward_np[:, batch_idx]
            pomo_indices = torch.topk(
                rewards_batch, k=num_sols, largest=True
            ).indices.tolist()
            instance_jobs.append(
                {
                    "batch_idx": batch_idx,
                    "args": (
                        locs[batch_idx],
                        demands_linehaul[batch_idx],
                        demands_backhaul[batch_idx],
                        distance_limit[batch_idx],
                        open_route[batch_idx],
                        time_windows[batch_idx],
                        service_time[batch_idx],
                        num_depots[batch_idx].item() if num_depots is not None else 1,
                        mixed_backhaul[batch_idx],
                    ),
                    "pomo_indices": pomo_indices,
                    "tours": [
                        tours_np[pomo_idx, batch_idx] for pomo_idx in pomo_indices
                    ],
                }
            )

        if not instance_jobs:
            return []

        # Submit all instance jobs to the process pool and wait for completion
        futures = {
            self._ls_executor.submit(
                _run_instance_search,
                job["batch_idx"],
                job["args"],
                job["pomo_indices"],
                job["tours"],
                self.args.tuner_params.get("ls_nb_granular", 20),
            ): job["batch_idx"]
            for job in instance_jobs
        }

        raw_improvements = []
        for fut in concurrent.futures.as_completed(futures):
            raw_improvements.extend(fut.result())

        return raw_improvements

    def _apply_local_search_improvements(
        self, td_init, reward, log_likelihood, improvements
    ):
        """Apply improvements found by local search to rewards and log-likelihoods.

        For each LS-improved tour (one per instance — always the best POMO solution),
        runs route_forward to obtain the model's reward and log-likelihood for the
        improved tour, then REPLACES the original entry in-place if the LS solution
        has a strictly lower cost (higher reward).  No new rows are appended.

        Filter: LS solution must strictly beat its own original POMO counterpart.
        """
        if len(improvements) == 0:
            return reward, log_likelihood

        reward = reward.clone()
        log_likelihood = log_likelihood.clone()
        device = reward.device

        # ── Build a flat batch of LS tours for route_forward ──────────────────
        seen_bidx: set = set()
        unique_batch_indices = []
        for imp in improvements:
            bidx = imp["batch_idx"]
            if bidx not in seen_bidx:
                seen_bidx.add(bidx)
                unique_batch_indices.append(bidx)

        batch_indices = torch.tensor(unique_batch_indices, device=device)
        bidx_to_local = {bidx: i for i, bidx in enumerate(unique_batch_indices)}

        num_flat = len(improvements)
        tour_lengths = torch.tensor(
            [len(imp["tour"]) for imp in improvements], device=device
        )
        max_len = int(tour_lengths.max().item())

        tours_tensor = torch.zeros((num_flat, max_len), dtype=torch.long, device=device)
        for idx, imp in enumerate(improvements):
            seq = torch.tensor(imp["tour"], dtype=torch.long, device=device)
            tours_tensor[idx, : seq.size(0)] = seq

        local_instance_idx = torch.tensor(
            [bidx_to_local[imp["batch_idx"]] for imp in improvements], device=device
        )

        td_unique = td_init[batch_indices].clone(recurse=True)
        td_flat = td_unique[local_instance_idx].clone(recurse=True)

        node_embed_unique = self.model.encoded_nodes[batch_indices]
        node_coords_unique = self.model.encoded_coords[batch_indices]
        node_embed_flat = node_embed_unique[local_instance_idx]
        node_coords_flat = node_coords_unique[local_instance_idx]

        use_ccl = getattr(self.model, "decoder", None) and getattr(
            self.model.decoder, "use_ccl", False
        )
        if use_ccl:
            ls_out = self.model.route_forward_ccl(
                td_flat,
                self.env,
                tours_tensor,
                tour_lengths,
                node_embed=node_embed_flat,
                node_coords=node_coords_flat,
            )
        else:
            ls_out = self.model.route_forward(
                td_flat,
                self.env,
                tours_tensor,
                tour_lengths,
                num_starts=1,
                node_embed=node_embed_flat,
                node_coords=node_coords_flat,
            )

        ls_rewards = ls_out["reward"]  # (num_flat,)
        ls_log_ll = ls_out["log_likelihood"].sum(1)  # (num_flat,)

        # ── Replace in-place when LS strictly improves the original ───────────
        for idx, imp in enumerate(improvements):
            bidx = imp["batch_idx"]
            pomo_idx = imp["pomo_idx"]
            ls_r = ls_rewards[idx].item()
            original_reward = reward[pomo_idx, bidx].item()

            if ls_r > original_reward:
                reward[pomo_idx, bidx] = ls_rewards[idx]
                log_likelihood[pomo_idx, bidx] = ls_log_ll[idx]

        return reward, log_likelihood

    def _compute_po_loss(self, reward, log_likelihood):
        """Compute Preference Optimization loss over solution pairs."""
        preference = reward[:, :, None] > reward[:, None, :]
        log_prob = self.args.tuner_params.get("po_alpha", 0.03) * log_likelihood
        log_prob_pair = log_prob[:, :, None] - log_prob[:, None, :]
        pf_log = torch.log(F.sigmoid(log_prob_pair))
        loss = -torch.mean(pf_log * preference)
        return loss

    def run(self):
        """Main fine-tuning loop with synchronous local search."""
        args = self.args
        tuner_params = args.tuner_params
        epochs = tuner_params.get("epochs", 10)

        self.time_estimator.reset(self.start_epoch)

        # Check for local search configuration
        use_ls_config = tuner_params.get("use_ls", False) and self.loss_function == "po"
        ls_start_epoch = tuner_params.get("ls_start_epoch", 1)
        ls_executor_started = False

        if args.test_only:
            args.log("=== Test-only mode ===")
            self._test_variants(0)
            return

        if args.test and self.start_epoch == 1:
            args.log("=== Zero-shot evaluation before fine-tuning ===")
            self._test_variants(0)

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                args.log(
                    "================================================================="
                )

                # Activate local search n epochs
                use_ls = use_ls_config and epoch >= ls_start_epoch
                if use_ls and not ls_executor_started:
                    self._start_ls_executor()
                    ls_executor_started = True
                    args.log(f"Local search activated at epoch {epoch}")

                if args.wandb != "" and not args.mute:
                    wandb.log({"lr": self.optimizer.param_groups[0]["lr"]}, step=epoch)

                start_time = time.time()
                self.model.train()

                train_episodes = tuner_params.get("train_episodes", 10000)
                batch_size = args.batch_size

                train_label = (
                    f"Tune | Epoch{str(epoch).zfill(3)}/{str(epochs).zfill(3)}"
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("-"),
                    TimeElapsedColumn(),
                    TextColumn("+"),
                    TimeRemainingColumn(),
                    console=self.console,
                    transient=True,
                ) as progress:
                    train_task = progress.add_task(train_label, total=train_episodes)

                    all_metric = []
                    ls_update_count = 0
                    instances_seen = 0
                    step = 0

                    while instances_seen < train_episodes:
                        if args.skip and instances_seen > 2 * batch_size:
                            break

                        self.optimizer.zero_grad()

                        remaining = train_episodes - instances_seen
                        current_batch_size = min(batch_size, remaining)

                        # Generate new instances
                        self.env.generator.num_loc = args.n_size
                        td = self.env.reset(batch_size=current_batch_size).to("cuda")
                        td_initial = td.clone(recurse=True)

                        if args.ddp:
                            torch.distributed.barrier()

                        with torch.amp.autocast(
                            device_type="cuda",
                            dtype=self.amp_dtype,
                            enabled=self.use_amp,
                        ):
                            out = self.model(td, self.env, with_greedy=use_ls)
                            reward = out["reward"].view(-1, current_batch_size)

                            if self.loss_function == "rl":
                                log_likelihood = (
                                    out["log_likelihood"].sum(1).view(-1, current_batch_size)
                                )
                                advantage = reward - reward.mean(dim=0, keepdims=True)
                                loss = -(advantage * log_likelihood).mean()
                            elif self.loss_function == "po":
                                log_likelihood = (
                                    out["log_likelihood"].sum(1).view(-1, current_batch_size)
                                )
                                if use_ls:
                                    tours = out["tours"].view(
                                        -1, current_batch_size, out["tours"].size(1)
                                    )
                                    # ---- Synchronous Local Search ----
                                    improvements = self._run_local_search(
                                        td_initial,
                                        reward,
                                        tours,
                                        num_instances=current_batch_size,
                                    )

                                    # Step 2: Apply LS improvements in-place
                                    if improvements:
                                        reward, log_likelihood = (
                                            self._apply_local_search_improvements(
                                                td_initial,
                                                reward,
                                                log_likelihood,
                                                improvements,
                                            )
                                        )
                                        ls_update_count += 1

                                # Step 3: Compute PO loss over combined solutions
                                loss = self._compute_po_loss(
                                    reward.transpose(0, 1),
                                    log_likelihood.transpose(0, 1),
                                )
                            else:
                                raise ValueError(
                                    f"Unknown loss function: {self.loss_function}"
                                )

                            max_pomo_reward, _ = reward.max(dim=0)
                            score_mean = -max_pomo_reward.float().mean()

                        if args.ddp:
                            torch.distributed.barrier()
                        
                        # Backward + optimizer step (always, for every batch)
                        if hasattr(self.model, "aux_loss"):
                            loss = loss + self.model.aux_loss
                        (self.scaler.scale(loss) if self.use_scaler else loss).backward()

                        # Clip gradients for RL mode
                        if self.loss_function == "rl":
                            grad_norms, grad_norms_clipped = clip_grad_norms(
                                self.optimizer.param_groups, 1.0
                            )

                        # Optimizer step
                        if self.use_scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        metric_list = [loss.item(), score_mean.item()]
                        all_metric.append(metric_list)

                        instances_seen += current_batch_size
                        step += 1

                        del td, td_initial, out, reward, loss
                        if "log_likelihood" in locals():
                            del log_likelihood
                        if "tours" in locals():
                            del tours
                        if "advantage" in locals():
                            del advantage
                        if "improvements" in locals():
                            del improvements

                        # Log
                        metric_info = "|".join(
                            [
                                f"{args.metric_label[i]} {metric_list[i]:.4f}"
                                for i in range(len(args.metric_label))
                            ]
                        )
                        ls_info = f"|LS:{ls_update_count}" if use_ls else ""
                        progress.update(
                            train_task,
                            description=f"{train_label} | {metric_info}{ls_info}",
                            advance=current_batch_size,
                        )

                if args.ddp:
                    torch.distributed.barrier()

                # Log Once, for each epoch
                metric_tensor = torch.tensor(all_metric).mean(dim=0)
                metric_list = metric_tensor.tolist()
                metric_info = "|".join(
                    [
                        f"{args.metric_label[i]} {metric_list[i]:.4f}"
                        for i in range(len(args.metric_label))
                    ]
                )
                elapsed = time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - start_time)
                )
                self.console.print(
                    f"[green]✓ Fine-tuning complete in {elapsed}[/green]"
                )
                ls_info = f"|LS updates:{ls_update_count}" if use_ls else ""
                args.log(
                    f"{train_label}|{elapsed}|{metric_info}|LR {self.optimizer.param_groups[0]['lr']:.2e}{ls_info}"
                )

                test_interval = tuner_params.get("test_interval", 5)
                if args.test and epoch % test_interval == 0:
                    self._test_variants(epoch)

                if self.scheduler is not None:
                    self.scheduler.step()

                elapsed_str, remain_str = self.time_estimator.get_est_string(
                    epoch, epochs
                )
                args.log(
                    "Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                        epoch, epochs, elapsed_str, remain_str
                    )
                )

                save_interval = tuner_params.get("model_save_interval", 5)
                if (epoch == epochs or epoch % save_interval == 0) and not args.mute:
                    args.log("Saving fine-tuned model")
                    checkpoint_dict = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict()
                        if not args.ddp
                        else self.model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "variant_present": self.variant_present,
                    }
                    if self.scheduler is not None:
                        checkpoint_dict["scheduler_state_dict"] = (
                            self.scheduler.state_dict()
                        )
                    if self.use_amp:
                        checkpoint_dict["scaler_state_dict"] = self.scaler.state_dict()

                    torch.save(
                        checkpoint_dict,
                        f"{args.result_dir}/tuned-{self.variant_present}-{epoch}.pt",
                    )

                if args.ddp:
                    torch.distributed.barrier()

            args.log(" *** Fine-tuning Complete *** ")

        finally:
            # Stop local search executor
            if ls_executor_started:
                self._stop_ls_executor()

    @torch.inference_mode()
    def _test_variants(self, epoch):
        """Test on variant test sets."""
        args = self.args
        self.model.eval()

        results = self.tester.test_tuning_variants(
            epoch=epoch,
            data_dir=args.env.get("data_dir", "./data"),
            variant_present=self.variant_present,
            sizes=[args.n_size],
            solution_source="pyvrp",
        )

        if results:
            avg_gap = sum(r["aug_gap"] for r in results.values()) / len(results)
            avg_score = sum(r["aug_score"] for r in results.values()) / len(results)
            args.log(
                f"Epoch {epoch} | Avg Aug Gap: {avg_gap:.3f}% | Avg Aug Score: {avg_score:.3f}"
            )

            if args.wandb != "" and not args.mute:
                wandb.log(
                    {
                        "avg_aug_gap": avg_gap,
                        "avg_aug_score": avg_score,
                        **{f"{k}_gap": v["aug_gap"] for k, v in results.items()},
                    },
                    step=epoch,
                )

        if hasattr(args, "result_dir") and results:
            df_data = {
                "Variant": list(results.keys()),
                "Score": [r["score"] for r in results.values()],
                "Aug Score": [r["aug_score"] for r in results.values()],
                "Gap": [r["gap"] for r in results.values()],
                "Aug Gap": [r["aug_gap"] for r in results.values()],
            }
            df = pd.DataFrame(df_data)
            df.to_excel(
                f"{args.result_dir}/tuning_results_{self.variant_present}_{epoch}.xlsx",
                index=False,
            )


def _run_instance_search(batch_idx, instance_args, pomo_indices, tours, nb_granular=20):
    """Run local search on a single instance."""
    (
        locs,
        demands_linehaul,
        demands_backhauls,
        distance_limit,
        open_route,
        time_windows,
        service_time,
        num_depots,
        mixed_backhaul,
    ) = instance_args
    search = Search(
        locs,
        demands_linehaul,
        demands_backhauls,
        distance_limit,
        open_route,
        time_windows,
        service_time,
        num_depots,
        mixed_backhaul,
        nb_granular=nb_granular,
        scaler=POLAR_SCALER,
    )
    improvements = []
    for pomo_idx, tour in zip(pomo_indices, tours):
        # Unique seed per (batch_idx, pomo_idx) → diverse LS trajectories.
        # Bounded to uint32 range required by RandomNumberGenerator.
        seed = (batch_idx * 100003 + pomo_idx * 1000003) & 0xFFFFFFFF
        cost, improved_tour = search.build_solution(tour, seed=seed)
        improvements.append(
            {
                "batch_idx": batch_idx,
                "pomo_idx": pomo_idx,
                "tour": improved_tour,
                "cost": cost,
            }
        )
    return improvements
