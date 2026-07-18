import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import random
import time
import vrplib
import pandas as pd
from math import ceil
import torch
import torch.distributed as dist
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
import concurrent.futures
import multiprocessing as mp
import wandb
import gc
import torch.nn.functional as F

from utils.functions import *
from models.model import VRPModel
from envs.mtvrp import MTVRPEnv, get_dataloader
from envs.transformer import StateAugmentation
from search import Search, POLAR_SCALER
from tester import VRPTester


TRAIN_METRIC_LABELS = ("loss", "cost")


def clear_gpu():
    """Clear GPU memory by collecting garbage and emptying CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def transform_dict_to_mean(dict_):
    """Convert list values in dict to their mean for metric aggregation."""
    for k, v in dict_.items():
        dict_[k] = torch.tensor(v).mean().item()


def cal_model_size(model, args):
    """Log the parameter and buffer counts of the model to help size tracking."""
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    args.log("Total number of parameters: {}".format(param_count))
    args.log("Total number of buffer elements: {}".format(buffer_count))


class VRPTrainer:
    def __init__(self, args):
        clear_gpu()
        self.args = args
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Build core components: model, environment, loss settings
        self._build_core_components()
        # Configure optimizer, scheduler, and AMP scaler
        self._configure_training_tools()
        # Restore from checkpoint if resuming
        self._restore_from_checkpoint()
        # Setup DDP if enabled
        self._setup_distributed_training()
        # Setup validation artifacts
        self._prepare_evaluation_artifacts()

        self.time_estimator = TimeEstimator()

        # Synchronous local search: process pool only (no threads, no queues)
        self._ls_executor = None
        self.console = Console()

    def _build_core_components(self):
        """Initialize model, environment, and training configuration."""
        args = self.args

        # Create MTL model
        self.model = VRPModel(args)
        cal_model_size(self.model, args)

        # Environment generates VRP instances and manages constraints
        self.env = MTVRPEnv(**args.env)
        self.use_amp = bool(args.trainer_params.get("use_amp", False))

        # AMP dtype: BF16 for Ampere+ (compute >= 8.0), otherwise FP16
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            self.supports_bf16 = capability[0] >= 8
            if self.use_amp:
                if self.supports_bf16:
                    self.amp_dtype = torch.bfloat16
                    args.log(
                        f"Using BFloat16 AMP (GPU: {torch.cuda.get_device_name()}, Compute Capability: {capability[0]}.{capability[1]})"
                    )
                else:
                    self.amp_dtype = torch.float16
                    args.log(
                        f"Using FP16 AMP - BF16 not supported (GPU: {torch.cuda.get_device_name()}, Compute Capability: {capability[0]}.{capability[1]})"
                    )
        else:
            self.supports_bf16 = False
            self.amp_dtype = torch.float16

        # Loss function: "po" (Preference Optimization) or "rl" (REINFORCE)
        self.loss_function = args.trainer_params.get("loss_function", "rl")
        self.po_mode = self.loss_function == "po"
        self.po_top_k = int(args.trainer_params.get("po_top_k", 4))

        # Set loss mode to environment and model
        if hasattr(self.env, "set_loss_mode"):
            self.env.set_loss_mode(self.loss_function)
        if hasattr(self.model, "set_loss_mode"):
            self.model.set_loss_mode(self.loss_function)

    def _configure_training_tools(self):
        """Setup optimizer, scheduler, and AMP gradient scaler."""
        args = self.args
        # GradScaler only needed for FP16
        use_scaler = self.use_amp and not self.supports_bf16
        self.scaler = GradScaler(enabled=use_scaler)
        if self.use_amp and self.supports_bf16:
            args.log("GradScaler disabled for BFloat16 (not needed)")
        self.use_scaler = use_scaler

        opt_conf = args.optimizer_params["optimizer"]

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_conf["lr"],
            weight_decay=opt_conf.get("weight_decay", 0),
        )

        # LR scheduler
        sched_conf = args.optimizer_params["scheduler"]
        if sched_conf["name"] == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=sched_conf["milestones"],
                gamma=sched_conf["gamma"],
            )
        else:
            raise NotImplementedError

    def _restore_from_checkpoint(self):
        """Optionally restore model/optimizer/env state while keeping RNGs aligned."""
        args = self.args
        self.start_epoch = 1
        model_load = args.trainer_params["model_load"]
        if not model_load["enable"]:
            return

        checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
        checkpoint = torch.load(
            checkpoint_fullname, map_location="cuda", weights_only=False
        )
        model_state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(model_state_dict, strict=True)
        self.start_epoch = 1 + model_load["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            sched_conf = args.optimizer_params["scheduler"]
            if sched_conf["name"] == "MultiStepLR":
                from collections import Counter

                self.scheduler.milestones = Counter(sched_conf["milestones"])
                self.scheduler.gamma = sched_conf["gamma"]
        elif self.scheduler is not None:
            self.scheduler.last_epoch = model_load["epoch"] - 1

        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        args.log(f"Saved Model Loaded from {checkpoint_fullname}.")

        if not args.ddp:
            self.env.__setstate__(checkpoint["env_state_dict"])
            torch.set_rng_state(checkpoint["rng_state_dict"]["torch.rng_state"].cpu())
            try:
                state = checkpoint["rng_state_dict"]["torch.cuda.rng_state"]
                torch.cuda.set_rng_state(state.cpu())
            except Exception as exc:
                print("Warning: Could not restore CUDA RNG state:", exc)
            random.setstate(checkpoint["rng_state_dict"]["random.state"])
        else:
            self.env.__setstate__(checkpoint["env_state_dict"], set_seed=False)

        self.env.data_dir = args.env["data_dir"]

    def _setup_distributed_training(self):
        """Wrap the model with DDP and broadcast parameters when requested."""
        args = self.args
        if not args.ddp:
            return

        dist.barrier()
        self.model = DistributedDataParallel(self.model)
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        args.log(f"use ddp, current device:{torch.cuda.current_device()}")

    def _prepare_evaluation_artifacts(self):
        """Create dataloaders/augmentations needed for validation runs."""
        args = self.args
        if not args.test and not args.test_only:
            return

        self.test_dataloader = get_dataloader(
            self.env.dataset(phase="test", data_size=args.env["test_episodes"]),
            batch_size=args.env["test_batch_size"],
            ddp=args.ddp,
            num_workers=args.num_workers,
        )
        self.augmentation = StateAugmentation()

        # Create VRPTester instance for evaluation
        self.tester = VRPTester(self.model, self.env, self.augmentation, args)

    # =========================================================================
    # Synchronous Local Search
    # =========================================================================

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
                self.args.trainer_params.get("ls_nb_granular", 20),
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
        # Binary preference matrix: 1 if solution i is better than j
        preference = reward[:, :, None] > reward[:, None, :]
        # shape: (batch, pomo, pomo)

        # Temperature-scaled log-probabilities
        log_prob = self.args.trainer_params["po_alpha"] * log_likelihood
        log_prob_pair = log_prob[:, :, None] - log_prob[:, None, :]

        pf_log = F.logsigmoid(log_prob_pair)
        loss = -torch.mean(pf_log * preference)
        return loss

    # =========================================================================
    # Test helpers
    # =========================================================================

    def _test(self, epoch, dataloader=None):
        """Run evaluation."""
        clear_gpu()

        if dataloader is not None:
            return self.tester.test(epoch, dataloader)
        else:
            return self.tester.test_lib(epoch)

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def run(self):
        """Main training loop with synchronous local search.

        For every batch in PO mode with LS enabled:
          1. Forward pass -> get reward, log_likelihood, tours
          2. Run local search synchronously (blocks until done)
          3. route_forward on LS-improved tours to get their log-likelihoods
          4. Concatenate LS solutions with original POMO solutions
          5. Compute PO loss over the combined set
          6. Backward + optimizer step

        No threads, no queues, no async — fully linear.
        """
        args = self.args
        self.time_estimator.reset(self.start_epoch)

        # test before training
        if args.test_lib:
            self._test(self.start_epoch - 1)
            return

        if args.test and self.start_epoch == 1:
            self._test(self.start_epoch - 1, self.test_dataloader)

        if args.test_only:
            self._test(self.start_epoch - 1, self.test_dataloader)
            exit(0)

        # LS configuration
        use_ls_config = args.trainer_params.get("use_ls", False) and self.po_mode
        ls_start_epoch = args.trainer_params.get("ls_start_epoch", 1)
        ls_executor_started = False

        try:
            # begin train
            for epoch in range(self.start_epoch, args.trainer_params["epochs"] + 1):
                args.log(
                    "================================================================="
                )

                # Activate local search after warmup period
                use_ls = use_ls_config and epoch > ls_start_epoch
                if use_ls and not ls_executor_started:
                    self._start_ls_executor()
                    ls_executor_started = True
                    args.log(f"Local search activated at epoch {epoch}")

                if args.wandb != "" and not args.mute:
                    wandb.log({f"lr": self.optimizer.param_groups[0]["lr"]}, step=epoch)

                # Training loop
                start_time = time.time()
                self.model.train()
                train_label = f"Train | Epoch{str(epoch).zfill(3)}/{str(args.trainer_params['epochs']).zfill(3)}"

                # Calculate training steps from total episodes and batch size
                train_episodes = args.trainer_params["train_episodes"]
                full_batch_size = args.batch_size
                num_full_batches = train_episodes // full_batch_size
                remaining_instances = train_episodes % full_batch_size
                total_steps = num_full_batches + (1 if remaining_instances > 0 else 0)

                # Rich progress bar for training
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("+"),
                    TimeRemainingColumn(),
                    console=self.console,
                    transient=True,
                ) as progress:
                    train_task = progress.add_task(
                        train_label,
                        total=train_episodes,
                    )
                    all_metric = []
                    ls_update_count = 0
                    instances_seen = 0
                    step = 0

                    while instances_seen < train_episodes:
                        if args.skip and step > 2:
                            break

                        self.optimizer.zero_grad()
                        n_loc = args.n_size

                        # Dynamic batch size for final incomplete batch
                        remaining = train_episodes - instances_seen
                        batch_size = min(full_batch_size, remaining)

                        self.env.generator.reset_n_loc(n_loc)
                        td = self.env.reset(batch_size=batch_size).to("cuda")
                        td_initial = td.clone(recurse=True)
                        if args.ddp:
                            torch.distributed.barrier()
                        with torch.amp.autocast(
                            device_type="cuda",
                            dtype=self.amp_dtype,
                            enabled=self.use_amp,
                        ):
                            out = self.model(
                                td, self.env, with_greedy=use_ls
                            )
                            reward = out["reward"].view(-1, batch_size)
                            if self.loss_function == "rl":
                                log_likelihood = (
                                    out["log_likelihood"].sum(1).view(-1, batch_size)
                                )
                                advantage = reward - reward.mean(dim=0, keepdims=True)
                                loss = -(advantage * log_likelihood).mean()
                            elif self.loss_function == "po":
                                log_likelihood = (
                                    out["log_likelihood"].sum(1).view(-1, batch_size)
                                )
                                tours = out["tours"].view(
                                    -1, batch_size, out["tours"].size(1)
                                )

                                # ---- Synchronous Local Search ----
                                # When use_ls=True, model() already included a greedy row
                                # (last row of reward/log_likelihood/tours) via with_greedy.
                                if use_ls:
                                    # Step 1: Run LS on the best solution per instance.
                                    # reward/tours already include the greedy row, so the
                                    # greedy tour is a valid candidate for LS as well.
                                    improvements = self._run_local_search(
                                        td_initial,
                                        reward,
                                        tours,
                                        num_instances=batch_size,
                                    )

                                    # Step 2: Apply LS improvements in-place — replaces
                                    # the corresponding row only when LS improves it.
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
                                    f"Unsupported loss_function: {self.loss_function}"
                                )

                            max_pomo_reward, _ = reward.max(dim=0)
                            score_mean = -max_pomo_reward.float().mean()
                        if args.ddp:
                            torch.distributed.barrier()

                        # Backward + optimizer step (always, for every batch)
                        if hasattr(self.model, "aux_loss"):
                            loss = loss + self.model.aux_loss
                        (self.scaler.scale(loss) if self.use_scaler else loss).backward()

                        # Clip gradients
                        grad_norms, grad_norms_clipped = clip_grad_norms(
                            self.optimizer.param_groups, 1.0
                        )

                        # Optimizer step
                        if self.use_scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        # Update instances count
                        instances_seen += batch_size
                        step += 1

                        # Log
                        metric_list = [
                            loss.item(),
                            score_mean.item(),
                        ]
                        all_metric.append(metric_list)
                        metric_info = " | ".join(
                            [
                                f"{TRAIN_METRIC_LABELS[i]} {metric_list[i]:.4f}"
                                for i in range(len(TRAIN_METRIC_LABELS))
                            ]
                        )
                        ls_info = f"LS:{ls_update_count}" if use_ls else ""
                        progress.update(
                            train_task,
                            description=f"{train_label} | {metric_info} | {ls_info}",
                            advance=batch_size,
                        )

                        # Explicitly release massive GPU tensors bounded locally to the iteration.
                        # Natively drops dangling VRAM usage scaling at the loop boundary.
                        del td, td_initial, out, reward, loss
                        if "log_likelihood" in locals():
                            del log_likelihood
                        if "tours" in locals():
                            del tours
                        if "advantage" in locals():
                            del advantage
                        if "improvements" in locals():
                            del improvements

                if args.ddp:
                    torch.distributed.barrier()

                # Log Once, for each epoch
                metric_tensor = torch.tensor(all_metric).mean(dim=0)
                metric_list = metric_tensor.tolist()
                metric_info = " | ".join(
                    [
                        f"{TRAIN_METRIC_LABELS[i]} {metric_list[i]:.4f}"
                        for i in range(len(TRAIN_METRIC_LABELS))
                    ]
                )
                elapsed = time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - start_time)
                )
                self.console.print(f"[green]Training complete in {elapsed}[/green]")
                ls_info = f"LS updates:{ls_update_count}" if use_ls else ""
                args.log(
                    f"{train_label} | {elapsed} | {metric_info} | LR {self.optimizer.param_groups[0]['lr']:.2e} | {ls_info}"
                )
                ## on all devices
                if args.ddp:
                    metric_tensor_ = metric_tensor.to("cuda")
                    dist.reduce(metric_tensor_, dst=0)
                    if args.rank == 0:
                        metric_avg = metric_tensor_ / dist.get_world_size()
                        metric_info = "|".join(
                            [
                                f"{TRAIN_METRIC_LABELS[i]} {metric_avg[i]:.4f}"
                                for i in range(len(TRAIN_METRIC_LABELS))
                            ]
                        )
                        args.log(
                            f"***ddp_reduce*** {train_label}|{elapsed}|{metric_info}"
                        )
                        if args.wandb != "" and args.rank == 0:
                            wandb.log(
                                {
                                    f"{TRAIN_METRIC_LABELS[i]}_train": metric_list[i]
                                    for i in range(len(TRAIN_METRIC_LABELS))
                                },
                                step=epoch,
                            )
                    torch.distributed.barrier()
                elif args.wandb != "":
                    wandb.log(
                        {
                            f"{TRAIN_METRIC_LABELS[i]}_train": metric_list[i]
                            for i in range(len(TRAIN_METRIC_LABELS))
                        },
                        step=epoch,
                    )
                # test during train
                if args.test and (
                    epoch % args.env["test_interval"] == 0
                    or epoch in args.env["test_epoch"]
                ):
                    self._test(epoch, self.test_dataloader)

                # End of epoch: LR decay and checkpointing
                if (
                    self.scheduler is not None
                    and args.optimizer_params["scheduler"]["name"] == "MultiStepLR"
                ):
                    self.scheduler.step()

                # Log time estimate
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                    epoch, args.trainer_params["epochs"]
                )
                args.log(
                    "Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                        epoch,
                        args.trainer_params["epochs"],
                        elapsed_time_str,
                        remain_time_str,
                    )
                )

                # Save checkpoint
                if (epoch == args.trainer_params["epochs"]) or (
                    epoch % args.trainer_params["model_save_interval"]
                ) == 0:
                    if not args.mute:
                        args.log("Saving trained_model")
                        checkpoint_dict = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict()
                            if not self.args.ddp
                            else self.model.module.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "env_state_dict": self.env.__getstate__(),
                            "rng_state_dict": {
                                "torch.rng_state": torch.get_rng_state(),
                                "torch.cuda.rng_state": torch.cuda.get_rng_state(),
                                "random.state": random.getstate(),
                            },
                        }
                        if self.scheduler is not None:
                            checkpoint_dict["scheduler_state_dict"] = (
                                self.scheduler.state_dict()
                            )
                        if self.use_amp:
                            checkpoint_dict["scaler_state_dict"] = (
                                self.scaler.state_dict()
                            )

                        torch.save(
                            checkpoint_dict,
                            "{}/checkpoint-{}.pt".format(args.result_dir, epoch),
                        )
                # end of epoch
                if args.ddp:
                    torch.distributed.barrier()

        finally:
            # Stop local search executor
            if ls_executor_started:
                self._stop_ls_executor()

        args.log(" *** Training Done *** ")


def _run_instance_search(batch_idx, instance_args, pomo_indices, tours, nb_granular=20):
    """Run local search on a single VRP instance."""
    (
        locs,
        demands_linehaul,
        demands_backhauls,
        distance_limit,
        open_route,
        time_windows,
        service_time,
    ) = instance_args
    search = Search(
        locs,
        demands_linehaul,
        demands_backhauls,
        distance_limit,
        open_route,
        time_windows,
        service_time,
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
