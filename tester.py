# -*- coding: utf-8 -*-
import os
import gc
import re
import time
import torch
import numpy as np
import pandas as pd
import vrplib
import concurrent.futures
import multiprocessing as mp
from math import ceil
from typing import Dict, List, Optional, Tuple
from tensordict import TensorDict
from einops import rearrange
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.console import Console
import torch.distributed as dist

from utils.functions import (
    gather_by_index,
    get_distance,
    load_npz_to_tensordict,
    batchify,
)
from utils.metrics import gap_percent_mean_torch, gap_percent_scalar
from search import Search
from search.vrplib_helpers import (
    VRPLIB_ROUND_FUNC_IDS,
    default_vrplib_round_func,
    vrplib_ils_time_limit,
)


def natural_sort_key(text: str) -> List:
    """Sort key for instance names with embedded numbers (e.g. X-n101-k25)."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def normalize_coord(coord: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Normalize coordinates to [0, 1] range."""
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    scale = max((x_max - x_min), (y_max - y_min))
    x_scaled = (x - x_min) / scale
    y_scaled = (y - y_min) / scale
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled, scale


def metric2str(metric_label: List[str], metric_list: List[float]) -> str:
    """Return a compact string representation for logging metric pairs."""
    return "|".join(
        [f"{metric_label[i]} {metric_list[i]:.3f}" for i in range(len(metric_label))]
    )


class VRPTester:
    """Handles testing and evaluation for VRP models."""

    def __init__(self, model, env, augmentation, args):
        self.model = model
        self.env = env
        self.augmentation = augmentation
        self.args = args
        self.console = Console()

        # AMP configuration
        self.use_amp = bool(args.trainer_params.get("use_amp", False))
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            self.supports_bf16 = capability[0] >= 8
            self.amp_dtype = torch.bfloat16 if self.supports_bf16 else torch.float16
        else:
            self.supports_bf16 = False
            self.amp_dtype = torch.float16

    @torch.inference_mode()
    def test(self, epoch: int, test_dataloader: dict) -> Dict[str, float]:
        """Run evaluation over all configured datasets and log aggregate metrics."""
        args = self.args
        self.model.eval()

        # Clear any cached state
        if hasattr(self.model, "encoded_nodes"):
            self.model.encoded_nodes = None

        # Force cleanup before starting
        gc.collect()
        torch.cuda.empty_cache()

        start_time = time.time()
        dataset_num = len(test_dataloader)

        s_a8gap_dict = {i: [] for i in args.env.get("test_size", [50, 100])}
        p_a8gap_dict = {i: [] for i in args.env.get("test_problem", [])}
        d_a8gap_dict = {i: [] for i in args.env.get("test_distribution", ["uniform"])}
        s_gap_dict = {i: [] for i in args.env.get("test_size", [50, 100])}
        p_gap_dict = {i: [] for i in args.env.get("test_problem", [])}
        d_gap_dict = {i: [] for i in args.env.get("test_distribution", ["uniform"])}

        # For Excel saving
        excel_data_gap = []  # List of (problem, gap) tuples
        excel_data_a8gap = []  # List of (problem, aug_gap) tuples

        tmp_test_metric_label = ["NO_AUG Obj.", "NO_AUG Gap", "AUG Obj.", "AUG Gap"]
        
        for data_idx, (dataset_name, dataloader) in enumerate(test_dataloader.items()):
            all_metric = []
            eval_label = f"Eval {dataset_name} {str(data_idx + 1).zfill(3)}/{str(dataset_num).zfill(3)} | Epoch{str(epoch).zfill(3)}"

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("-"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                eval_task = progress.add_task(eval_label, total=len(dataloader))

                for step, inp in enumerate(dataloader):
                    if args.skip and step > 1:
                        break

                    batch_size = inp.batch_size[0]

                    # Store p_s_tag and opt_cost before reset (they get lost in reset)
                    p_s_tag = inp["p_s_tag"].clone()
                    opt_cost = inp["opt_cost"].clone()

                    # Reset environment
                    td = self.env.reset(td=inp.to("cuda"))

                    # Restore p_s_tag after reset
                    td["p_s_tag"] = p_s_tag.to("cuda")

                    with torch.amp.autocast(
                        device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
                    ):
                        td_aug = self.augmentation(td)
                        if args.ddp:
                            torch.distributed.barrier()
                        
                        if args.tester_params.get("use_refinement", False):
                            reward = self.model.iterative_refinement(
                                td_aug,
                                self.env,
                                ls_nb_granular=args.tester_params.get("ls_nb_granular", 40),
                                num_iters=args.tester_params.get("num_iters", 5000),
                                stop_condition=args.tester_params.get(
                                    "stop_condition", "iterations"
                                ),
                                num_seconds=args.tester_params.get("num_seconds", 100.0),
                                dmax=args.tester_params.get("dmax", 30),
                                dmin=args.tester_params.get("dmin", 15),
                                gamma=args.tester_params.get("gamma", 30),
                                eta_min=args.tester_params.get("eta_min", 0.01),
                            )
                        else:
                            out = self.model(td_aug, self.env)
                            reward = out["reward"]
                            del out

                        all_reward = reward.view(
                            -1, self.augmentation.num_augment, batch_size
                        )
                        all_reward, _ = all_reward.max(dim=0)
                        score = -all_reward[0, :].float()
                        aug_reward, _ = all_reward.max(dim=0)
                        aug_score = -aug_reward.float()

                    # Delete large tensors immediately
                    del td, td_aug, reward
                    if hasattr(self.model, "encoded_nodes"):
                        self.model.encoded_nodes = None

                    # Compute gap using stored opt_cost
                    opt_score = opt_cost.to("cuda")
                    gap = gap_percent_mean_torch(score.abs(), opt_score.abs())
                    aug_gap = gap_percent_mean_torch(aug_score.abs(), opt_score.abs())
                    metric_list = [
                        score.mean().item(),
                        gap,
                        aug_score.mean().item(),
                        aug_gap,
                    ]

                    # Delete remaining tensors
                    del (
                        score,
                        aug_score,
                        all_reward,
                        aug_reward,
                        opt_score,
                        inp,
                        p_s_tag,
                        opt_cost,
                    )

                    # Force memory cleanup every batch
                    torch.cuda.empty_cache()

                    all_metric.append(metric_list)
                    metric_info = metric2str(tmp_test_metric_label, metric_list)
                    progress.update(
                        eval_task, advance=1, description=f"{eval_label}|{metric_info}"
                    )

            # Force cleanup after each dataset
            gc.collect()
            torch.cuda.empty_cache()

            # Log dataset results
            if args.ddp:
                torch.distributed.barrier()

            metric_mean = torch.tensor(all_metric).mean(dim=0).tolist()
            metric_info = metric2str(tmp_test_metric_label, metric_mean)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            args.log(f"{eval_label} | {elapsed} | {metric_info}")

            # Parse dataset name
            parts = dataset_name.split("_")
            size = int(parts[0])
            problem = parts[1] if len(parts) > 1 else "cvrp"
            distribution = parts[2] if len(parts) > 2 else "uniform"

            # Store for Excel
            excel_data_gap.append({"problem": dataset_name, "gap": metric_mean[1]})
            excel_data_a8gap.append({"problem": dataset_name, "gap": metric_mean[3]})

            # Update metric dictionaries
            if size in s_gap_dict:
                s_a8gap_dict[size].append(metric_mean[3])
                s_gap_dict[size].append(metric_mean[1])
            if problem in p_gap_dict:
                p_a8gap_dict[problem].append(metric_mean[3])
                p_gap_dict[problem].append(metric_mean[1])
            if distribution in d_gap_dict:
                d_a8gap_dict[distribution].append(metric_mean[3])
                d_gap_dict[distribution].append(metric_mean[1])

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Save Excel files
        if hasattr(args, "result_dir") and args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)

            # Save gap_{epoch}.xlsx
            df_gap = pd.DataFrame(excel_data_gap)
            df_gap.to_excel(
                f"{args.result_dir}/gap_{epoch}.xlsx", index=False, engine="openpyxl"
            )

            # Save a8gap_{epoch}.xlsx
            df_a8gap = pd.DataFrame(excel_data_a8gap)
            df_a8gap.to_excel(
                f"{args.result_dir}/a8gap_{epoch}.xlsx", index=False, engine="openpyxl"
            )

            args.log(
                f"Saved test results to {args.result_dir}/gap_{epoch}.xlsx and a8gap_{epoch}.xlsx"
            )

        # Compute aggregate metrics
        results = {}
        all_gaps = []
        all_aug_gaps = []

        for name, d in [
            ("size_aug_gap", s_a8gap_dict),
            ("size_gap", s_gap_dict),
            ("problem_aug_gap", p_a8gap_dict),
            ("problem_gap", p_gap_dict),
            ("dist_aug_gap", d_a8gap_dict),
            ("dist_gap", d_gap_dict),
        ]:
            for k, v in d.items():
                if v:
                    results[f"{name}/{k}"] = np.mean(v)
                    if "aug_gap" in name:
                        all_aug_gaps.extend(v)
                    elif "gap" in name and "aug" not in name:
                        all_gaps.extend(v)

        # Print overall average gaps
        if all_gaps:
            avg_gap = np.mean(all_gaps)
            args.log(f">>> Overall Average NO_AUG Gap: {avg_gap:.3f}%")
            results["overall_gap"] = avg_gap
        if all_aug_gaps:
            avg_aug_gap = np.mean(all_aug_gaps)
            args.log(f">>> Overall Average AUG Gap: {avg_aug_gap:.3f}%")
            results["overall_aug_gap"] = avg_aug_gap

        return results

    @torch.inference_mode()
    def test_tuning_variants(
        self,
        epoch: int,
        data_dir: str = "./data",
        variant_present: str = "mb",
        sizes: List[int] = [50, 100],
        solution_source: str = "pyvrp",
    ) -> Dict[str, float]:
        """Test on tuning variant datasets (MB, MD, both)."""
        args = self.args
        self.model.eval()

        # Clear any cached state
        if hasattr(self.model, "encoded_nodes"):
            self.model.encoded_nodes = None

        # Force cleanup before starting
        gc.collect()
        torch.cuda.empty_cache()

        # Get variant folders based on variant_present
        variant_folders = self._get_variant_folders(data_dir, variant_present)

        results = {}
        start_time = time.time()
        tmp_test_metric_label = ["NO_AUG Obj.", "NO_AUG Gap", "AUG Obj.", "AUG Gap"]

        # For Excel saving
        excel_data_gap = []
        excel_data_a8gap = []

        dataset_num = len(variant_folders) * len(sizes)
        dataset_idx = 0

        for variant_folder in variant_folders:
            variant_name = os.path.basename(variant_folder)

            for size in sizes:
                dataset_idx += 1
                test_file = os.path.join(variant_folder, "test", f"{size}.npz")
                sol_file = os.path.join(
                    variant_folder, "test", f"{size}_sol_{solution_source}.npz"
                )

                if not os.path.exists(test_file):
                    continue

                eval_label = f"Eval {variant_name}_{size} {str(dataset_idx).zfill(3)}/{str(dataset_num).zfill(3)} | Epoch{str(epoch).zfill(3)}"

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold green]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("-"),
                    TimeElapsedColumn(),
                    console=self.console,
                    transient=True,
                ) as progress:
                    # Load data
                    td = load_npz_to_tensordict(test_file).to("cuda")

                    # Load solutions if available
                    opt_costs = None
                    if os.path.exists(sol_file):
                        sol_data = np.load(sol_file)
                        if "costs" in sol_data:
                            opt_costs = torch.tensor(sol_data["costs"], device="cuda")

                    # Build p_s_tag
                    td = self._build_p_s_tag(td, variant_name, size)
                    td = self._add_missing_fields(td)

                    # Run evaluation
                    batch_size = td.batch_size[0]
                    eval_batch_size = min(batch_size, int(args.env["test_batch_size"]))
                    total_eval_batches = (batch_size + eval_batch_size - 1) // eval_batch_size
                    eval_task = progress.add_task(eval_label, total=total_eval_batches)

                    all_scores = []
                    all_aug_scores = []

                    for start_idx in range(0, batch_size, eval_batch_size):
                        end_idx = min(start_idx + eval_batch_size, batch_size)
                        batch_td = td[start_idx:end_idx]
                        batch_opt = (
                            opt_costs[start_idx:end_idx].abs()
                            if opt_costs is not None
                            else None
                        )

                        batch_td = self.env.reset(td=batch_td)

                        with torch.amp.autocast(
                            device_type="cuda",
                            dtype=self.amp_dtype,
                            enabled=self.use_amp,
                        ):
                            batch_td = self.augmentation(batch_td)
                            if args.tester_params.get("use_refinement", False):
                                reward = self.model.iterative_refinement(
                                    batch_td,
                                    self.env,
                                    ls_nb_granular=args.tester_params.get(
                                        "ls_nb_granular", 40
                                    ),
                                    num_iters=args.tester_params.get("num_iters", 5000),
                                    stop_condition=args.tester_params.get(
                                        "stop_condition", "iterations"
                                    ),
                                    num_seconds=args.tester_params.get(
                                        "num_seconds", 100.0
                                    ),
                                    dmax=args.tester_params.get("dmax", 30),
                                    dmin=args.tester_params.get("dmin", 15),
                                    gamma=args.tester_params.get("gamma", 30),
                                    eta_min=args.tester_params.get("eta_min", 0.01),
                                )
                            else:
                                out = self.model(batch_td, self.env)
                                reward = out["reward"]

                        batch_reward = reward.view(
                            -1, self.augmentation.num_augment, end_idx - start_idx
                        )
                        batch_reward, _ = batch_reward.max(dim=0)

                        batch_score = -batch_reward[0, :].float()
                        all_scores.append(batch_score)
                        aug_reward, _ = batch_reward.max(dim=0)
                        batch_aug_score = -aug_reward.float()
                        all_aug_scores.append(batch_aug_score)

                        if batch_opt is not None:
                            b_score_abs = batch_score.abs()
                            b_aug_abs = batch_aug_score.abs()
                            batch_gap = gap_percent_mean_torch(b_score_abs, batch_opt)
                            batch_aug_gap = gap_percent_mean_torch(b_aug_abs, batch_opt)
                            batch_metric = [
                                b_score_abs.mean().item(),
                                batch_gap,
                                b_aug_abs.mean().item(),
                                batch_aug_gap,
                            ]
                        else:
                            batch_metric = [
                                batch_score.mean().item(),
                                0.0,
                                batch_aug_score.mean().item(),
                                0.0,
                            ]
                        batch_metric_info = metric2str(tmp_test_metric_label, batch_metric)

                        progress.update(
                            eval_task,
                            advance=1,
                            description=f"{eval_label}|{batch_metric_info}",
                        )
                        torch.cuda.empty_cache()

                    scores = torch.cat(all_scores)
                    aug_scores = torch.cat(all_aug_scores)

                    # Compute gaps
                    if opt_costs is not None:
                        opt_costs = opt_costs.abs()
                        scores = scores.abs()
                        aug_scores = aug_scores.abs()

                        gap = gap_percent_mean_torch(scores, opt_costs)
                        aug_gap = gap_percent_mean_torch(aug_scores, opt_costs)
                    else:
                        gap = aug_gap = 0.0

                    score_mean = scores.mean().item()
                    aug_score_mean = aug_scores.mean().item()
                    metric_list = [score_mean, gap, aug_score_mean, aug_gap]

                    key = f"{variant_name}_{size}"
                    results[key] = {
                        "score": score_mean,
                        "aug_score": aug_score_mean,
                        "gap": gap,
                        "aug_gap": aug_gap,
                    }

                    # Store for Excel
                    excel_data_gap.append({"problem": key, "gap": gap})
                    excel_data_a8gap.append({"problem": key, "gap": aug_gap})

                    metric_info = metric2str(tmp_test_metric_label, metric_list)
                    progress.update(eval_task, description=f"{eval_label}|{metric_info}")

                    # Force cleanup
                    gc.collect()
                    torch.cuda.empty_cache()

                # Log dataset results
                if args.ddp:
                    torch.distributed.barrier()

                metric_info = metric2str(tmp_test_metric_label, metric_list)
                elapsed = time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - start_time)
                )
                args.log(f"{eval_label} | {elapsed} | {metric_info}")

                del td, scores, aug_scores
                gc.collect()
                torch.cuda.empty_cache()

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Save Excel files
        if hasattr(args, "result_dir") and args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)

            # Save gap_{epoch}.xlsx
            df_gap = pd.DataFrame(excel_data_gap)
            df_gap.to_excel(
                f"{args.result_dir}/tuning_gap_{variant_present}_{epoch}.xlsx",
                index=False,
                engine="openpyxl",
            )

            # Save a8gap_{epoch}.xlsx
            df_a8gap = pd.DataFrame(excel_data_a8gap)
            df_a8gap.to_excel(
                f"{args.result_dir}/tuning_a8gap_{variant_present}_{epoch}.xlsx",
                index=False,
                engine="openpyxl",
            )

            args.log(
                f"Saved tuning results to {args.result_dir}/tuning_gap_{variant_present}_{epoch}.xlsx and tuning_a8gap_{variant_present}_{epoch}.xlsx"
            )

        # Print overall average gaps
        all_gaps = [r["gap"] for r in results.values()]
        all_aug_gaps = [r["aug_gap"] for r in results.values()]

        if all_gaps:
            avg_gap = np.mean(all_gaps)
            args.log(f">>> Overall Average NO_AUG Gap: {avg_gap:.3f}%")
        if all_aug_gaps:
            avg_aug_gap = np.mean(all_aug_gaps)
            args.log(f">>> Overall Average AUG Gap: {avg_aug_gap:.3f}%")

        return results

    def _get_variant_folders(self, data_dir: str, variant_present: str) -> List[str]:
        """Get list of variant folders based on variant_present setting."""
        all_dirs = [
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]

        if variant_present == "mb":
            # Mixed backhaul single-depot variants
            return [
                os.path.join(data_dir, d)
                for d in all_dirs
                if "mb" in d.lower() and not d.lower().startswith("md")
            ]
        elif variant_present == "md":
            # Multi-depot variants (without mixed backhaul)
            return [
                os.path.join(data_dir, d)
                for d in all_dirs
                if d.lower().startswith("md") and "mb" not in d.lower()
            ]
        elif variant_present == "both":
            # Multi-depot with mixed backhaul
            return [
                os.path.join(data_dir, d)
                for d in all_dirs
                if d.lower().startswith("md") and "mb" in d.lower()
            ]
        else:
            raise ValueError(f"Unknown variant_present: {variant_present}")

    def _build_p_s_tag(
        self, td: TensorDict, variant_name: str, size: int
    ) -> TensorDict:
        """Build p_s_tag tensor for the given variant."""
        batch_size = td.batch_size[0]
        variant_lower = variant_name.lower()

        # Parse variant flags
        has_open = "ovrp" in variant_lower or (
            variant_lower.startswith("o") and "vrp" in variant_lower
        )
        has_tw = "tw" in variant_lower
        has_limit = "l" in variant_lower and (
            "ltw" in variant_lower
            or "bl" in variant_lower
            or variant_lower.endswith("l")
        )
        has_backhaul = "b" in variant_lower
        has_mixed_backhaul = "mb" in variant_lower
        has_multi_depot = variant_lower.startswith("md")

        # p_s_tag: [C, O, TW, L, B, MB, MD, size]
        p_s_tag = torch.zeros((batch_size, 8), dtype=torch.float32, device=td.device)
        p_s_tag[:, 0] = float(not has_open)
        p_s_tag[:, 1] = float(has_open)
        p_s_tag[:, 2] = float(has_tw)
        p_s_tag[:, 3] = float(has_limit)
        p_s_tag[:, 4] = float(has_backhaul)
        p_s_tag[:, 5] = float(has_mixed_backhaul)
        p_s_tag[:, 6] = float(has_multi_depot)
        p_s_tag[:, 7] = size / 2000.0

        td["p_s_tag"] = p_s_tag
        return td

    def _add_missing_fields(self, td: TensorDict) -> TensorDict:
        """Add missing fields with generator-consistent defaults."""
        batch = td.batch_size[0]
        n_locs = td["locs"].shape[1]
        num_depots = td["num_depots"][0].item() if "num_depots" in td.keys() else 1
        device = td["locs"].device
        
        if num_depots > 1:
            if "demand_backhaul" not in td.keys():
                td["demand_backhaul"] = torch.zeros(batch, n_locs - num_depots, device=device)
        else:
            if "demand_backhaul" not in td.keys():
                td["demand_backhaul"] = torch.zeros(batch, n_locs, device=device)
            
            if td["demand_linehaul"].shape[1] != n_locs:
                td["demand_linehaul"] = torch.cat([torch.zeros(batch,  n_locs - td["demand_linehaul"].shape[1], device=device), td["demand_linehaul"]], dim=1)
            if td["demand_backhaul"].shape[1] != n_locs:
                td["demand_backhaul"] = torch.cat([torch.zeros(batch,  n_locs - td["demand_backhaul"].shape[1], device=device), td["demand_backhaul"]], dim=1)
        
        if "distance_limit" not in td.keys():
            td["distance_limit"] = torch.full((batch, 1), float("inf"), device=device)
        
        if "service_time" not in td.keys():
            td["service_time"] = torch.zeros(batch, n_locs, device=device)
        elif td["service_time"].shape[1] == n_locs - num_depots:
            td["service_time"] = torch.cat(
                [torch.zeros(batch, num_depots, device=device), td["service_time"]], dim=1
            )
        
        if "time_windows" not in td.keys():
            tw = torch.zeros(batch, n_locs, 2, device=device)
            tw[..., 1] = float("inf")
            td["time_windows"] = tw
        
        if "capacity_original" not in td.keys():
            td["capacity_original"] = torch.ones(batch, 1, device=device)
        
        if "open_route" not in td.keys():
            td["open_route"] = torch.zeros(batch, 1, dtype=torch.bool, device=device)

        return td

    @torch.inference_mode()
    def test_lib(self, epoch: int) -> Dict[str, float]:
        """Run evaluation on CVRPLIB benchmark instances.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of dataset -> gap metrics
        """
        args = self.args
        self.model.eval()

        # all_test_dataset = ["A", "B", "F", "P", "X"]
        all_test_dataset = ["X"]
        size_limit = 1001

        all_dataset_dict = {
            "A": [],
            "B": [],
            "F": [],
            "M": [],
            "P": [],
            "X-100-300": [],
            "X-300-500": [],
            "X-500-700": [],
            "X-700-1000": [],
        }

        for dataset in all_test_dataset:
            dataset_dir = f"./data/lib_data/{dataset}"
            sol_dir = f"./data/lib_data/{dataset}"

            if not os.path.exists(dataset_dir):
                continue

            path_list = [
                os.path.join(dataset_dir, x)
                for x in os.listdir(dataset_dir)
                if not (
                    os.path.isdir(os.path.join(dataset_dir, x))
                    or x.endswith(".sol")
                    or x.startswith(".")
                )
            ]
            path_list.sort(
                key=lambda p: natural_sort_key(
                    os.path.splitext(os.path.basename(p))[0]
                )
            )

            all_gap, all_aug_gap, all_aug_score = [], [], []
            gap_dict, aug_gap_dict, aug_score_dict = {}, {}, {}
            time_dict, opt_score_dict = {}, {}

            if dataset == "X":
                X_opt_score_list = [[], [], [], []]
                X_aug_score_list = [[], [], [], []]
                X_aug_gap_list = [[], [], [], []]

            for path in path_list:
                if (
                    os.path.isdir(path)
                    or path.endswith(".sol")
                    or os.path.basename(path).startswith(".")
                ):
                    continue

                # Load problem
                problem = vrplib.read_instance(path)
                coords = torch.tensor(problem["node_coord"]).float()
                coords_norm, scale = normalize_coord(coords)
                original_capacity = problem["capacity"]
                demand = torch.tensor(problem["demand"][1:]).float() / original_capacity
                original_capacity = torch.tensor(original_capacity)[None]
                lib_round_func = default_vrplib_round_func(dataset)

                td_instance = TensorDict(
                    {
                        "locs": coords_norm.unsqueeze(0),
                        "demand_linehaul": demand.unsqueeze(0),
                        "capacity_original": original_capacity.unsqueeze(0),
                        # CVRPLIB-aligned LS (PyVRP round_func); neural still uses normalized locs.
                        "vrplib_coords": coords.unsqueeze(0),
                        "vrplib_demands": torch.tensor(problem["demand"], dtype=torch.float32).unsqueeze(0),
                        "vrplib_capacity": original_capacity.unsqueeze(0),
                        "vrplib_round_func_id": torch.tensor(
                            [VRPLIB_ROUND_FUNC_IDS[lib_round_func]], dtype=torch.long
                        ),
                        "vrplib_edge_weight": torch.tensor(
                            problem["edge_weight"], dtype=torch.float32
                        ).unsqueeze(0),
                    },
                    batch_size=[1],
                )

                td_reset = self.env.reset(td_instance, lib_data=True).to("cuda")

                # Build p_s_tag for CVRP
                keep_mask = torch.zeros((td_reset.shape[0], 5), dtype=torch.bool)
                keep_mask[:, 0] = True  # C (closed)
                td_reset["p_s_tag"] = torch.cat(
                    [
                        keep_mask.float(),
                        torch.zeros(
                            (td_reset.shape[0], 2), dtype=torch.float32, device="cuda"
                        ),  # MB, MD
                        torch.full(
                            (td_reset.shape[0], 1),
                            td_reset["locs"].shape[1] / 2000,
                            dtype=torch.float32,
                            device="cuda",
                        ),
                    ],
                    dim=-1,
                )

                if size_limit is not None and td_reset["locs"].shape[1] > size_limit:
                    continue

                # Get optimal cost
                instance_name = os.path.basename(path).split(".")[0]
                sol_path = os.path.join(sol_dir, f"{instance_name}.sol")
                solution = vrplib.read_solution(sol_path)
                opt = solution["cost"]

                # Solve
                start_time = time.time()
                with torch.amp.autocast(
                    device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    td = self.augmentation(td_reset)
                    if args.ddp:
                        torch.distributed.barrier()

                    if args.tester_params.get("use_refinement", False):
                        num_nodes = len(problem["node_coord"])
                        lib_num_seconds = vrplib_ils_time_limit(
                            num_nodes, args.tester_params.get("num_seconds")
                        )
                        reward = self.model.iterative_refinement(
                            td,
                            self.env,
                            ls_nb_granular=args.tester_params.get("ls_nb_granular", 40),
                            num_iters=args.tester_params.get("num_iters", 5000),
                            stop_condition=args.tester_params.get(
                                "stop_condition", "iterations"
                            ),
                            num_seconds=lib_num_seconds,
                            dmax=args.tester_params.get("dmax", 30),
                            dmin=args.tester_params.get("dmin", 15),
                            gamma=args.tester_params.get("gamma", 30),
                            eta_min=args.tester_params.get("eta_min", 0.01),
                        )
                    else:
                        out = self.model(td, self.env)
                        reward = out["reward"]
                        del out

                use_time = time.time() - start_time

                all_reward = reward.view(-1, self.augmentation.num_augment, 1)
                all_reward = rearrange(all_reward, "r a b -> (r b) a").unsqueeze(-1)
                all_reward, _ = all_reward.max(dim=0)

                score = -all_reward[0, :].float().item()
                aug_reward, _ = all_reward.max(dim=0)
                aug_score = -aug_reward.float().item()
                
                use_refinement = args.tester_params.get("use_refinement", False)
                if use_refinement and "vrplib_coords" in td_reset.keys():
                    # NAILS refinement reports costs in CVRPLIB integer units.
                    score, aug_score = ceil(score), ceil(aug_score)
                else:
                    # Neural decode uses normalized coordinates; rescale to
                    # original coordinate units before comparing to BKS.
                    score, aug_score = ceil(score * scale), ceil(aug_score * scale)
                gap = gap_percent_scalar(score, opt)
                aug_gap = gap_percent_scalar(aug_score, opt)

                args.log(
                    f"{instance_name}, aug score {aug_score:.1f}, aug gap {aug_gap:.3f}%"
                )

                all_gap.append(gap)
                all_aug_gap.append(aug_gap)
                all_aug_score.append(aug_score)

                gap_dict[instance_name] = gap
                aug_gap_dict[instance_name] = aug_gap
                aug_score_dict[instance_name] = aug_score
                time_dict[instance_name] = use_time
                opt_score_dict[instance_name] = opt

                # X dataset size categorization
                if dataset == "X":
                    num = int(instance_name.split("-")[1][1:])
                    idx_ = (
                        0
                        if num <= 300
                        else (1 if num <= 500 else (2 if num <= 700 else 3))
                    )
                    X_opt_score_list[idx_].append(opt)
                    X_aug_score_list[idx_].append(aug_score)
                    X_aug_gap_list[idx_].append(aug_gap)

            if len(all_aug_gap) > 0:
                avg_gap = sum(all_aug_gap) / len(all_aug_gap)
                avg_obj = sum(all_aug_score) / len(all_aug_score)
                args.log(
                    f"\nDataset {dataset}: Avg aug Obj {avg_obj:.1f}, Avg aug Gap {avg_gap:.3f}%\n"
                )

            # Save results
            if hasattr(args, "result_dir"):
                data = {
                    "Instance Name": list(gap_dict.keys()),
                    "Opt Score": [opt_score_dict[name] for name in gap_dict.keys()],
                    "Aug Score": [aug_score_dict[name] for name in gap_dict.keys()],
                    "Aug Gap": [aug_gap_dict[name] for name in gap_dict.keys()],
                    "Use Time": [time_dict[name] for name in gap_dict.keys()],
                }
                df = pd.DataFrame(data)
                df = df.sort_values("Instance Name")
                df.to_excel(
                    f"{args.result_dir}/{dataset}.xlsx", index=False, engine="openpyxl"
                )

            # Update aggregate dict
            if dataset != "X" and len(opt_score_dict) > 0:
                all_dataset_dict[dataset] = [
                    np.mean(list(opt_score_dict.values())),
                    np.mean(list(aug_score_dict.values())),
                    np.mean(list(aug_gap_dict.values())),
                ]
            elif dataset == "X":
                for idx_, num_ in enumerate(
                    ["100-300", "300-500", "500-700", "700-1000"]
                ):
                    if X_opt_score_list[idx_]:
                        all_dataset_dict[f"X-{num_}"] = [
                            np.mean(X_opt_score_list[idx_]),
                            np.mean(X_aug_score_list[idx_]),
                            np.mean(X_aug_gap_list[idx_]),
                        ]

        # Save aggregate results
        if hasattr(args, "result_dir"):
            all_dataset_dict_ = {k: v for k, v in all_dataset_dict.items() if v}
            if all_dataset_dict_:
                df = pd.DataFrame(all_dataset_dict_)
                df.to_excel(
                    f"{args.result_dir}/cvrplib.xlsx", index=False, engine="openpyxl"
                )

        # Print overall average gap across all CVRPLIB datasets
        all_aug_gaps = []
        all_aug_scores = []
        all_opt_scores = []
        for dataset_name, values in all_dataset_dict.items():
            if values and len(values) >= 3:
                all_opt_scores.append(values[0])
                all_aug_scores.append(values[1])
                all_aug_gaps.append(values[2])

        if all_aug_gaps:
            avg_aug_gap = np.mean(all_aug_gaps)
            avg_aug_score = np.mean(all_aug_scores)
            args.log(f">>> CVRPLIB Overall Average AUG Score: {avg_aug_score:.1f}")
            args.log(f">>> CVRPLIB Overall Average AUG Gap: {avg_aug_gap:.3f}%")

        return all_dataset_dict
