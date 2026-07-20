"""
Single-Depot Multi-Task VRP Environment

Supports all 24 single-depot VRP variants:
- Original 16: CVRP, OVRP, VRPB, VRPL, VRPTW + combinations (backhaul_class=1)
- New 8 mixed backhaul: VRPMB, OVRPMB, etc. (backhaul_class=2)

The environment handles:
- Capacity constraints (C)
- Open routes (O)
- Time windows (TW)
- Distance limits (L)
- Backhaul with classical ordering (B, backhaul_class=1)
- Mixed backhaul with arbitrary ordering (MB, backhaul_class=2)
"""

import os
from os.path import join as pjoin
import torch
import numpy as np
from typing import Optional, List, Union
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.envs import EnvBase
from torch.utils.data import DataLoader

from utils.functions import gather_by_index, get_distance, load_npz_to_tensordict, get_torch_device
from envs.mtvrp.generator import MTVRPGenerator

EPS = 1e-6

def get_dataloader(dataset, batch_size, ddp=False, num_workers=0):
    """Create dataloader(s) from dataset."""

    def get_single_dataloader(dataset_, batch_size_, ddp_=False, num_workers_=0):
        def return_x(x):
            return x

        sampler_ = (
            torch.utils.data.distributed.DistributedSampler(dataset_, shuffle=False)
            if ddp_
            else None
        )
        return DataLoader(
            dataset_,
            batch_size=batch_size_,
            sampler=sampler_,
            shuffle=False,
            num_workers=num_workers_,
            collate_fn=return_x,
        )

    if isinstance(dataset, dict):
        assert isinstance(batch_size, int)
        batch_size = [batch_size for _ in list(dataset.keys())]
        return {
            name: get_single_dataloader(dset, bsize, ddp, num_workers)
            for (name, dset), bsize in zip(dataset.items(), batch_size)
        }
    else:
        assert isinstance(batch_size, int)
        return get_single_dataloader(dataset, batch_size, ddp, num_workers)


class MTVRPEnv(EnvBase):
    """Single-Depot Multi-Task VRP Environment.

    Features:
    - Capacity (C): Vehicle has maximum capacity for linehaul and backhaul
    - Time Windows (TW): Nodes have time windows [early, late] and service times
    - Open Routes (O): Vehicles don't need to return to depot
    - Backhaul (B): Some customers require pickup (backhaul_class=1 means classical ordering)
    - Mixed Backhaul (MB): Backhaul with arbitrary ordering (backhaul_class=2)
    - Distance Limits (L): Maximum route length
    """

    name = "mtvrp"

    def __init__(
        self,
        generator_params: dict = {},
        data_dir: str = "data/",
        test_size: list = None,
        test_problem: list = None,
        test_distribution: list = None,
        check_solution: bool = False,
        seed: int = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device)
        self.data_dir = data_dir
        self.val_file = self.val_dataloader_names = None
        self.test_size = test_size
        self.test_problem = test_problem
        self.test_distribution = test_distribution
        self.test_file = []
        self.test_dataloader_names = []
        self.loss_mode = "rl"
        self.check_solution = check_solution

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        self.generator = MTVRPGenerator(**generator_params)
        self._make_spec()

    def set_loss_mode(self, mode: str):
        self.loss_mode = mode

    def step(self, td: TensorDict) -> TensorDict:
        td = self._step(td)
        return {"next": td}

    def _step(self, td: TensorDict) -> TensorDict:
        """Execute one step in the environment."""
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc = gather_by_index(td["locs"], prev_node)
        curr_loc = gather_by_index(td["locs"], curr_node)
        distance = get_distance(prev_loc, curr_loc)[..., None]

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        curr_time = (curr_node[:, None] != 0) * (
            torch.max(td["current_time"] + distance / td["speed"], start_times)
            + service_time
        )

        # Update current route length (reset at depot)
        curr_route_length = (curr_node[:, None] != 0) * (
            td["current_route_length"] + distance
        )

        # Update capacities
        selected_demand_linehaul = gather_by_index(
            td["demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["demand_backhaul"], curr_node, dim=1, squeeze=False
        )
        used_capacity_linehaul = (curr_node[:, None] != 0) * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = (curr_node[:, None] != 0) * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        # Update visited
        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done).float()

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def reset(
        self, td: Optional[TensorDict] = None, batch_size=None, lib_data=False
    ) -> TensorDict:
        """Reset the environment."""
        if batch_size is None:
            batch_size = td.batch_size
        if td is None or td.is_empty():
            td = self.generator(batch_size=batch_size).to(get_torch_device())
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.to(td.device)
        return super().reset(td, batch_size=batch_size, lib_data=lib_data)

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
        lib_data=False,
    ) -> TensorDict:
        """Internal reset implementation."""
        device = td.device

        if not lib_data:
            td_reset = TensorDict(
                {
                    "locs": td["locs"],
                    "demand_backhaul": td["demand_backhaul"],
                    "demand_linehaul": td["demand_linehaul"],
                    "backhaul_class": td.get(
                        "backhaul_class",
                        torch.ones((*batch_size, 1), dtype=torch.float32),
                    ),
                    "distance_limit": td["distance_limit"],
                    "service_time": td["service_time"],
                    "open_route": td["open_route"],
                    "time_windows": td["time_windows"],
                    "vehicle_capacity": td["vehicle_capacity"],
                    "capacity_original": td["capacity_original"],
                    "speed": td["speed"],
                    "current_node": torch.zeros(
                        (*batch_size,), dtype=torch.long, device=device
                    ),
                    "current_route_length": torch.zeros(
                        (*batch_size, 1), dtype=torch.float32, device=device
                    ),
                    "current_time": torch.zeros(
                        (*batch_size, 1), dtype=torch.float32, device=device
                    ),
                    "used_capacity_backhaul": torch.zeros(
                        (*batch_size, 1), device=device
                    ),
                    "used_capacity_linehaul": torch.zeros(
                        (*batch_size, 1), device=device
                    ),
                    "visited": torch.zeros(
                        (*batch_size, td["locs"].shape[-2]),
                        dtype=torch.bool,
                        device=device,
                    ),
                },
                batch_size=batch_size,
                device=device,
            )
        else:
            # Handle lib_data loading format
            demand_linehaul = torch.cat(
                [
                    torch.zeros_like(td["demand_linehaul"][..., :1]),
                    td["demand_linehaul"],
                ],
                dim=1,
            )
            demand_backhaul = td.get(
                "demand_backhaul", torch.zeros_like(td["demand_linehaul"])
            )
            demand_backhaul = torch.cat(
                [torch.zeros_like(td["demand_linehaul"][..., :1]), demand_backhaul],
                dim=1,
            )

            backhaul_class = td.get(
                "backhaul_class", torch.full((*batch_size, 1), 1, dtype=torch.int32)
            )

            time_windows = td.get("time_windows", None)
            if time_windows is None:
                time_windows = torch.zeros_like(td["locs"])
                time_windows[..., 1] = float("inf")
            service_time = td.get("service_time", torch.zeros_like(demand_linehaul))

            open_route = td.get(
                "open_route",
                torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool),
            )
            distance_limit = td.get(
                "distance_limit",
                torch.full_like(demand_linehaul[..., :1], float("inf")),
            )

            reset_fields = {
                "locs": td["locs"],
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "service_time": service_time,
                "open_route": open_route,
                "time_windows": time_windows,
                "speed": td.get("speed", torch.ones_like(demand_linehaul[..., :1])),
                "vehicle_capacity": td.get(
                    "vehicle_capacity", torch.ones_like(demand_linehaul[..., :1])
                ),
                "capacity_original": td.get(
                    "capacity_original", torch.ones_like(demand_linehaul[..., :1])
                ),
                "current_node": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_route_length": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "current_time": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "used_capacity_backhaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),
                "used_capacity_linehaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ),
            }
            for key in (
                "vrplib_coords",
                "vrplib_demands",
                "vrplib_capacity",
                "vrplib_round_func_id",
                "vrplib_edge_weight",
            ):
                if key in td.keys():
                    reset_fields[key] = td.get(key)
            td_reset = TensorDict(
                reset_fields,
                batch_size=batch_size,
                device=device,
            )

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Compute action mask considering all constraints including mixed backhaul."""
        curr_node = td["current_node"]
        locs = td["locs"]
        d_ij = get_distance(gather_by_index(locs, curr_node)[..., None, :], locs)
        d_j0 = get_distance(locs, locs[..., 0:1, :])

        # Time constraint (TW)
        early_tw, late_tw = td["time_windows"][..., 0], td["time_windows"][..., 1]
        arrival_time = td["current_time"] + (d_ij / td["speed"])
        can_reach_customer = arrival_time < late_tw + EPS
        can_reach_depot = (
            torch.max(arrival_time, early_tw)
            + td["service_time"]
            + (d_j0 / td["speed"])
        ) * ~td["open_route"] < late_tw[..., 0:1] + EPS

        # Distance limit (L)
        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"] + EPS
        )

        # Capacity constraints
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"]
            > td["vehicle_capacity"] + EPS
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"]
            > td["vehicle_capacity"] + EPS
        )

        # Get backhaul class
        backhaul_class = td.get(
            "backhaul_class", torch.ones_like(td["vehicle_capacity"])
        )

        # Backhaul class 1 (classical): linehauls before backhauls
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[
            ..., None
        ]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["demand_backhaul"], idx=curr_node, dim=1, squeeze=False
            )
            > 0
        )

        meets_demand_constraint_backhaul_1 = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))

        # Backhaul class 2 (mixed): arbitrary ordering
        cannot_serve_linehaul = (
            td["demand_linehaul"]
            > td["vehicle_capacity"] - td["used_capacity_backhaul"] + EPS
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )

        # Select constraint based on backhaul class
        is_class_1 = (
            (backhaul_class == 1).squeeze(-1)
            if backhaul_class.dim() > 1
            else backhaul_class == 1
        )
        if is_class_1.dim() == 0:
            is_class_1 = is_class_1.unsqueeze(0)

        meets_demand_constraint = torch.where(
            is_class_1.unsqueeze(-1).expand_as(meets_demand_constraint_backhaul_1),
            meets_demand_constraint_backhaul_1,
            meets_demand_constraint_backhaul_2,
        )

        # Combine all constraints
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

        # Mask depot
        can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))

        return can_visit

    def get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Compute reward (negative tour length)."""
        if self.check_solution:
            self.check_solution_validity(td, actions)

        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)
        loc_from = gather_by_index(td["locs"], go_from)
        loc_to = gather_by_index(td["locs"], go_to)

        distances = get_distance(loc_from, loc_to)
        tour_length = (distances * ~((go_to == 0) & td["open_route"])).sum(-1)

        return -tour_length, actions

    def select_start_nodes(
        self, td, po_B: Optional[int] = None, with_greedy: bool = False
    ):
        """Select start nodes for multi-start decoding (POMO-style).

        When ``with_greedy=True`` an extra start at depot (node 0) is appended
        after all POMO customer starts.  That slot is flagged in ``greedy_mask``
        so the model decodes it greedily while all other slots are sampled.
        """
        batch = td.batch_size[0]
        device = td["locs"].device
        customer_nodes = td["locs"].shape[-2] - 1
        pomo_starts = torch.arange(
            1, customer_nodes + 1, dtype=torch.long, device=device
        )

        if getattr(self, "loss_mode", "rl") == "po" and po_B is not None:
            B = max(1, int(po_B))
            base_list = [0] + [int(x) for x in pomo_starts.tolist()]
            base_len = len(base_list)

            if B <= base_len:
                chosen = base_list[:B]
            else:
                repeats_needed = B - base_len
                full_replications = repeats_needed // customer_nodes
                remainder = repeats_needed % customer_nodes
                chosen = list(base_list)
                for _ in range(full_replications):
                    chosen.extend([int(x) for x in pomo_starts.tolist()])
                if remainder > 0:
                    perm = torch.randperm(customer_nodes, device=device)[:remainder]
                    chosen.extend([int(x) + 1 for x in perm.tolist()])
            start_nodes = torch.tensor(chosen, dtype=torch.long, device=device)
        else:
            start_nodes = pomo_starts

        # Append one depot-0 greedy start when requested (and not already present
        # as the first element from the po_B branch above).
        if with_greedy and not (start_nodes[0].item() == 0):
            depot = torch.zeros(1, dtype=torch.long, device=device)
            start_nodes = torch.cat([start_nodes, depot], dim=0)

        greedy_flags = start_nodes == 0
        num_starts = start_nodes.numel()
        selected = start_nodes.repeat_interleave(batch)
        greedy_mask = greedy_flags.repeat_interleave(batch).to(torch.bool)

        return num_starts, selected, greedy_mask

    def dataset(self, data_size=None, phase="train"):
        if phase == "train":
            td = self.generator(data_size)
            return td
        assert phase == "test"

        dataset = {}

        # Check for new directory structure: data/{problem}/test/{size}.npz
        # or old structure: data/{size}_{problem}_{distribution}.npz
        use_new_structure = False

        # Check if we have the new directory structure
        for problem in self.test_problem:
            problem_dir = pjoin(self.data_dir, problem)
            if os.path.isdir(problem_dir):
                use_new_structure = True
                break

        if use_new_structure:
            # New directory structure: data/{problem}/test/{size}.npz
            for problem in self.test_problem:
                problem_dir = pjoin(self.data_dir, problem, "test")
                if not os.path.isdir(problem_dir):
                    continue

                for size in self.test_size:
                    data_file = pjoin(problem_dir, f"{size}.npz")
                    if not os.path.exists(data_file):
                        continue

                    # Load the data
                    td = load_npz_to_tensordict(data_file).to("cpu")
                    if data_size is not None and td.batch_size[0] > data_size:
                        td = td[:data_size]

                    batch = td.batch_size[0]
                    n_locs = td["locs"].shape[1]

                    # Load optimal costs from solution file if available
                    for sol_source in ["pyvrp", "ortools", "hgs"]:
                        sol_file = pjoin(problem_dir, f"{size}_sol_{sol_source}.npz")
                        if os.path.exists(sol_file):
                            sol_data = np.load(sol_file)
                            if "costs" in sol_data:
                                costs = sol_data["costs"][:batch]
                                # Costs may be stored as negative (rewards) - convert to positive tour lengths
                                td["opt_cost"] = torch.tensor(
                                    np.abs(costs), dtype=torch.float32
                                )
                            break

                    # If no opt_cost found, use zeros (will result in NaN gaps)
                    if "opt_cost" not in td.keys():
                        td["opt_cost"] = torch.zeros(
                            batch, dtype=torch.float32, device=td["locs"].device
                        )

                    # ============ ADD DEFAULT VALUES FOR MISSING FIELDS ============
                    # Ensure demand_linehaul exists and has depot (index 0) with 0 demand
                    if "demand_linehaul" in td.keys():
                        if td["demand_linehaul"].shape[1] == n_locs - 1:
                            # Add zero demand for depot
                            td["demand_linehaul"] = torch.cat(
                                [
                                    torch.zeros(batch, 1, device=td["locs"].device),
                                    td["demand_linehaul"],
                                ],
                                dim=1,
                            )
                    else:
                        td["demand_linehaul"] = torch.zeros(
                            batch, n_locs, device=td["locs"].device
                        )

                    # Demand backhaul - default to zeros
                    if "demand_backhaul" not in td.keys():
                        td["demand_backhaul"] = torch.zeros(
                            batch, n_locs, device=td["locs"].device
                        )
                    elif td["demand_backhaul"].shape[1] == n_locs - 1:
                        td["demand_backhaul"] = torch.cat(
                            [
                                torch.zeros(batch, 1, device=td["locs"].device),
                                td["demand_backhaul"],
                            ],
                            dim=1,
                        )

                    # Time windows - default to [0, inf]
                    if "time_windows" not in td.keys():
                        tw = torch.zeros(batch, n_locs, 2, device=td["locs"].device)
                        tw[..., 1] = float("inf")
                        td["time_windows"] = tw

                    # Service time - default to zeros
                    if "service_time" not in td.keys():
                        td["service_time"] = torch.zeros(
                            batch, n_locs, device=td["locs"].device
                        )
                    elif td["service_time"].shape[1] == n_locs - 1:
                        td["service_time"] = torch.cat(
                            [
                                torch.zeros(batch, 1, device=td["locs"].device),
                                td["service_time"],
                            ],
                            dim=1,
                        )

                    # Open route - default to False (closed)
                    if "open_route" not in td.keys():
                        td["open_route"] = torch.zeros(
                            batch, 1, dtype=torch.bool, device=td["locs"].device
                        )

                    # Distance limit - default to inf
                    if "distance_limit" not in td.keys():
                        td["distance_limit"] = torch.full(
                            (batch, 1), float("inf"), device=td["locs"].device
                        )

                    # Vehicle capacity - default to 1.0 (normalized)
                    if "vehicle_capacity" not in td.keys():
                        td["vehicle_capacity"] = torch.ones(
                            batch, 1, device=td["locs"].device
                        )

                    # Capacity original - default to 1.0
                    if "capacity_original" not in td.keys():
                        td["capacity_original"] = torch.ones(
                            batch, 1, device=td["locs"].device
                        )

                    # Speed - default to 1.0
                    if "speed" not in td.keys():
                        td["speed"] = torch.ones(batch, 1, device=td["locs"].device)

                    # Build p_s_tag with 8 elements: [C, O, TW, L, B, MB, MD, size]
                    problem_lower = problem.lower()
                    keep_mask = torch.zeros(
                        (batch, 5), dtype=torch.bool, device=td["locs"].device
                    )

                    # Parse problem name for constraints
                    has_open = problem_lower.startswith("o") or "ovrp" in problem_lower
                    has_tw = "tw" in problem_lower
                    has_limit = "l" in problem_lower and (
                        "ltw" in problem_lower
                        or "bl" in problem_lower
                        or problem_lower.endswith("l")
                    )
                    has_backhaul = "b" in problem_lower and "mb" not in problem_lower
                    has_mixed_backhaul = "mb" in problem_lower
                    has_multi_depot = problem_lower.startswith("md")

                    keep_mask[:, 0] = not has_open  # C (closed route)
                    keep_mask[:, 1] = has_open  # O
                    keep_mask[:, 2] = has_tw  # TW
                    keep_mask[:, 3] = has_limit  # L
                    keep_mask[:, 4] = has_backhaul or has_mixed_backhaul  # B

                    td["p_s_tag"] = torch.cat(
                        [
                            keep_mask.float(),
                            torch.full(
                                (batch, 1),
                                float(has_mixed_backhaul),
                                dtype=torch.float32,
                                device=td["locs"].device,
                            ),  # MB
                            torch.full(
                                (batch, 1),
                                float(has_multi_depot),
                                dtype=torch.float32,
                                device=td["locs"].device,
                            ),  # MD
                            torch.full(
                                (batch, 1),
                                size / 2000,
                                dtype=torch.float32,
                                device=td["locs"].device,
                            ),  # size
                        ],
                        dim=-1,
                    )

                    # Use uniform as default distribution for new structure
                    dataset_name = f"{size}_{problem}_uniform"
                    dataset[dataset_name] = td
        else:
            # Old structure: data/{size}_{problem}_{distribution}.npz
            if len(self.test_file) == 0:
                all_test_file = os.listdir(self.data_dir)
                for file_i in all_test_file:
                    if not file_i.endswith(".npz"):
                        continue
                    spilt_file = file_i.split(".")[0].split("_")
                    if len(spilt_file) < 3:
                        continue
                    if spilt_file[-1] != "hgs" and spilt_file[-1] != "uniform":
                        continue
                    s_i, p_i, d_i = int(spilt_file[0]), spilt_file[1], spilt_file[2]
                    if (
                        s_i in self.test_size
                        and p_i in self.test_problem
                        and d_i in self.test_distribution
                    ):
                        self.test_file.append(pjoin(self.data_dir, file_i))
                        if spilt_file[-1] != "hgs":
                            self.test_dataloader_names.append(file_i.split(".")[0])
                        else:
                            self.test_dataloader_names.append(file_i.split(".")[0][:-4])

            for name, _f in zip(self.test_dataloader_names, self.test_file):
                td = load_npz_to_tensordict(_f).to("cpu")
                if data_size is not None and td.batch_size[0] > data_size:
                    td = td[:data_size]

                tmp_size = int(name.split("_")[0])
                tmp_p = name.split("_")[1]

                # Build p_s_tag with 8 elements: [C, O, TW, L, B, MB, MD, size]
                keep_mask = torch.zeros((td.shape[0], 5), dtype=torch.bool)
                for id_, p_tag in enumerate(["c", "o", "tw", "l", "b"]):
                    keep_mask[:, id_] = True if p_tag in tmp_p else False
                keep_mask[:, 0:1] = ~keep_mask[:, 1:2]  # C = not O

                # Detect MB and MD from problem name
                is_mb = "mb" in tmp_p.lower()
                is_md = tmp_p.lower().startswith("md")

                td["p_s_tag"] = torch.cat(
                    [
                        keep_mask.float(),
                        torch.full((td.shape[0], 1), float(is_mb), dtype=torch.float32),
                        torch.full((td.shape[0], 1), float(is_md), dtype=torch.float32),
                        torch.full_like(
                            td["open_route"],
                            tmp_size / 2000,
                            dtype=torch.float32,
                            device=keep_mask.device,
                        ),
                    ],
                    dim=-1,
                )

                dataset[name] = td

        return dataset  # dict{str: td}

    @staticmethod
    def check_variants(td):
        """Check which variants are present in the TensorDict."""
        has_open = td["open_route"].squeeze(-1)
        has_tw = (td["time_windows"][:, :, 1] != float("inf")).any(-1)
        has_limit = (td["distance_limit"] != float("inf")).squeeze(-1)
        has_backhaul = (td["demand_backhaul"] != 0).any(-1)
        backhaul_class = td.get(
            "backhaul_class", torch.ones_like(has_open, dtype=torch.float32)
        )
        return has_open, has_tw, has_limit, has_backhaul, backhaul_class

    @staticmethod
    def get_variant_names(td: TensorDict) -> Union[str, List[str]]:
        """Get variant names for instances in the TensorDict."""
        has_open, has_tw, has_limit, has_backhaul, backhaul_class = (
            MTVRPEnv.check_variants(td)
        )

        def _name(o, b, bc, l_, tw):
            if not o and not b and not l_ and not tw:
                return "CVRP"
            name = "VRP"
            if o:
                name = "O" + name
            if b:
                if bc == 2:
                    name += "M"
                name += "B"
            if l_:
                name += "L"
            if tw:
                name += "TW"
            return name

        if len(has_open.shape) == 0:
            return _name(has_open, has_backhaul, backhaul_class, has_limit, has_tw)
        else:
            return [
                _name(o, b, bc, l_, tw)
                for o, b, bc, l_, tw in zip(
                    has_open, has_backhaul, backhaul_class, has_limit, has_tw
                )
            ]

    def _make_spec(self):
        """Make the observation and action specs."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.generator.min_loc,
                maximum=self.generator.max_loc,
                shape=(self.generator.num_loc + 1, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1,), dtype=torch.int64, device=self.device
            ),
            demand_linehaul=BoundedTensorSpec(
                minimum=-self.generator.capacity,
                maximum=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            demand_backhaul=BoundedTensorSpec(
                minimum=-self.generator.capacity,
                maximum=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.generator.num_loc + 1, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            minimum=0,
            maximum=self.generator.num_loc + 1,
            shape=(1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), dtype=torch.float32, device=self.device
        )
        self.done_spec = UnboundedDiscreteTensorSpec(
            shape=(1,), dtype=torch.bool, device=self.device
        )

    def print_presets(self):
        self.generator.print_presets()

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def to(self, device):
        if device is None:
            return self
        return super().to(device)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["rng"] = state["rng"].get_state()
        return state

    def __setstate__(self, state, set_seed: bool = True):
        attrs_to_skip = [
            "test_size",
            "test_problem",
            "test_distribution",
            "test_file",
            "test_dataloader_names",
        ]
        for attr in attrs_to_skip:
            if attr in state:
                del state[attr]
        if not set_seed:
            del state["rng"]
        self.__dict__.update(state)
        if set_seed:
            self.rng = torch.manual_seed(0)
            self.rng.set_state(state["rng"].cpu())
