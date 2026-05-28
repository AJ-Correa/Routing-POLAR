"""
Multi-Depot Multi-Task VRP Environment

Supports all 24 multi-depot VRP variants:
- 16 standard MDVRP variants with classical backhaul (backhaul_class=1)
- 8 mixed backhaul MDVRP variants (backhaul_class=2)

Node indexing:
- Depots: indices 0 to num_depots-1
- Customers: indices num_depots to num_locs-1
"""

import os
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

from utils.functions import gather_by_index, get_distance, load_npz_to_tensordict
from envs.mtdvrp.generator import MTVRPGenerator

EPS = 1e-6


def get_starting_points(
    actions: torch.Tensor, num_depots: torch.Tensor
) -> torch.Tensor:
    """Get the starting depot for each step in the sequence.

    For MDVRP, we need to track which depot each route starts from to correctly
    compute return distances.

    Args:
        actions: [batch, seq_len] action sequence
        num_depots: [batch, 1] or scalar number of depots

    Returns:
        starting_points: [batch, seq_len] depot index for each step
    """
    # Create mask for numbers < num_depots
    mask = actions < num_depots  # shape: (batch_size, seq_len)
    batch_size, seq_len = actions.shape

    # Compute the cumulative sum of the mask to get segment IDs
    segment_ids = torch.cumsum(mask.long(), dim=1)  # shape: (batch_size, seq_len)

    # Adjust segment IDs for indexing (shift by -1)
    segment_indices = segment_ids - 1

    # Create a mask for valid segment positions
    valid_positions = segment_ids > 0

    # Compute the number of masked elements per batch
    num_values_per_batch = mask.sum(dim=1)  # shape: (batch_size,)
    max_num_values = num_values_per_batch.max().item()

    # Generate batch indices
    batch_indices = (
        torch.arange(batch_size, device=actions.device)
        .unsqueeze(1)
        .expand(batch_size, seq_len)
    )

    # Get indices where mask is True
    masked_indices = torch.where(
        mask,
        torch.cumsum(mask.long(), dim=1) - 1,
        torch.tensor(-1, device=actions.device),
    )
    valid_masked_positions = masked_indices >= 0

    # Gather valid batch and masked indices
    valid_batch_indices = batch_indices[valid_masked_positions]
    valid_masked_indices = masked_indices[valid_masked_positions]
    valid_actions = actions[valid_masked_positions]

    # Initialize padded values tensor
    values_padded = torch.zeros(
        batch_size, max_num_values, dtype=actions.dtype, device=actions.device
    )

    # Fill in the padded values tensor
    values_padded[valid_batch_indices, valid_masked_indices] = valid_actions

    # Initialize the starting_points tensor
    starting_points = torch.zeros_like(actions)

    # Fill in the starting_points tensor using advanced indexing
    starting_points[valid_positions] = values_padded[
        batch_indices[valid_positions], segment_indices[valid_positions]
    ]

    return starting_points


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
    """Multi-Depot Multi-Task VRP Environment.

    Same as single-depot but with multiple depots at indices 0 to num_depots-1.
    """

    name = "mtdvrp"

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
        """Execute one step in the environment. Same logic as single-depot but depot check uses num_depots."""
        num_depots = int(td["num_depots"][0].item())
        depot_idx = torch.arange(num_depots, device=td.device)

        # Get locations and distance
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc = gather_by_index(td["locs"], prev_node)
        curr_loc = gather_by_index(td["locs"], curr_node)
        depot_loc = gather_by_index(td["locs"], td["current_depot"])
        distance = get_distance(prev_loc, curr_loc)[..., None]
        dist2depot = get_distance(prev_loc, depot_loc)[..., None]

        # for indexing
        in_depot = torch.isin(curr_node, depot_idx)
        not_in_depot = ~in_depot[..., None]  # note the dimensions
        depot2depot = torch.isin(prev_node, depot_idx) & in_depot

        distance[in_depot] = dist2depot[in_depot]  # always return to *current* depot
        distance[in_depot & td["open_route"].squeeze(-1)] = (
            0.0  # discard for open route
        )
        distance[depot2depot] = 0.0
        td["current_depot"][in_depot] = curr_node[in_depot]  # update current depot

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        # we cannot start before we arrive and we should start at least at start times
        curr_time = not_in_depot * (
            torch.max(td["current_time"] + distance / td["speed"], start_times)
            + service_time
        )

        # Update current route length (reset at depot)
        curr_route_length = not_in_depot * (td["current_route_length"] + distance)
        total_distance = td["total_distance"] + distance

        # Linehaul (delivery) demands
        selected_demand_linehaul = gather_by_index(
            td["demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["demand_backhaul"], curr_node, dim=1, squeeze=False
        )

        # Backhaul (pickup) demands
        # this holds for backhaul_classes 0, 1, and 2:
        used_capacity_linehaul = not_in_depot * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = not_in_depot * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        # Done when all customers are visited
        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited[..., num_depots:].all(-1)
        reward = torch.zeros_like(
            done
        ).float()  # we use the `get_reward` method to compute the reward

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "total_distance": total_distance,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
            }
        )
        td = self.get_action_mask(td)
        return td

    def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        """Reset the environment."""
        if batch_size is None:
            batch_size = td.batch_size
        if td is None or td.is_empty():
            td = self.generator(batch_size=batch_size).to("cuda")
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.to(td.device)
        return super().reset(td, batch_size=batch_size)

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        """Internal reset implementation."""
        device = td.device
        num_depots = int(td["num_depots"][0].item())

        # Demands: linehaul (C) and backhaul (B). Backhaul defaults to 0
        demand_linehaul = torch.cat(
            [
                torch.zeros_like(td["demand_linehaul"][..., :num_depots]),
                td["demand_linehaul"],
            ],
            dim=1,
        )
        demand_backhaul = td.get(
            "demand_backhaul",
            torch.zeros_like(td["demand_linehaul"]),
        )
        demand_backhaul = torch.cat(
            [
                torch.zeros_like(td["demand_linehaul"][..., :num_depots]),
                demand_backhaul,
            ],
            dim=1,
        )
        # Backhaul class (MB). 1 is the default backhaul class
        backhaul_class = td.get(
            "backhaul_class",
            torch.full((*batch_size, 1), 1, dtype=torch.int32),
        )

        # Time windows (TW). Defaults to [0, inf] and service time to 0
        time_windows = td.get("time_windows", None)
        if time_windows is None:
            time_windows = torch.zeros_like(td["locs"])
            time_windows[..., 1] = float("inf")
        service_time = td.get("service_time", torch.zeros_like(demand_linehaul))

        # Open (O) route. Defaults to 0
        open_route = td.get(
            "open_route", torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool)
        )

        # Distance limit (L). Defaults to inf
        distance_limit = td.get(
            "distance_limit", torch.full_like(demand_linehaul[..., :1], float("inf"))
        )

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "num_depots": td["num_depots"],
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
                "current_depot": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_node": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_route_length": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),  # for distance limits
                "current_time": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),  # for time windows
                "total_distance": torch.full(
                    (*batch_size, 1), -1, dtype=torch.float32, device=device
                ),  # for reward calculation
                "used_capacity_backhaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),  # for capacity constraints in backhaul
                "used_capacity_linehaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),  # for capacity constraints in linehaul
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ),
                "depot_available": torch.ones(
                    (*batch_size, num_depots),
                    dtype=torch.bool,
                    device=device,
                ),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset = self.get_action_mask(td_reset)
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> TensorDict:
        """Compute action mask. Same logic as single-depot but with multiple depots."""
        num_depots = int(td["num_depots"][0].item())
        if (td["total_distance"] == -1).all():
            # in the first step sample a depot
            initial_mask = torch.zeros_like(td["demand_linehaul"], dtype=torch.bool)
            initial_mask[..., :num_depots] = True
            td["total_distance"][...] = 0
            td.set("action_mask", initial_mask)
            return td

        curr_node = td["current_node"]  # note that this was just updated!
        locs = td["locs"]
        d_ij = get_distance(
            gather_by_index(locs, curr_node)[..., None, :], locs
        )  # i (current) -> j (next)

        # distance to *current* depot
        curr_depot_loc = locs[
            torch.arange(locs.shape[0], device=locs.device), None, td["current_depot"]
        ]
        d_j0 = get_distance(locs, curr_depot_loc)  # j (next) -> 0 (depot)

        # Time constraint (TW):
        early_tw, late_tw = (
            td["time_windows"][..., 0],
            td["time_windows"][..., 1],
        )
        arrival_time = td["current_time"] + (d_ij / td["speed"])
        # can reach in time -> only need to *start* in time
        can_reach_customer = arrival_time < late_tw + EPS
        # we must ensure that we can return to depot in time *if* route is closed
        # i.e. start time + service time + time back to depot < late_tw
        can_reach_depot = (
            torch.max(arrival_time, early_tw)
            + td["service_time"]
            + (d_j0 / td["speed"])
        ) * ~td["open_route"] < late_tw[
            ..., 0:1
        ] + EPS  # note tws are the same for all depots

        # Distance limit (L): do not add distance to depot if open route (O)
        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"] + EPS
        )

        # Capacity constraints linehaul (C) and backhaul (B)
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"]
            > td["vehicle_capacity"] + EPS
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"]
            > td["vehicle_capacity"] + EPS
        )

        # Backhaul class 1 (classical backhaul) (B)
        # every customer is either backhaul or linehaul, all linehauls are visited before backhauls
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[
            ..., None
        ]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["demand_backhaul"],
                idx=curr_node,
                dim=1,
                squeeze=False,
            )
            > 0
        )
        meets_demand_constraint_backhaul_1 = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))

        # Backhaul class 2 (mixed pickup and delivery / mixed backhaul) (MB)
        # to serve linehaul customers we additionally need to check the remaining capacity in the vehicle
        # capacity is vehicle_capacity-used_capacity_backhauls, as all used_capacity_linehaul at this point have already been *delivered*
        cannot_serve_linehaul = (
            td["demand_linehaul"]
            > td["vehicle_capacity"] - td["used_capacity_backhaul"] + EPS
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )

        # Now we merge the constraints of backhaul class 1 and 2 depending on the backhaul class
        meets_demand_constraint = (
            (td["backhaul_class"] == 1) & meets_demand_constraint_backhaul_1
        ) | ((td["backhaul_class"] == 2) & meets_demand_constraint_backhaul_2)

        # Condense constraints
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

        # Mask depot: don't visit depot if coming from there and there are still customer nodes I can visit
        can_visit[:, :num_depots] = ~(
            (torch.isin(curr_node, torch.arange(num_depots, device=curr_node.device)))
            & (can_visit[:, num_depots:].sum(-1) > 0)
            # TODO depot available?
        ).reshape(-1, 1)

        # If we are in a depot, not all customers have been visited, but we cannot visit any customer, we have a deadlock
        depot_deadlock = (
            (td["current_node"] < num_depots)
            & (~td["visited"][..., num_depots:].all(-1))
            & (~can_visit[:, num_depots:].any(-1))
        )

        # # if we are in a deadlock and only the current depot is available, set all depots as available
        depot_available = torch.where(
            depot_deadlock[..., None] & can_visit[:, :num_depots].sum(-1, keepdim=True)
            == 0,
            torch.ones_like(td["depot_available"]),
            td["depot_available"],
        )  # [b, num_depots]

        # set current depot as unavailable if there is a deadlock since it got us stuck
        depot_available.scatter_(
            -1, td["current_depot"][..., None], ~depot_deadlock[..., None]
        )
        # if there is a deadlock, set visitable depots as depot_available
        can_visit[:, :num_depots] = torch.where(
            depot_deadlock[:, None], depot_available, can_visit[:, :num_depots]
        )  # [b, num_depots]

        td.set("depot_available", depot_available)
        td.set("action_mask", can_visit)
        return td

    def get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Compute reward (negative tour length)."""
        go_from = actions  # note: we don't append any slack action here
        go_to = torch.roll(go_from, -1, dims=1)  # [b, seq_len]
        loc_from = gather_by_index(td["locs"], go_from)
        loc_to = gather_by_index(td["locs"], go_to)

        starting_points = get_starting_points(actions, td["num_depots"])
        actual_depot = torch.roll(
            starting_points, 1, dims=1
        )  # "overwrite" the destination depot with the actual depot
        loc_actual_depot = gather_by_index(td["locs"], actual_depot)

        # Get tour length. If route is open and goes to depot, don't count the distance
        distances = get_distance(loc_from, loc_to)  # [b, seq_len]
        distances_to_depot = get_distance(loc_from, loc_actual_depot)  # [b, seq_len]

        # where the route goes back to depot, the distance is to depot
        is_depot = go_to < td["num_depots"]
        distances = torch.where(
            is_depot, distances_to_depot * ~td["open_route"], distances
        )

        # If depot to depot, distance is 0
        is_depot_to_depot = (go_from < td["num_depots"]) & (go_to < td["num_depots"])
        distances = torch.where(
            is_depot_to_depot, torch.zeros_like(distances), distances
        )

        # Sum up and return
        tour_length = distances.sum(-1)  # [b]
        return -tour_length, actions  # reward is negative cost

    def select_start_nodes(
        self, td, po_B: Optional[int] = None, with_greedy: bool = False
    ):
        """Select start nodes for multi-start decoding (POMO-style).

        For multi-depot variants, we create all depot-customer combinations.
        For a problem with 3 depots and 50 customers, this gives 3 * 50 = 150 starts.

        The starts are ordered as:
        - First 50: depot 0 with customers 0-49 (indices num_depots to num_depots+49)
        - Next 50: depot 1 with customers 0-49
        - Last 50: depot 2 with customers 0-49

        If po_B is specified, we take the first po_B starts from this ordering.
        E.g., po_B=55 means: 50 from depot 0, 5 from depot 1.

        When ``with_greedy=True`` an extra start at depot 0 (node 0) is appended.
        That slot is flagged in ``greedy_mask`` so the model decodes it greedily.
        For multi-depot problems a second step action of depot 0 itself is stored
        in ``_pomo_customer_starts`` for that slot (the decoder will re-visit depot
        0, which is a no-op first customer, giving a valid greedy rollout).

        Returns:
            num_starts: number of POMO starts
            selected: depot indices to select first (batch-expanded)
            greedy_mask: True only for the appended greedy slot

        Also stores self._pomo_customer_starts for use after first depot selection.
        """
        batch = td.batch_size[0]
        device = td["locs"].device
        num_depots = td["num_depots"][0].int().item()
        num_customers = td["locs"].shape[-2] - num_depots

        if num_depots > 1:
            # Multi-depot: create all depot-customer combinations
            # Total starts = num_depots * num_customers

            # Customer indices: num_depots, num_depots+1, ..., num_depots+num_customers-1
            customer_indices = torch.arange(
                num_depots, num_depots + num_customers, dtype=torch.long, device=device
            )

            # For each depot, pair with all customers
            # Customers repeated for each depot: [c0,c1,...,cN, c0,c1,...,cN, c0,c1,...,cN]
            all_customer_starts = customer_indices.repeat(num_depots)

            # Depot for each start: [0,0,...,0, 1,1,...,1, 2,2,...,2]
            all_depot_starts = torch.arange(
                num_depots, dtype=torch.long, device=device
            ).repeat_interleave(num_customers)

            total_possible_starts = num_depots * num_customers
        else:
            # Single-depot: start from all customers only
            all_customer_starts = torch.arange(
                num_depots, num_depots + num_customers, dtype=torch.long, device=device
            )
            all_depot_starts = torch.zeros(
                num_customers, dtype=torch.long, device=device
            )
            total_possible_starts = num_customers

        if po_B is not None:
            # Limit to first po_B starts
            B = max(1, min(int(po_B), total_possible_starts))
            customer_starts = all_customer_starts[:B]
            depot_starts = all_depot_starts[:B]
        else:
            customer_starts = all_customer_starts
            depot_starts = all_depot_starts

        # Greedy flags: False for all regular POMO starts
        greedy_flags = torch.zeros(
            customer_starts.numel(), dtype=torch.bool, device=device
        )

        # Append one greedy slot: depot 0, customer start = first customer index
        if with_greedy:
            greedy_depot = torch.zeros(1, dtype=torch.long, device=device)
            greedy_customer = torch.tensor(
                [num_depots], dtype=torch.long, device=device
            )
            depot_starts = torch.cat([depot_starts, greedy_depot], dim=0)
            customer_starts = torch.cat([customer_starts, greedy_customer], dim=0)
            greedy_flags = torch.cat(
                [greedy_flags, torch.ones(1, dtype=torch.bool, device=device)], dim=0
            )

        num_starts = customer_starts.numel()

        # The first action should be the depot selection
        # After batchify, we'll have num_starts * batch instances
        # selected_depots: which depot each instance starts from
        selected_depots = depot_starts.repeat_interleave(batch)

        # Store customer starts for the second step (after depot is selected)
        # This will be used by the model to set the second action
        self._pomo_customer_starts = customer_starts.repeat_interleave(batch)

        greedy_mask = greedy_flags.repeat_interleave(batch)

        return num_starts, selected_depots, greedy_mask

    def get_pomo_customer_starts(self):
        """Get the customer starts for the second POMO step (after depot selection)."""
        return getattr(self, "_pomo_customer_starts", None)

    def dataset(self, data_size=None, phase="train"):
        """Create dataset for training or testing."""
        if phase == "train":
            return self.generator(data_size)

        assert phase == "test"
        if len(self.test_file) == 0:
            all_test_dirs = [
                d
                for d in os.listdir(self.data_dir)
                if d.lower().startswith("md")
                and os.path.isdir(os.path.join(self.data_dir, d))
            ]

            for variant_dir in all_test_dirs:
                test_path = os.path.join(self.data_dir, variant_dir, "test")
                if not os.path.exists(test_path):
                    continue

                for size_file in os.listdir(test_path):
                    if not size_file.endswith(".npz") or "_sol" in size_file:
                        continue

                    size = int(size_file.replace(".npz", ""))
                    if self.test_size and size not in self.test_size:
                        continue

                    self.test_file.append(os.path.join(test_path, size_file))
                    self.test_dataloader_names.append(f"{size}_{variant_dir}_uniform")

        dataset = {}
        for name, _f in zip(self.test_dataloader_names, self.test_file):
            td = load_npz_to_tensordict(_f).to("cpu")
            if data_size is not None and td.batch_size[0] > data_size:
                td = td[:data_size]

            tmp_size = int(name.split("_")[0])
            tmp_p = name.split("_")[1].lower()

            # Build p_s_tag: [C, O, TW, L, B, MB, MD, size]
            batch_size = td.shape[0]
            keep_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
            for id_, p_tag in enumerate(["c", "o", "tw", "l", "b"]):
                keep_mask[:, id_] = p_tag in tmp_p
            keep_mask[:, 0:1] = ~keep_mask[:, 1:2]  # C = not O

            is_mb = "mb" in tmp_p
            is_md = "md" in tmp_p

            td["p_s_tag"] = torch.cat(
                [
                    keep_mask.float(),
                    torch.full((batch_size, 1), float(is_mb), dtype=torch.float32),
                    torch.full((batch_size, 1), float(is_md), dtype=torch.float32),
                    torch.full((batch_size, 1), tmp_size / 2000, dtype=torch.float32),
                ],
                dim=-1,
            )

            dataset[name] = td

        return dataset

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
        num_depots = td["num_depots"][0].int().item()
        multi_depot = num_depots > 1
        return has_open, has_tw, has_limit, has_backhaul, backhaul_class, multi_depot

    @staticmethod
    def get_variant_names(td: TensorDict) -> Union[str, List[str]]:
        """Get variant names for instances in the TensorDict."""
        has_open, has_tw, has_limit, has_backhaul, backhaul_class, multi_depot = (
            MTVRPEnv.check_variants(td)
        )

        def _name(o, b, bc, l_, tw, md):
            if not o and not b and not l_ and not tw:
                name = "CVRP"
            else:
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
            if md:
                name = "MD" + name
            return name

        if len(has_open.shape) == 0:
            return _name(
                has_open, has_backhaul, backhaul_class, has_limit, has_tw, multi_depot
            )

        return [
            _name(o, b, bc, l_, tw, multi_depot)
            for o, b, bc, l_, tw in zip(
                has_open, has_backhaul, backhaul_class, has_limit, has_tw
            )
        ]

    def _make_spec(self):
        """Make the observation and action specs."""
        total_nodes = self.generator.num_loc + self.generator.num_depots
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.generator.min_loc,
                maximum=self.generator.max_loc,
                shape=(total_nodes, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1,), dtype=torch.int64, device=self.device
            ),
            demand_linehaul=BoundedTensorSpec(
                minimum=-self.generator.capacity,
                maximum=self.generator.max_demand,
                shape=(total_nodes, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            demand_backhaul=BoundedTensorSpec(
                minimum=-self.generator.capacity,
                maximum=self.generator.max_demand,
                shape=(total_nodes, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(total_nodes, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            minimum=0,
            maximum=total_nodes,
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
