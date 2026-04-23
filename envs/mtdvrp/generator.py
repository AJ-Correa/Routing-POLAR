"""
Multi-Depot Multi-Task VRP Generator

Supports all 24 multi-depot VRP variants:
- 16 standard MDVRP variants with classical backhaul (backhaul_class=1)
- 8 mixed backhaul MDVRP variants (backhaul_class=2)

Same pattern as single-depot generator but with multiple depots.
"""

import torch
import logging
from typing import Callable, Tuple, Union
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from utils.functions import get_distance, save_tensordict_to_npz

log = logging.getLogger(__name__)


def get_vehicle_capacity(num_loc: int) -> int:
    """Capacity computation for MDVRP."""
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap


def get_sampler(
    distribution: Union[int, float, str, type, Callable],
    val_name: str = 'loc', low: float = 0, high: float = 1.0, **kwargs,
):
    """Get the sampler for the variable with the given distribution."""
    if isinstance(distribution, (int, float)):
        return Uniform(low=distribution, high=distribution)
    elif distribution == Uniform or distribution == "uniform":
        return Uniform(low=low, high=high)
    elif isinstance(distribution, Callable):
        return distribution(**kwargs)
    else:
        raise ValueError(f"Invalid distribution type of {distribution}")


# Variant generation presets for multi-depot VRP
VARIANT_GENERATION_PRESETS = {
    # Training presets
    "all": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
    "single_feat": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
    "mixed_backhaul": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 1.0},
    
    # Standard backhaul MD variants (backhaul_class=1)
    "mdcvrp": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0},
    "mdovrp": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0},
    "mdvrpb": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "mdvrpl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0},
    "mdvrptw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0},
    "mdovrptw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0},
    "mdovrpb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "mdovrpl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0},
    "mdvrpbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "mdvrpbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "mdvrpltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0},
    "mdovrpbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "mdovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "mdovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0},
    "mdvrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    "mdovrpbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    
    # Mixed backhaul MD variants (backhaul_class=2)
    "mdvrpmb": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "mdovrpmb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "mdvrpmbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "mdvrpmbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "mdovrpmbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "mdovrpmbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "mdvrpmbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    "mdovrpmbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
}

# Mixed backhaul MD variants list
MIXED_BACKHAUL_MD_VARIANTS = [
    "mdvrpmb", "mdovrpmb", "mdvrpmbl", "mdvrpmbtw",
    "mdovrpmbl", "mdovrpmbtw", "mdvrpmbltw", "mdovrpmbltw"
]


class MTVRPGenerator:
    """Multi-Depot MTVRP Generator.
    
    Same pattern as single-depot generator but with multiple depots.
    """
    
    def __init__(
        self,
        num_loc: int = 20,
        num_depots: int = 3,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        capacity: float = None,
        min_demand: int = 1,
        max_demand: int = 10,
        min_backhaul: int = 1,
        max_backhaul: int = 10,
        scale_demand: bool = True,
        max_time: float = 4.6,
        backhaul_ratio: float = 0.2,
        backhaul_class: int = 1,
        sample_backhaul_class: bool = False,
        max_distance_limit: float = 2.8,
        speed: float = 1.0,
        prob_open: float = 0.5,
        prob_time_window: float = 0.5,
        prob_limit: float = 0.5,
        prob_backhaul: float = 0.5,
        variant_preset: str = None,
        use_combinations: bool = True,
        subsample: bool = True,
        **kwargs,
    ) -> None:
        self.num_loc = num_loc
        self.num_depots = num_depots
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.loc_sampler = get_sampler(loc_distribution, low=min_loc, high=max_loc, **kwargs)
        
        if capacity is None:
            capacity = get_vehicle_capacity(num_loc)
        self.capacity = capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_backhaul = min_backhaul
        self.max_backhaul = max_backhaul
        self.scale_demand = scale_demand
        self.backhaul_ratio = backhaul_ratio
        self.backhaul_class = backhaul_class
        self.sample_backhaul_class = sample_backhaul_class
        
        self.max_time = max_time
        self.max_distance_limit = max_distance_limit
        self.speed = speed
        
        assert not (subsample and (variant_preset is None)), (
            "Cannot use subsample if variant_preset is not specified."
        )
        
        if variant_preset is not None:
            log.info(f"Using variant generation preset: {variant_preset}")
            variant_probs = VARIANT_GENERATION_PRESETS.get(variant_preset)
            assert variant_probs is not None, (
                f"Variant preset '{variant_preset}' not found. "
                f"Available: {list(VARIANT_GENERATION_PRESETS.keys())}"
            )
            # Check if this is a mixed backhaul variant
            if variant_preset in MIXED_BACKHAUL_MD_VARIANTS:
                self.backhaul_class = 2
        else:
            variant_probs = {
                "O": prob_open,
                "TW": prob_time_window,
                "L": prob_limit,
                "B": prob_backhaul,
            }
        
        for key, prob in variant_probs.items():
            assert 0 <= prob <= 1, f"Probability {key} must be between 0 and 1"
        
        self.variant_probs = variant_probs
        self.variant_preset = variant_preset
        
        if isinstance(variant_preset, str) and variant_preset not in ("all", "single_feat", "mixed_backhaul"):
            log.info(f"{variant_preset} selected. Will not use feature combination!")
            use_combinations = False
        
        self.use_combinations = use_combinations
        self.subsample = subsample
    
    def __call__(self, batch_size) -> TensorDict:
        """Generate a batch of multi-depot MTVRP instances."""
        # Number of depots
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        num_depots = torch.full(
            size=(*batch_size, 1), fill_value=self.num_depots, dtype=torch.int32, device='cuda'
        )

        # Locations
        locs = self.generate_locations(
            batch_size=batch_size, num_depots=self.num_depots, num_loc=self.num_loc
        )

        # Vehicle capacity (C, B) - applies to both linehaul and backhaul
        vehicle_capacity = torch.full(
            (*batch_size, 1), self.capacity, dtype=torch.float32, device='cuda'
        )
        capacity_original = vehicle_capacity.clone()

        # linehaul demand / delivery (C) and backhaul / pickup demand (B)
        demand_linehaul, demand_backhaul = self.generate_demands(
            batch_size=batch_size, num_loc=self.num_loc
        )

        backhaul_class = self.generate_backhaul_class(
            shape=(*batch_size, 1), sample=self.sample_backhaul_class
        )

        # Open (O)
        open_route = self.generate_open_route(shape=(*batch_size, 1))

        # Time windows (TW)
        speed = self.generate_speed(shape=(*batch_size, 1))
        time_windows, service_time = self.generate_time_windows(
            locs=locs,
            speed=speed,
        )

        # Distance limit (L)
        distance_limit = self.generate_distance_limit(shape=(*batch_size, 1), locs=locs)

        # scaling
        if self.scale_demand:
            demand_backhaul = demand_backhaul / vehicle_capacity
            demand_linehaul = demand_linehaul / vehicle_capacity
            vehicle_capacity = vehicle_capacity / vehicle_capacity

        # Put all variables together
        td = TensorDict(
            {
                "num_depots": num_depots,
                "locs": locs,
                "demand_backhaul": demand_backhaul,  # (C)
                "demand_linehaul": demand_linehaul,  # (B)
                "backhaul_class": backhaul_class,  # (B)
                "distance_limit": distance_limit,  # (L)
                "time_windows": time_windows,  # (TW)
                "service_time": service_time,  # (TW)
                "vehicle_capacity": vehicle_capacity,  # (C)
                "capacity_original": capacity_original,  # unscaled capacity (C)
                "open_route": open_route,  # (O)
                "speed": speed,  # common
            },
            batch_size=batch_size,
            device='cuda',
        )

        if self.subsample:
            # Subsample problems based on given instructions
            td = self.subsample_problems(td)
        return td
    
    def subsample_problems(self, td):
        """Create subproblems starting from seed probabilities depending on their variant.
        If random seed sampled in [0, 1] in batch is greater than prob, remove the constraint
        thus, if prob high, it is less likely to remove the constraint (i.e. prob=0.9, 90% chance to keep constraint)
        """
        batch_size = td.batch_size[0]
        device = td.device

        variant_probs = torch.tensor(list(self.variant_probs.values()), device=device)

        if self.use_combinations:
            # in a batch, multiple variants combinations can be picked
            keep_mask = torch.rand(batch_size, 4, device=device) <= variant_probs  # O, TW, L, B
        else:
            # in a batch, only a variant can be picked.
            # we assign a 0.5 prob to the last variant (which is normal cvrp)
            if self.variant_preset in list(
                VARIANT_GENERATION_PRESETS.keys()
            ) and self.variant_preset not in (
                "all",
                "cvrp",
                "single_feat",
                "single_feat_otw",
            ):
                cvrp_prob = 0
            else:
                cvrp_prob = 0.5
            if self.variant_preset in ("all", "cvrp", "single_feat", "single_feat_otw"):
                probs = torch.tensor(list(self.variant_probs.values()) + [cvrp_prob], device=device)
                indices = torch.distributions.Categorical(
                    probs[None].repeat(batch_size, 1)
                ).sample()
                if self.variant_preset == "single_feat_otw":
                    keep_mask = torch.zeros((batch_size, 6), dtype=torch.bool, device=device)
                    keep_mask[torch.arange(batch_size, device=device), indices] = True

                    # If keep_mask[:, 4] is True, make both keep_mask[:, 0] and keep_mask[:, 1] True
                    keep_mask[:, :2] |= keep_mask[:, 4:5]
                else:
                    keep_mask = torch.zeros((batch_size, 5), dtype=torch.bool, device=device)
                    keep_mask[torch.arange(batch_size, device=device), indices] = True
            else:
                # if the variant is specified, we keep the attributes with probability > 0
                keep_mask = torch.zeros((batch_size, 4), dtype=torch.bool, device=device)
                indices = torch.nonzero(variant_probs).squeeze()
                keep_mask[:, indices] = True

        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_window(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])

        has_open = keep_mask[:, 0]
        has_tw = keep_mask[:, 1]
        has_limit = keep_mask[:, 2]
        has_any_backhaul = keep_mask[:, 3]

        # Check if the instance class is actually 'mixed' (class 2)
        is_mixed_class = (td["backhaul_class"].squeeze(-1) == 2)
        is_mixed_backhaul = has_any_backhaul & is_mixed_class
        is_standard_backhaul = has_any_backhaul & (~is_mixed_class)
        is_multi_depot = torch.ones(batch_size, dtype=torch.bool, device=device)  # Always True for MDVRP
        
        p_s_tag = torch.cat([
            (~has_open[:, None]).float(),           # C
            has_open[:, None].float(),              # O
            has_tw[:, None].float(),                # TW
            has_limit[:, None].float(),             # L
            is_standard_backhaul[:, None].float(),  # B
            is_mixed_backhaul[:, None].float(),     # MB
            is_multi_depot[:, None].float(),        # MD
            torch.full((*td.batch_size, 1), (td['locs'].shape[1] - 1) / 2000, dtype=torch.float32), # size
        ], dim=-1)
        
        td['p_s_tag'] = p_s_tag

        return td

    @staticmethod
    def _default_open(td, remove):
        td["open_route"][remove] = False
        return td

    @staticmethod
    def _default_time_window(td, remove):
        default_tw = torch.zeros_like(td["time_windows"])
        default_tw[..., 1] = float("inf")
        td["time_windows"][remove] = default_tw[remove]
        td["service_time"][remove] = torch.zeros_like(td["service_time"][remove])
        return td

    @staticmethod
    def _default_distance_limit(td, remove):
        td["distance_limit"][remove] = float("inf")
        return td

    @staticmethod
    def _default_backhaul(td, remove):
        # by default, where there is a backhaul, linehaul is 0. therefore, we add backhaul to linehaul
        # and set backhaul to 0 where we want to remove backhaul
        td["demand_linehaul"][remove] = (
            td["demand_linehaul"][remove] + td["demand_backhaul"][remove]
        )
        td["demand_backhaul"][remove] = 0
        return td

    def generate_locations(self, batch_size, num_depots, num_loc) -> torch.Tensor:
        """Generate seed locations.

        Returns:
            locs: [B, N+1, 2] where the first location is the depot.
        """
        locs = torch.FloatTensor(*batch_size, num_depots + num_loc, 2).uniform_(
            self.min_loc, self.max_loc
        ).cuda()
        return locs

    def generate_demands(self, batch_size: int, num_loc: int) -> torch.Tensor:
        """Classical lineahul demand / delivery from depot (C) and backhaul demand / pickup to depot (B) generation.
        Initialize the demand for nodes except the depot, which are added during reset.
        Demand sampling Following Kool et al. (2019), demands as integers between 1 and 10.
        Generates a slightly different distribution than using torch.randint.

        Returns:
            linehaul_demand: [B, N]
            backhaul_demand: [B, N]
        """
        linehaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_demand - 1, self.max_demand - 1
        ).cuda()
        linehaul_demand = (linehaul_demand.int() + 1).float()
        # Backhaul demand sampling
        backhaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_backhaul - 1, self.max_backhaul - 1
        ).cuda()
        backhaul_demand = (backhaul_demand.int() + 1).float()
        is_linehaul = torch.rand(*batch_size, num_loc, device='cuda') > self.backhaul_ratio
        backhaul_demand = (
            backhaul_demand * ~is_linehaul
        )  # keep only values where they are not linehauls
        linehaul_demand = linehaul_demand * is_linehaul
        return linehaul_demand, backhaul_demand

    def generate_time_windows(
        self,
        locs: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time windows (TW) and service times for each location including depot.
        We refer to the generation process in "Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization"
        (Liu et al., 2024). Note that another way to generate is from "Learning to Delegate for Large-scale Vehicle Routing" (Li et al, 2021) which
        is used in "MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024). Note that however, in that case
        the distance limit would have no influence when time windows are present, since the tw for depot is the same as distance with speed=1.
        This function can be overridden for that implementation.
        See also https://github.com/RoyalSkye/Routing-MVMoE

        Args:
            locs: [B, N+1, 2] (depot, locs)
            speed: [B]

        Returns:
            time_windows: [B, N+1, 2]
            service_time: [B, N+1]
        """
        device = locs.device
        batch_size, n_loc = locs.shape[0], locs.shape[1] - self.num_depots  # no depot

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc, device=device)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc, device=device)

        # Expand dimensions to make them compatible for broadcasting
        tensor_a = locs[:, 0 : self.num_depots]  # Shape: [B, num_depots, 2]
        tensor_b = locs[:, self.num_depots :]  # Shape: [B, N, 2]

        tensor_a_expanded = tensor_a[:, :, None, :]  # Shape: [B, num_depots, 1, 2]
        tensor_b_expanded = tensor_b[:, None, :, :]  # Shape: [B, 1, N, 2]

        d_0i = get_distance(
            tensor_a_expanded, tensor_b_expanded
        )  # Shape: [B, num_depots, N]

        d_0i, _ = torch.max(d_0i, dim=1)  # Shape: [B, N]

        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc, device=device)) * d_0i / speed
        tw_end = tw_start + tw_length

        # Depot tw is [0, max_time]
        time_windows = torch.stack(
            (
                torch.cat(
                    (torch.zeros(batch_size, self.num_depots, device=device), tw_start), dim=-1
                ),  # start
                torch.cat(
                    (torch.full((batch_size, self.num_depots), self.max_time, device=device), tw_end),
                    dim=-1,
                ),
            ),  # end
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat(
            (torch.zeros(batch_size, self.num_depots, device=device), service_time), dim=-1
        )
        return time_windows, service_time  # [B, num_depots + N, 2], [B, num_depots + N]

    def generate_distance_limit(
        self, shape: Tuple[int, int], locs: torch.Tensor
    ) -> torch.Tensor:
        """Generates distance limits (L).
        The distance lower bound is dist_lower_bound = 2 * max(depot_to_location_distance),
        then the max can be max_lim = min(max_distance_limit, dist_lower_bound + EPS). Ensures feasible yet challenging
        constraints, with each instance having a unique, meaningful limit

        Returns:
            distance_limit: [B, 1]
        """
        device = locs.device
        dist = torch.cdist(
            locs[:, 0 : self.num_depots], locs[:, self.num_depots :]
        )  # [B, num_depots, N]
        # we take the max per depot, but the min of depots' max is enough to ensure feasibility
        max_dist = dist.max(dim=-1)[0].min(dim=-1)[0]  # [B]

        dist_lower_bound = 2 * max_dist + 1e-6
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.max_distance_limit),
            dist_lower_bound + 1e-6,
        )

        # We need to sample from the `distribution` module to get the same distribution with a tensor as input
        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ]

    def generate_open_route(self, shape: Tuple[int, int]):
        """Generate open route flags (O). Here we could have a sampler but we simply return True here so all
        routes are open. Afterwards, we subsample the problems.
        """
        return torch.ones(shape, dtype=torch.bool, device='cuda')

    def generate_speed(self, shape: Tuple[int, int]):
        """We simply generate the speed as constant here"""
        # in this version, the speed is constant but this class may be overridden
        return torch.full(shape, self.speed, dtype=torch.float32, device='cuda')

    def generate_backhaul_class(self, shape: Tuple[int, int], sample: bool = False):
        """Generate backhaul class (B) for each node. If sample is True, we sample the backhaul class
        otherwise, we return the same class for all nodes.
        - Backhaul class 1: classic backhaul (VRPB), linehauls must be served before backhauls in a route (every customer is either, not both)
        - Backhaul class 2: mixed backhaul (VRPMPD or VRPMB), linehauls and backhauls can be served in any order (every customer is either, not both)
        """
        if sample:
            return torch.randint(1, 3, shape, dtype=torch.float32, device='cuda')
        else:
            return torch.full(shape, self.backhaul_class, dtype=torch.float32, device='cuda')
    
    @staticmethod
    def save_data(td: TensorDict, path: str, compress: bool = False):
        save_tensordict_to_npz(td, path)
    
    @staticmethod
    def print_presets():
        for key, value in VARIANT_GENERATION_PRESETS.items():
            print(f"{key}: {value}")
    
    @staticmethod
    def available_variants(*args, **kwargs):
        return list(VARIANT_GENERATION_PRESETS.keys())[2:]  # Skip 'all' and 'single_feat'
