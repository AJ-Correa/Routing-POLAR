"""
Single-Depot Multi-Task VRP Generator

Supports all 24 single-depot VRP variants:
- Original 16: CVRP, OVRP, VRPB, VRPL, VRPTW + combinations (backhaul_class=1)
- New 8 mixed backhaul: VRPMB, OVRPMB, etc. (backhaul_class=2)

The generator creates instances with all features enabled, then subsamples
based on the variant_preset to create specific problem types.
"""

import torch
import logging
from typing import Callable, Tuple, Union
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from utils.functions import get_distance, save_tensordict_to_npz

log = logging.getLogger(__name__)


def get_vehicle_capacity(num_loc: int) -> int:
    """Capacity should be 30 + num_loc/5 if num_loc > 20 as described in Liu et al. 2024 (POMO-MTL).
    For every N over 1000, we add 1 of capacity every 33.3 nodes to align with Ye et al. 2024 (GLOP).
    """
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


# Variant generation presets for single-depot VRP
# Keys: O=Open, TW=TimeWindow, L=DistanceLimit, B=Backhaul
VARIANT_GENERATION_PRESETS = {
    # Training presets
    "all": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},  # Original 16 variants
    "single_feat": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
    "cvrp": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0},
    "mixed_backhaul": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 1.0},
    
    # Standard backhaul variants (backhaul_class=1)
    "ovrp": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0},
    "vrpb": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "vrpl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0},
    "vrptw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0},
    "ovrptw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0},
    "ovrpb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "ovrpl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0},
    "vrpbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "vrpbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "vrpltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0},
    "ovrpbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "ovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "ovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0},
    "vrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    "ovrpbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    
    # Mixed backhaul variants (backhaul_class=2) - NEW
    "vrpmb": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "ovrpmb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "vrpmbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "vrpmbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "ovrpmbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "ovrpmbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "vrpmbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    "ovrpmbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
}

# Mixed backhaul variants list
MIXED_BACKHAUL_VARIANTS = [
    "vrpmb", "ovrpmb", "vrpmbl", "vrpmbtw",
    "ovrpmbl", "ovrpmbtw", "vrpmbltw", "ovrpmbltw"
]


class MTVRPGenerator:
    """Single-Depot MTVRP Generator.
    
    Generates instances for all 24 single-depot VRP variants:
    - 16 standard variants with classical backhaul (backhaul_class=1)
    - 8 mixed backhaul variants (backhaul_class=2)
    
    Args:
        num_loc: Number of customer locations
        min_loc: Minimum location coordinate
        max_loc: Maximum location coordinate
        loc_distribution: Distribution for sampling locations
        capacity: Vehicle capacity (None = auto-compute)
        min_demand: Minimum demand value
        max_demand: Maximum demand value
        scale_demand: Whether to scale demands by capacity
        max_time: Maximum time window value
        backhaul_ratio: Fraction of nodes that are backhaul
        backhaul_class: 1=classical, 2=mixed
        sample_backhaul_class: If True, randomly sample backhaul class
        max_distance_limit: Maximum distance limit
        speed: Vehicle speed
        variant_preset: Preset for variant probabilities
        use_combinations: Allow multiple features per instance
        subsample: Whether to subsample features
    """
    
    def __init__(
        self,
        num_loc: int = 20,
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
            if variant_preset in MIXED_BACKHAUL_VARIANTS:
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
    
    def reset_n_loc(self, num_loc: int):
        """Reset number of locations and update capacity accordingly."""
        self.num_loc = num_loc
        self.capacity = get_vehicle_capacity(num_loc)
    
    def reset_variant_preset(self, variant_preset: str):
        """Reset the variant preset."""
        self.variant_preset = variant_preset
        variant_probs = VARIANT_GENERATION_PRESETS.get(variant_preset)
        self.variant_probs = variant_probs
        if variant_preset in MIXED_BACKHAUL_VARIANTS:
            self.backhaul_class = 2
        else:
            self.backhaul_class = 1
    
    def __call__(self, batch_size) -> TensorDict:
        """Generate a batch of MTVRP instances."""
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        
        # Locations: depot + customers
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2)).to('cuda')
        depot = torch.FloatTensor(*batch_size, 1, 2).to('cuda').uniform_(self.min_loc, self.max_loc)
        locs = torch.cat((depot, locs), dim=-2)
        
        # Vehicle capacity
        vehicle_capacity = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)
        capacity_original = vehicle_capacity.clone()
        
        # Demands
        demand_linehaul, demand_backhaul = self.generate_demands(batch_size, self.num_loc)
        demand_linehaul = torch.cat([torch.zeros(size=(*batch_size, 1)), demand_linehaul], dim=1)
        demand_backhaul = torch.cat([torch.zeros(size=(*batch_size, 1)), demand_backhaul], dim=1)
        
        # Backhaul class
        backhaul_class = self.generate_backhaul_class((*batch_size, 1), sample=self.sample_backhaul_class)
        
        # Open route
        open_route = self.generate_open_route((*batch_size, 1))
        
        # Time windows
        speed = self.generate_speed((*batch_size, 1))
        time_windows, service_time = self.generate_time_windows(locs, speed)
        
        # Distance limit
        distance_limit = self.generate_distance_limit((*batch_size, 1), locs)
        
        # Scale demands
        if self.scale_demand:
            demand_backhaul = demand_backhaul / vehicle_capacity
            demand_linehaul = demand_linehaul / vehicle_capacity
            vehicle_capacity = vehicle_capacity / vehicle_capacity
        
        td = TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "time_windows": time_windows,
                "service_time": service_time,
                "vehicle_capacity": vehicle_capacity,
                "capacity_original": capacity_original,
                "open_route": open_route,
                "speed": speed,
            },
            batch_size=batch_size,
        )
        
        if self.subsample:
            return self.subsample_problems(td)
        return td
    
    def subsample_problems(self, td: TensorDict) -> TensorDict:
        """Subsample problems based on variant preset probabilities."""
        batch_size = td.batch_size[0]
        variant_probs = torch.tensor(list(self.variant_probs.values()))
        
        if self.use_combinations:
            keep_mask = torch.rand(batch_size, 4) <= variant_probs  # O, TW, L, B
        else:
            if self.variant_preset in ("all", "cvrp", "single_feat"):
                cvrp_prob = 0.5
            else:
                cvrp_prob = 0
            
            if self.variant_preset in ("all", "cvrp", "single_feat"):
                indices = torch.distributions.Categorical(
                    torch.Tensor(list(self.variant_probs.values()) + [cvrp_prob])[None].repeat(batch_size, 1)
                ).sample()
                keep_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
                keep_mask[torch.arange(batch_size), indices] = True
            else:
                keep_mask = torch.zeros((batch_size, 4), dtype=torch.bool)
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
        is_multi_depot = torch.zeros(batch_size, dtype=torch.bool)

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
        td["demand_linehaul"][remove] = td["demand_linehaul"][remove] + td["demand_backhaul"][remove]
        td["demand_backhaul"][remove] = 0
        return td
    
    def generate_demands(self, batch_size, num_loc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate linehaul and backhaul demands."""
        linehaul_demand = (
            torch.FloatTensor(*batch_size, num_loc).to('cuda')
            .uniform_(self.min_demand - 1, self.max_demand - 1).int() + 1
        ).float()
        
        backhaul_demand = (
            torch.FloatTensor(*batch_size, num_loc).to('cuda')
            .uniform_(self.min_backhaul - 1, self.max_backhaul - 1).int() + 1
        ).float()
        
        is_linehaul = torch.rand(*batch_size, num_loc) > self.backhaul_ratio
        backhaul_demand = backhaul_demand * ~is_linehaul
        linehaul_demand = linehaul_demand * is_linehaul
        
        return linehaul_demand, backhaul_demand
    
    def generate_time_windows(self, locs: torch.Tensor, speed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate time windows and service times."""
        batch_size, n_loc = locs.shape[0], locs.shape[1] - 1
        
        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)
        d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i / speed
        tw_end = tw_start + tw_length
        
        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),
            dim=-1,
        )
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)
        
        return time_windows, service_time
    
    def generate_distance_limit(self, shape: Tuple[int, ...], locs: torch.Tensor) -> torch.Tensor:
        """Generate distance limits."""
        max_dist = torch.max(torch.cdist(locs[:, 0:1], locs[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist + 1e-6
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.max_distance_limit),
            dist_lower_bound + 1e-6,
        )
        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[..., None]
    
    def generate_open_route(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate open route flags."""
        return torch.ones(shape, dtype=torch.bool)
    
    def generate_speed(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate speed values."""
        return torch.full(shape, self.speed, dtype=torch.float32)
    
    def generate_backhaul_class(self, shape: Tuple[int, ...], sample: bool = False) -> torch.Tensor:
        """Generate backhaul class (1=classical, 2=mixed)."""
        if sample:
            return torch.randint(1, 3, shape, dtype=torch.float32)
        return torch.full(shape, self.backhaul_class, dtype=torch.float32)
    
    @staticmethod
    def save_data(td: TensorDict, path: str, compress: bool = False):
        save_tensordict_to_npz(td, path)
    
    @staticmethod
    def print_presets():
        for key, value in VARIANT_GENERATION_PRESETS.items():
            print(f"{key}: {value}")
    
    @staticmethod
    def available_variants(*args, **kwargs):
        return list(VARIANT_GENERATION_PRESETS.keys())[3:]
