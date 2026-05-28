import time

import pyvrp
from pyvrp import SolveParams
from pyvrp.PenaltyManager import PenaltyManager
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from pyvrp._pyvrp import Route
from pyvrp._pyvrp import (
    RandomNumberGenerator,
    Solution,
)
from pyvrp.search import (
    LocalSearch,
    NeighbourhoodParams,
    compute_neighbours,
)
from pyvrp.crossover import selective_route_exchange as srex
from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve
import numpy as np
from scipy.spatial.distance import cdist
from .vrplib_helpers import compute_vrplib_cost_matrix, resolve_round_func

_BOOSTER_SCHEDULE = (12, 200, 3_000, 50_000, 1_000_000)

from .cython_heuristics.heuristics import (
    concentric_removal,
    insertion_by_cost,
    insertion_by_distance,
    random_removal,
    sequence_removal,
)


def compute_cost_matrix(
    locs,
    demands_linehauls,
    demands_backhauls,
    open_route,
    max_distance=None,
    num_depots=1,
    mixed_backhaul=False,
):
    """Vectorized cost matrix computation with backhaul -> linehaul constraint."""
    matrix = cdist(locs, locs, metric='euclidean')

    if open_route:
        matrix[:, :num_depots] = 0

    if np.any(demands_backhauls) and not mixed_backhaul:
        linehaul_mask = demands_linehauls > 0
        backhaul_mask = demands_backhauls > 0
        matrix[np.ix_(backhaul_mask, linehaul_mask)] = 1 << 32

    return matrix


class PerturbationHeuristics:
    """AILS-II style removal and addition heuristics for solution perturbation."""

    def __init__(self, data, cost_matrix, rng):
        self.data = data
        self.cost_matrix = np.ascontiguousarray(cost_matrix, dtype=np.float64)
        self.rng = rng
        self.num_depots = len(data.depots())

    def random_removal(self, tour, omega):
        """Python wrapper → C extension. Uses actual RNG."""
        tour_arr = np.asarray(tour, dtype=np.int64)

        # Count customers to generate proper random permutation
        n_customers = sum(1 for node in tour if node >= self.num_depots)
        if n_customers <= omega:
            return tour, []

        # Generate random permutation using actual numpy RNG
        rng_indices = self.rng.permutation(n_customers).astype(np.int64)

        return random_removal(tour_arr, omega, self.num_depots, rng_indices)

    def insertion_by_cost(self, partial_tour, removed_vertices):
        """Python wrapper → C extension. Returns list like Python."""
        tour_arr = np.asarray(partial_tour, dtype=np.int64)
        removed_arr = np.asarray(removed_vertices, dtype=np.int64)
        return insertion_by_cost(tour_arr, removed_arr, self.cost_matrix, self.num_depots)

    def insertion_by_distance(self, partial_tour, removed_vertices):
        """Insert removed vertices closest to their nearest neighbor in tour."""
        tour_arr = np.asarray(partial_tour, dtype=np.int64)
        removed_arr = np.asarray(removed_vertices, dtype=np.int64)
        return insertion_by_distance(tour_arr, removed_arr, self.cost_matrix, self.num_depots)

    def concentric_removal(self, tour, omega):
        """Remove omega vertices closest to a randomly chosen seed vertex."""
        tour_arr = np.asarray(tour, dtype=np.int64)
        n_customers = int(np.sum(tour_arr >= self.num_depots))
        if n_customers <= omega:
            return tour, []

        seed_cust_idx = int(self.rng.randint(0, n_customers))
        new_tour, removed, status = concentric_removal(
            tour_arr,
            omega,
            self.num_depots,
            self.cost_matrix,
            seed_cust_idx,
        )
        if status != 0:
            return self.random_removal(tour, omega)
        return new_tour, removed

    def sequence_removal(self, tour, omega):
        """Remove omega consecutive vertices from a random starting point."""
        tour_arr = np.asarray(tour, dtype=np.int64)
        n_customers = int(np.sum(tour_arr >= self.num_depots))
        if n_customers <= omega:
            return tour, []

        rng_choice = int(self.rng.randint(0, 2**31 - 1))
        new_tour, removed, status = sequence_removal(
            tour_arr, omega, self.num_depots, rng_choice
        )
        if status != 0:
            return self.random_removal(tour, omega)
        return new_tour, removed

    def perturb(self, tour, omega):
        """Remove omega vertices then reinsert (concentric/sequence + cost/distance)."""
        tour = list(tour)
        if self.rng.random() < 0.5:
            partial_tour, removed = self.concentric_removal(tour, omega)
        else:
            partial_tour, removed = self.sequence_removal(tour, omega)

        if len(removed) == 0:
            return tour

        if self.rng.random() < 0.5:
            return self.insertion_by_distance(partial_tour, removed)
        return self.insertion_by_cost(partial_tour, removed)


class Search:
    def __init__(
        self,
        locs,
        demands_linehauls,
        demands_backhauls,
        distance_limit,
        open_route,
        time_windows,
        service_times,
        num_depots=1,
        mixed_backhaul=False,
        nb_granular=20,
        metric: Literal["normalized", "vrplib"] = "normalized",
        round_func: Union[str, Callable[[np.ndarray], np.ndarray]] = "round",
        vehicle_capacity: Optional[int] = None,
        edge_weight: Optional[np.ndarray] = None,
        vrplib_options: Optional[dict] = None,
    ):
        """
        Construct a PyVRP local-search model.

        metric="normalized" (default)
            Coordinates in [0, 1], fractional demands, capacity 1000 — training / synthetic.
        metric="vrplib"
            Integer CVRPLIB geometry and costs (same rounding as ``pyvrp.read``).
            Pass raw ``node_coord``, integer ``demand`` (depot included), and file capacity.
            Alternatively pass *vrplib_options* with keys
            ``coords``, ``demands``, ``capacity``, optional ``round_func``, ``edge_weight``.
        """
        if vrplib_options is not None:
            metric = "vrplib"
            locs = vrplib_options["coords"]
            demands_linehauls = vrplib_options["demands"]
            round_func = vrplib_options.get("round_func", round_func)
            vehicle_capacity = vrplib_options["capacity"]
            edge_weight = vrplib_options.get("edge_weight", edge_weight)

        self.metric = metric
        self.num_depots = num_depots
        self.num_customers = len(locs) - self.num_depots
        self.eps = 1e-30

        if metric == "vrplib":
            if vehicle_capacity is None:
                raise ValueError("vehicle_capacity is required when metric='vrplib'.")
            self._init_vrplib_problem(
                locs=locs,
                demands_linehauls=demands_linehauls,
                demands_backhauls=demands_backhauls,
                distance_limit=distance_limit,
                open_route=open_route,
                time_windows=time_windows,
                service_times=service_times,
                vehicle_capacity=vehicle_capacity,
                round_func=round_func,
                edge_weight=edge_weight,
                mixed_backhaul=mixed_backhaul,
                nb_granular=nb_granular,
            )
        else:
            self._init_normalized_problem(
                locs=locs,
                demands_linehauls=demands_linehauls,
                demands_backhauls=demands_backhauls,
                distance_limit=distance_limit,
                open_route=open_route,
                time_windows=time_windows,
                service_times=service_times,
                mixed_backhaul=mixed_backhaul,
                nb_granular=nb_granular,
            )

    def _init_normalized_problem(
        self,
        locs,
        demands_linehauls,
        demands_backhauls,
        distance_limit,
        open_route,
        time_windows,
        service_times,
        mixed_backhaul,
        nb_granular,
    ):
        """Training / synthetic instances: coords in [0,1], internal scale 1000."""
        self.scaler = 1_000
        nd = self.num_depots
        nc = self.num_customers

        locs_s = np.asarray(locs) * self.scaler
        dlin_s = np.asarray(demands_linehauls) * self.scaler
        dbac_s = np.asarray(demands_backhauls) * self.scaler
        svc_s = np.asarray(service_times) * self.scaler
        tw_arr = np.asarray(time_windows)

        cost_matrix = np.ascontiguousarray(
            compute_cost_matrix(
                np.asarray(locs),
                np.asarray(demands_linehauls),
                np.asarray(demands_backhauls),
                open_route=open_route,
                num_depots=nd,
                mixed_backhaul=mixed_backhaul,
            ),
        )
        cost_matrix = np.round(cost_matrix * self.scaler)

        depots, clients, vehicle_types, tw_finite = self._build_nodes_and_vehicles(
            xs_i=locs_s[:, 0],
            ys_i=locs_s[:, 1],
            dlin_i=np.round(dlin_s),
            dbac_i=np.round(dbac_s),
            svc_i=svc_s + self.eps * self.scaler,
            tw_arr=tw_arr,
            distance_limit=distance_limit,
            capacity=int(round((1 - self.eps) * self.scaler)),
        )
        self._finish_init(cost_matrix, depots, clients, vehicle_types, nb_granular)

    def _init_vrplib_problem(
        self,
        locs,
        demands_linehauls,
        demands_backhauls,
        distance_limit,
        open_route,
        time_windows,
        service_times,
        vehicle_capacity,
        round_func,
        edge_weight,
        mixed_backhaul,
        nb_granular,
    ):
        """CVRPLIB / benchmark instances: integer costs aligned with pyvrp.read."""
        self.scaler = 1
        rf = resolve_round_func(round_func)

        cost_matrix, coords = compute_vrplib_cost_matrix(
            locs,
            round_func=round_func,
            edge_weight=edge_weight,
        )

        dlin_i = rf(np.asarray(demands_linehauls, dtype=np.float64))
        dbac_i = np.zeros_like(dlin_i)  # CVRPLIB path is CVRP-only.
        svc_i = rf(np.asarray(service_times, dtype=np.float64))
        tw_arr = np.asarray(time_windows, dtype=np.float64)

        cap_i = int(rf(np.atleast_1d(vehicle_capacity))[0])
        depots, clients, vehicle_types, _ = self._build_nodes_and_vehicles(
            xs_i=coords[:, 0],
            ys_i=coords[:, 1],
            dlin_i=dlin_i,
            dbac_i=dbac_i,
            svc_i=svc_i,
            tw_arr=tw_arr,
            distance_limit=distance_limit,
            capacity=cap_i,
            scale_coords=False,
        )
        self._finish_init(cost_matrix, depots, clients, vehicle_types, nb_granular)

    def _build_nodes_and_vehicles(
        self,
        xs_i,
        ys_i,
        dlin_i,
        dbac_i,
        svc_i,
        tw_arr,
        distance_limit,
        capacity: int,
        scale_coords: bool = True,
    ):
        nd = self.num_depots
        nc = self.num_customers

        tw_early_all = tw_arr[:, 0]
        tw_late_all = tw_arr[:, 1]
        tw_finite = np.isfinite(tw_early_all) & np.isfinite(tw_late_all)

        if scale_coords:
            _safe_early = np.where(tw_finite, tw_early_all, 0.0)
            _safe_late = np.where(tw_finite, tw_late_all, 0.0)
            tw_early_i = (_safe_early + self.eps) * self.scaler
            tw_late_i = (_safe_late - self.eps) * self.scaler
            svc_dur = svc_i
        else:
            rf = resolve_round_func("round")
            _safe_early = np.where(tw_finite, tw_early_all, 0.0)
            _safe_late = np.where(tw_finite, tw_late_all, 0.0)
            tw_early_i = rf(_safe_early)
            tw_late_i = rf(_safe_late)
            svc_dur = rf(np.asarray(svc_i))

        depots = [Depot(x=int(xs_i[i]), y=int(ys_i[i])) for i in range(nd)]

        clients = []
        for i in range(nd, nd + nc):
            if tw_finite[i]:
                clients.append(
                    Client(
                        x=int(xs_i[i]),
                        y=int(ys_i[i]),
                        delivery=[int(dlin_i[i])],
                        pickup=[int(dbac_i[i])],
                        service_duration=int(svc_dur[i]),
                        tw_early=int(tw_early_i[i]),
                        tw_late=int(tw_late_i[i]),
                    )
                )
            else:
                clients.append(
                    Client(
                        x=int(xs_i[i]),
                        y=int(ys_i[i]),
                        delivery=[int(dlin_i[i])],
                        pickup=[int(dbac_i[i])],
                        service_duration=int(svc_dur[i]),
                    )
                )

        max_dist_i = None
        if np.isfinite(distance_limit):
            max_dist_i = (
                int((distance_limit - self.eps) * self.scaler)
                if scale_coords
                else int(resolve_round_func("round")(np.atleast_1d(distance_limit))[0])
            )

        vehicle_types = []
        for i in range(nd):
            vkw = dict(num_available=nc, start_depot=i, end_depot=i, capacity=[capacity])
            if max_dist_i is not None:
                vkw["max_distance"] = max_dist_i
            if tw_finite[i]:
                vkw["tw_early"] = int(tw_early_i[i])
                vkw["tw_late"] = int(tw_late_i[i])
            vehicle_types.append(VehicleType(**vkw))

        return depots, clients, vehicle_types, tw_finite

    def _finish_init(self, cost_matrix, depots, clients, vehicle_types, nb_granular):
        self._data = ProblemData(
            clients, depots, vehicle_types, [cost_matrix], [cost_matrix]
        )
        self.model = None
        self._cost_matrix = cost_matrix.astype(np.float64) / self.scaler

        self.rng = RandomNumberGenerator(seed=0)
        self.params = SolveParams()
        nb_params = NeighbourhoodParams(nb_granular=nb_granular)
        neighbours = compute_neighbours(self._data, nb_params)
        ls = LocalSearch(self._data, self.rng, neighbours)

        for node_op in self.params.node_ops:
            ls.add_node_operator(node_op(self._data))
        for route_op in self.params.route_ops:
            ls.add_route_operator(route_op(self._data))

        self._pm = PenaltyManager.init_from(self._data, self.params.penalty)
        self._neighbours = neighbours
        self._search = ls

    @classmethod
    def from_vrplib_instance(
        cls,
        instance: dict,
        round_func: Union[str, Callable[[np.ndarray], np.ndarray]] = "round",
        nb_granular: int = 20,
        num_depots: int = 1,
    ) -> "Search":
        """
        Build Search from a ``vrplib.read_instance`` dict (same metric as ``pyvrp.read``).
        """
        coords = instance["node_coord"]
        demands = instance.get("demand", instance.get("linehaul"))
        capacity = instance["capacity"]
        n = len(coords)
        tw = np.stack(
            [np.zeros(n), np.full(n, np.inf)],
            axis=1,
            dtype=np.float64,
        )
        return cls(
            coords,
            demands,
            np.zeros_like(demands, dtype=np.float64),
            np.inf,
            False,
            tw,
            np.zeros(n, dtype=np.float64),
            num_depots=num_depots,
            nb_granular=nb_granular,
            metric="vrplib",
            round_func=round_func,
            vehicle_capacity=capacity,
            edge_weight=instance.get("edge_weight"),
        )

    def tour_cost(self, tour) -> float:
        """Total distance of *tour* in external units (CVRPLIB integers or normalized)."""
        sol = self._tour_to_solution(tour)
        if not sol.is_feasible():
            return float("inf")
        return sol.distance() / self.scaler

    def _make_search(self, seed: int) -> LocalSearch:
        """Create a new LocalSearch instance with the given seed."""
        rng = RandomNumberGenerator(seed=seed)
        ls = LocalSearch(self._data, rng, self._neighbours)
        for node_op in self.params.node_ops:
            ls.add_node_operator(node_op(self._data))
        for route_op in self.params.route_ops:
            ls.add_route_operator(route_op(self._data))
        return ls

    def _tour_to_solution(self, tour):
        """Convert tour to pyvrp Solution. May raise if invalid."""
        tour_arr = np.asarray(tour, dtype=np.intp)
        depot_positions = np.flatnonzero(tour_arr < self.num_depots)

        routes = []
        if depot_positions.size == 0:
            customers = tour_arr.tolist()
            if customers:
                routes.append([0] + customers)
        else:
            starts = depot_positions
            ends = np.empty_like(starts)
            ends[:-1] = starts[1:]
            ends[-1] = len(tour_arr)
            for s, e in zip(starts.tolist(), ends.tolist()):
                seg = tour_arr[s:e]
                if len(seg) > 1:
                    routes.append(seg.tolist())
            if starts[0] > 0:
                pre = tour_arr[: starts[0]].tolist()
                if pre:
                    first_depot = int(tour_arr[starts[0]])
                    routes.insert(0, [first_depot] + pre)

        ls_routes = [Route(self._data, r[1:], r[0]) for r in routes]
        return Solution(self._data, ls_routes)

    def _extract_tour(self, sol):
        """Constructs LS-improved solution to match the format expected by the model."""
        tour = []
        if self.num_depots == 1:
            for route in sol.routes():
                tour.extend(route)
                tour.append(0)
        else:
            for route in sol.routes():
                tour.append(route.start_depot())
                tour.extend(route)
        return tour

    @staticmethod
    def _tour_edges(tour):
        if tour is None or len(tour) < 2:
            return set()
        return {(tour[i], tour[i + 1]) for i in range(len(tour) - 1)}

    @staticmethod
    def _edge_hamming(tour_a, tour_b):
        ea = Search._tour_edges(tour_a)
        eb = Search._tour_edges(tour_b)
        return len(ea.symmetric_difference(eb))

    def build_solution(self, tour, seed: int = 0):
        """Build VRP routes from the sequence of nodes generated by the model,
        and proceeds to call the LS-improvement loop.

        Parameters
        ----------
        tour:
            Node sequence produced by the model.
        seed:
            RNG seed for this specific search call.  Pass a unique value per
            (batch_idx, pomo_idx) pair to get diverse LS trajectories.
        """
        self._search = self._make_search(seed)
        num_depots = self.num_depots

        tour_arr = np.asarray(tour, dtype=np.intp)
        depot_positions = np.flatnonzero(tour_arr < num_depots)

        routes = []
        if depot_positions.size == 0:
            customers = tour_arr.tolist()
            if customers:
                routes.append([0] + customers)
        else:
            starts = depot_positions
            ends = np.empty_like(starts)
            ends[:-1] = starts[1:]
            ends[-1] = len(tour_arr)
            for s, e in zip(starts.tolist(), ends.tolist()):
                seg = tour_arr[s:e]
                if len(seg) > 1:
                    routes.append(seg.tolist())
            if starts[0] > 0:
                pre = tour_arr[: starts[0]].tolist()
                if pre:
                    first_depot = int(tour_arr[starts[0]])
                    routes.insert(0, [first_depot] + pre)

        ls_routes = [Route(self._data, r[1:], r[0]) for r in routes]
        solution = Solution(self._data, ls_routes)
        self.params.penalty.repair_booster = 12
        return self.run(solution)

    def run(self, solution, search_instance=None):
        """Main LS-improvement loop with given search instance."""
        if search_instance is None:
            search_instance = self._search

        if solution.is_feasible():
            base_distance = solution.distance()
        else:
            base_distance = float("inf")

        for booster in _BOOSTER_SCHEDULE:
            self._pm._params.repair_booster = booster
            sol = search_instance(solution, self._pm.booster_cost_evaluator())
            if sol.is_feasible() and sol.distance() < base_distance:
                return sol.distance() / self.scaler, self._extract_tour(sol)

        return base_distance / self.scaler, self._extract_tour(solution)

    def iterated_perturbation_search(
        self,
        tour,
        seed: int = 0,
        num_iters: int = 5,
        time_limit: Optional[float] = None,
        dmax: int = 10,
        dmin: int = 5,
        acceptance_rate: float = 0.01,
        no_improvement: int = 8,
    ):
        """
        ILS-style: iterate perturbation → LS until num_iters or time_limit is reached.
        Always returns the best feasible solution found.

        When *time_limit* is set (seconds), only the perturbation loop is timed; the
        initial LS on the input tour is excluded. *num_iters* is ignored in that case.

        dmax/dmin schedule omega over iteration index or elapsed-time fraction.
        acceptance_rate: accept if cost <= current_cost * (1 + acceptance_rate).
        no_improvement: reset current tour to best after this many rejections.
        """
        rng = np.random.RandomState(seed)
        perturbation = PerturbationHeuristics(self._data, self._cost_matrix, rng)
        dmax = max(1, int(dmax))
        dmin = max(1, int(dmin))
        if dmin > dmax:
            dmin = dmax
        no_improvement = max(1, int(no_improvement))
        ratio = dmin / dmax

        sol = self._tour_to_solution(tour)
        search = self._make_search(seed)
        base_cost, base_tour = self.run(sol, search_instance=search)

        best_cost = base_cost
        best_tour = base_tour
        current_cost = base_cost
        current_tour = base_tour
        no_improvement_count = 0

        timed = time_limit is not None and time_limit > 0.0
        ils_start = time.perf_counter()
        ils_deadline = ils_start + time_limit if timed else None
        it = 0

        while True:
            if timed:
                now = time.perf_counter()
                if now >= ils_deadline:
                    break
                progress = min(1.0, (now - ils_start) / time_limit)
            else:
                if it >= num_iters:
                    break
                progress = it / max(1, num_iters - 1)

            current_omega = max(dmin, int(round(dmax * (ratio**progress))))

            perturbed = perturbation.perturb(current_tour, omega=current_omega)

            try:
                sol = self._tour_to_solution(perturbed)
                cost, improved_tour = self.run(sol, search_instance=search)

                if cost < best_cost:
                    best_cost = cost
                    best_tour = improved_tour

                accept_limit = current_cost * (1.0 + acceptance_rate)
                if cost <= accept_limit:
                    current_cost = cost
                    current_tour = improved_tour
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= no_improvement:
                        current_cost = best_cost
                        current_tour = best_tour
                        no_improvement_count = 0

            except Exception:
                no_improvement_count += 1

            it += 1

        return best_cost, best_tour


def _ls_instance_iterated(
    instance_args,
    tour,
    nb_granular,
    seed,
    num_iters=5,
    time_limit=None,
    dmax=10,
    dmin=5,
    acceptance_rate=0.01,
    no_improvement=8,
    vrplib_options: Optional[dict] = None,
    mixed_backhaul: bool = False,
):
    """
    Iterated perturbation + LS on one instance.
    Returns (cost, improved_tour) where improved_tour is guaranteed feasible.

    When *vrplib_options* is set, LS uses CVRPLIB integer costs (see Search metric='vrplib').
    """
    (
        locs,
        demands_linehaul,
        demands_backhaul,
        distance_limit,
        open_route,
        time_windows,
        service_time,
        num_depots,
    ) = instance_args
    search = Search(
        locs,
        demands_linehaul,
        demands_backhaul,
        distance_limit,
        open_route,
        time_windows,
        service_time,
        num_depots=num_depots,
        mixed_backhaul=mixed_backhaul,
        nb_granular=nb_granular,
        vrplib_options=vrplib_options,
    )
    cost, improved_tour = search.iterated_perturbation_search(
        tour,
        seed=seed,
        num_iters=num_iters,
        time_limit=time_limit,
        dmax=dmax,
        dmin=dmin,
        acceptance_rate=acceptance_rate,
        no_improvement=no_improvement,
    )
    return cost, improved_tour

