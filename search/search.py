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
    NODE_OPERATORS,
    ROUTE_OPERATORS,
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

# Internal PyVRP integer scale for normalized [0, 1] instances.
POLAR_SCALER = 10_000_000  # training / tuning (POLAR local search)
NAILS_SCALER = 1_000       # inference (NAILS local search)

# Default NAILS local-search operator set (PyVRP minus low-impact / slow operators).
NAILS_EXCLUDED_OPERATORS = (
    "Exchange30",
    "Exchange31",
    "Exchange32",
    "Exchange33",
    "TripRelocate",
    "SwapRoutes",
)


def build_solve_params(
    exclude_operators: Optional[List[str]] = None,
    *,
    use_all_operators: bool = False,
) -> SolveParams:
    """
    Build PyVRP ``SolveParams`` for NAILS local search.

    By default, ``NAILS_EXCLUDED_OPERATORS`` are removed. Pass
    ``use_all_operators=True`` for the full PyVRP operator set, or extend
    ``exclude_operators`` with additional operator class names.
    """
    if use_all_operators:
        excluded = set(exclude_operators or [])
    else:
        excluded = set(NAILS_EXCLUDED_OPERATORS)
        if exclude_operators:
            excluded.update(exclude_operators)

    node_ops = [op for op in NODE_OPERATORS if op.__name__ not in excluded]
    route_ops = [op for op in ROUTE_OPERATORS if op.__name__ not in excluded]
    if not node_ops:
        raise ValueError("At least one node operator must remain active.")
    if not route_ops:
        raise ValueError("At least one route operator must remain active.")
    return SolveParams(node_ops=node_ops, route_ops=route_ops)


from .perturbation import PerturbationHeuristics


def compute_cost_matrix(
    locs,
    demands_linehauls,
    demands_backhauls,
    open_route,
    max_distance=None,
    num_depots=1,
    mixed_backhaul=False,
    *,
    coords_scaled: bool = False,
):
    """
    Euclidean distance matrix with optional VRPB backhaul masking.

    ``coords_scaled=False`` (NAILS): ``cdist`` on normalized [0, 1] coords; caller
    scales with ``round(matrix * scaler)``.

    ``coords_scaled=True`` (POLAR): distances on coords already multiplied by
    ``POLAR_SCALER`` (no extra ``* scaler``).
    """
    locs = np.asarray(locs, dtype=np.float64)
    if coords_scaled:
        sq = np.sum(locs * locs, axis=1)
        matrix = sq[:, None] + sq[None, :] - 2.0 * locs @ locs.T
        np.maximum(matrix, 0, out=matrix)
        np.sqrt(matrix, out=matrix)
    else:
        matrix = cdist(locs, locs, metric="euclidean")

    if open_route:
        matrix[:, :num_depots] = 0

    if np.any(demands_backhauls) and not mixed_backhaul:
        linehaul_mask = demands_linehauls > 0
        backhaul_mask = demands_backhauls > 0
        matrix[np.ix_(backhaul_mask, linehaul_mask)] = 1 << 32

    return matrix


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
        scaler: int = NAILS_SCALER,
        exclude_operators: Optional[List[str]] = None,
        use_all_operators: bool = True,
        metric: Literal["normalized", "vrplib"] = "normalized",
        round_func: Union[str, Callable[[np.ndarray], np.ndarray]] = "round",
        vehicle_capacity: Optional[int] = None,
        edge_weight: Optional[np.ndarray] = None,
        vrplib_options: Optional[dict] = None,
    ):
        """
        Construct a PyVRP local-search model for NAILS.

        Local search uses the full PyVRP operator set by default (training LS).
        Pass ``use_all_operators=False`` for the reduced NAILS test-time set
        (see ``NAILS_EXCLUDED_OPERATORS``), or pass ``exclude_operators`` to
        drop more.

        metric="normalized" (default)
            Coordinates in [0, 1], fractional demands. ``scaler`` sets the internal
            PyVRP integer scale (``POLAR_SCALER`` for training/tuning, ``NAILS_SCALER`` for NAILS).
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
        self._exclude_operators = exclude_operators
        self._use_all_operators = use_all_operators

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
                scaler=scaler,
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
        scaler,
    ):
        """Synthetic instances in [0, 1]. NAILS vs POLAR differ in cost-matrix construction."""
        self.scaler = scaler
        S = self.scaler
        nd = self.num_depots
        nc = self.num_customers
        polar = scaler == POLAR_SCALER

        locs_u = np.asarray(locs, dtype=np.float64)
        dlin_u = np.asarray(demands_linehauls, dtype=np.float64)
        dbac_u = np.asarray(demands_backhauls, dtype=np.float64)
        locs_s = locs_u * S
        dlin_s = dlin_u * S
        dbac_s = dbac_u * S
        svc_s = np.asarray(service_times, dtype=np.float64) * S
        tw_arr = np.asarray(time_windows, dtype=np.float64)

        if polar:
            cost_matrix = np.ascontiguousarray(
                compute_cost_matrix(
                    locs_s,
                    dlin_s,
                    dbac_s,
                    open_route=open_route,
                    num_depots=nd,
                    mixed_backhaul=mixed_backhaul,
                    coords_scaled=True,
                ),
                dtype=np.float64,
            )
        else:
            cost_matrix = np.ascontiguousarray(
                np.round(
                    compute_cost_matrix(
                        locs_u,
                        dlin_u,
                        dbac_u,
                        open_route=open_route,
                        num_depots=nd,
                        mixed_backhaul=mixed_backhaul,
                        coords_scaled=False,
                    )
                    * S
                ),
                dtype=np.float64,
            )

        depots, clients, vehicle_types, tw_finite = self._build_nodes_and_vehicles(
            xs_i=locs_s[:, 0],
            ys_i=locs_s[:, 1],
            dlin_i=np.round(dlin_s),
            dbac_i=np.round(dbac_s),
            svc_i=svc_s + self.eps * S,
            tw_arr=tw_arr,
            distance_limit=distance_limit,
            capacity=int(round((1 - self.eps) * S)),
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
        self.params = build_solve_params(
            self._exclude_operators,
            use_all_operators=self._use_all_operators,
        )
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
        num_iters: int = 5000,
        time_limit: Optional[float] = None,
        dmax: int = 30,
        dmin: int = 15,
        gamma: int = 30,
        eta_min: float = 0.01,
    ):
        """AILS-II: Adaptive Iterated Local Search (Máximo, Cordeau & Nascimento 2024).

        Implements the full AILS-II diversity-control loop:

        Perturbation degree
            Target edge-distance dβ decays from *dmax* → *dmin* over the run.
            Actual removal count ω adapts every *gamma* iterations so that the
            edge-Hamming distance dist(s, s_r) tracks dβ: if dist > dβ then ω--;
            if dist < dβ then ω++.

        Acceptance criterion (convergent, threshold-accepting style)
            b̄ = f* + η · (f̄ − f*)
            where f* = best cost ever, f̄ = running average of all LS-solution costs.
            Accept s as next reference if f(s) ≤ b̄.
            η decays from 1 → eta_min: at the start b̄ ≈ f̄ (relaxed); at the end
            b̄ ≈ f* (strict).

        Stopping: wall-clock *time_limit* (seconds) or *num_iters* iterations.
        The initial LS on the input tour is excluded from the time budget.
        """
        rng = np.random.RandomState(seed)
        perturbation = PerturbationHeuristics(self._data, self._cost_matrix, rng)

        dmax = max(1, int(dmax))
        dmin = max(1, min(int(dmin), dmax))
        gamma = max(1, int(gamma))
        eta_min = float(np.clip(eta_min, 1e-9, 1.0))
        n_customers = max(1, self.num_customers)

        # ── Initial LS ────────────────────────────────────────────────
        sol = self._tour_to_solution(tour)
        search_inst = self._make_search(seed)
        base_cost, ref_tour = self.run(sol, search_instance=search_inst)

        best_cost = base_cost
        best_tour = ref_tour
        f_bar = float(base_cost)   # running average of ALL LS-solution costs
        n_sols = 1

        omega = float(dmax)        # actual perturbation count (continuous for smooth ±1 steps)
        gamma_diffs: list = []     # accumulated (dist − dβ) over last gamma iterations

        timed = time_limit is not None and time_limit > 0.0
        ils_start = time.perf_counter()
        ils_deadline = ils_start + time_limit if timed else None
        it = 0

        while True:
            # ── Stopping & progress ───────────────────────────────────
            if timed:
                now = time.perf_counter()
                if now >= ils_deadline:
                    break
                # Clamp away from exactly 1 so eta/d_beta never fully collapse
                progress = min(1.0 - 1e-9, (now - ils_start) / time_limit)
            else:
                if it >= num_iters:
                    break
                progress = it / max(1, num_iters - 1) if num_iters > 1 else 0.0

            # ── Adaptive parameters ───────────────────────────────────
            # dβ: target edge-distance between ref and post-LS solution
            d_beta = dmax * ((dmin / dmax) ** progress)
            # η: acceptance relaxation, 1 (relaxed) → eta_min (strict)
            eta = eta_min ** progress
            # Acceptance threshold: b̄ = f* + η·(f̄ − f*)
            b_bar = best_cost + eta * (f_bar - best_cost)

            # ── Perturbation ──────────────────────────────────────────
            omega_int = max(1, min(int(round(omega)), n_customers - 1))
            perturbed = perturbation.perturb(ref_tour, omega=omega_int)

            try:
                sol = self._tour_to_solution(perturbed)
                cost, ls_tour = self.run(sol, search_instance=search_inst)

                # Edge-Hamming distance between LS result and reference
                dist = Search._edge_hamming(ls_tour, ref_tour)
                gamma_diffs.append(dist - d_beta)

                # Running average of all generated LS costs
                n_sols += 1
                f_bar += (cost - f_bar) / n_sols

                # Track global best
                if cost < best_cost:
                    best_cost = cost
                    best_tour = ls_tour

                # Acceptance criterion
                if cost <= b_bar:
                    ref_tour = ls_tour

                # Adjust ω every gamma iterations based on average distance deviation
                if len(gamma_diffs) >= gamma:
                    avg_diff = sum(gamma_diffs) / len(gamma_diffs)
                    if avg_diff > 0:           # distances too large → less perturbation
                        omega = max(1.0, omega - 1.0)
                    elif avg_diff < 0:         # distances too small → more perturbation
                        omega = min(float(n_customers - 1), omega + 1.0)
                    gamma_diffs.clear()

            except Exception:
                pass

            it += 1

        return best_cost, best_tour


def _ls_instance_iterated(
    instance_args,
    tour,
    nb_granular,
    seed,
    num_iters=5000,
    time_limit=None,
    dmax=30,
    dmin=15,
    gamma=30,
    eta_min=0.01,
    vrplib_options: Optional[dict] = None,
    mixed_backhaul: bool = False,
):
    """
    AILS-II perturbation + LS on one instance.
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
        use_all_operators=False,
        scaler=NAILS_SCALER,
        vrplib_options=vrplib_options,
    )
    cost, improved_tour = search.iterated_perturbation_search(
        tour,
        seed=seed,
        num_iters=num_iters,
        time_limit=time_limit,
        dmax=dmax,
        dmin=dmin,
        gamma=gamma,
        eta_min=eta_min,
    )
    return cost, improved_tour

