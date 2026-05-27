import pyvrp
from pyvrp import SolveParams
from pyvrp.PenaltyManager import PenaltyManager
from typing import List, Tuple, Optional
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
        return insertion_by_cost(tour_arr, removed_arr, self.cost_matrix)
    
    def insertion_by_distance(self, partial_tour, removed_vertices):
        """Insert removed vertices closest to their nearest neighbor in tour."""
        tour_arr = np.asarray(partial_tour, dtype=np.int64)
        removed_arr = np.asarray(removed_vertices, dtype=np.int64)
        return insertion_by_distance(tour_arr, removed_arr, self.cost_matrix)

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
    ):
        """Constructs the initial search model for the input instance given."""
        self.num_depots = num_depots
        self.num_customers = len(locs) - self.num_depots
        self.scaler = 1_000
        self.eps = 1e-30

        nd = num_depots
        nc = self.num_customers

        # ── vectorised pre-scaling ────
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
                max_distance=distance_limit,
                num_depots=nd,
                mixed_backhaul=mixed_backhaul,
            ),
        )
        cost_matrix = np.round(cost_matrix * self.scaler)

        xs_i = locs_s[:, 0]
        ys_i = locs_s[:, 1]
        dlin_i = np.round(dlin_s)
        dbac_i = np.round(dbac_s)
        svc_i = (svc_s + self.eps * self.scaler)

        tw_early_all = tw_arr[:, 0]
        tw_late_all = tw_arr[:, 1]
        tw_finite = np.isfinite(tw_early_all) & np.isfinite(tw_late_all)

        _safe_early = np.where(tw_finite, tw_early_all, 0.0)
        _safe_late = np.where(tw_finite, tw_late_all, 0.0)
        tw_early_i = ((_safe_early + self.eps) * self.scaler)
        tw_late_i = ((_safe_late - self.eps) * self.scaler)

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
                        service_duration=int(svc_i[i]),
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
                        service_duration=int(svc_i[i]),
                    )
                )

        cap_i = int(round((1 - self.eps) * self.scaler))
        max_dist_i = (
            int((distance_limit - self.eps) * self.scaler) if np.isfinite(distance_limit) else None
        )
        vehicle_types = []
        for i in range(nd):
            vkw = dict(num_available=nc, start_depot=i, end_depot=i, capacity=[cap_i])
            if max_dist_i is not None:
                vkw["max_distance"] = max_dist_i
            if tw_finite[i]:
                vkw["tw_early"] = int(tw_early_i[i])
                vkw["tw_late"] = int(tw_late_i[i])
            vehicle_types.append(VehicleType(**vkw))

        self._data = ProblemData(
            clients, depots, vehicle_types, [cost_matrix], [cost_matrix]
        )
        self.model = None
        self._cost_matrix = cost_matrix / self.scaler

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
                    routes.insert(0, [0] + pre)

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
                    routes.insert(0, [0] + pre)

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
        dmax: int = 10,
        dmin: int = 5,
        acceptance_rate: float = 0.01,
        no_improvement: int = 8,
    ):
        """
        ILS-style: iterate perturbation → LS for num_iters cycles.
        Always returns the best feasible solution found.

        dmax/dmin schedule the number of vertices removed (omega) per iteration.
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

        sol = self._tour_to_solution(tour)
        search = self._make_search(seed)
        base_cost, base_tour = self.run(sol, search_instance=search)

        best_cost = base_cost
        best_tour = base_tour
        current_cost = base_cost
        current_tour = base_tour
        no_improvement_count = 0

        for it in range(num_iters):
            current_omega = max(
                dmin,
                int(round(dmax * ((dmin / dmax) ** (it / max(1, num_iters - 1))))),
            )

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
                continue

        return best_cost, best_tour


def _ls_instance_iterated(
    instance_args,
    tour,
    nb_granular,
    seed,
    num_iters=5,
    dmax=10,
    dmin=5,
    acceptance_rate=0.01,
    no_improvement=8,
):
    """
    Iterated perturbation + LS on one instance.
    Returns (cost, improved_tour) where improved_tour is guaranteed feasible.
    """
    (
        locs,
        demands_linehaul,
        demands_backhaul,
        distance_limit,
        open_route,
        time_windows,
        service_time,
    ) = instance_args
    search = Search(
        locs,
        demands_linehaul,
        demands_backhaul,
        distance_limit,
        open_route,
        time_windows,
        service_time,
        nb_granular=nb_granular,
    )
    cost, improved_tour = search.iterated_perturbation_search(
        tour,
        seed=seed,
        num_iters=num_iters,
        dmax=dmax,
        dmin=dmin,
        acceptance_rate=acceptance_rate,
        no_improvement=no_improvement,
    )
    return cost, improved_tour
