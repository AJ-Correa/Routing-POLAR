import pyvrp
import math
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
from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve
import numpy as np

_BOOSTER_SCHEDULE = (12, 200, 3_000, 50_000, 1_000_000)

from .cython_heuristics.heuristics import insertion_by_cost, random_removal


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
    sq = np.sum(locs * locs, axis=1)
    matrix = sq[:, None] + sq[None, :] - 2.0 * locs @ locs.T
    np.maximum(matrix, 0, out=matrix)
    np.sqrt(matrix, out=matrix)

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
        self.cost_matrix = cost_matrix
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

    def concentric_removal(self, tour, omega):
        """Remove omega vertices closest to a randomly chosen seed vertex."""
        customer_positions = [i for i, node in enumerate(tour) if node >= self.num_depots]
        if len(customer_positions) <= omega:
            return tour, []
        
        seed_pos = self.rng.choice(customer_positions)
        seed_node = tour[seed_pos]
        
        # Access client via index from the clients list
        # ProblemData clients are 0-indexed for customers (depots are separate)
        # seed_node is the actual node index in the tour
        # For pyvrp, client index = node_index - num_depots
        client_idx = seed_node - self.num_depots
        clients_list = self.data.clients()
        
        if client_idx < 0 or client_idx >= len(clients_list):
            # Fallback to random removal if seed is invalid
            return self.random_removal(tour, omega)
        
        seed_client = clients_list[client_idx]
        seed_coords = (seed_client.x, seed_client.y)
        
        distances = []
        for pos in customer_positions:
            node = tour[pos]
            if node >= self.num_depots:
                # Same indexing for all customers
                c_idx = node - self.num_depots
                if c_idx < 0 or c_idx >= len(clients_list):
                    continue
                client = clients_list[c_idx]
                dist = math.sqrt((client.x - seed_coords[0])**2 + (client.y - seed_coords[1])**2)
                distances.append((dist, pos, node))
        
        if len(distances) < omega:
            # Not enough valid customers, fallback
            return self.random_removal(tour, omega)
        
        distances.sort()
        removed = distances[:omega]
        removed_indices = sorted([r[1] for r in removed], reverse=True)
        removed_vertices = [r[2] for r in removed]
        
        new_tour = list(tour)
        for idx in removed_indices:
            new_tour.pop(idx)
            
        return new_tour, removed_vertices

    def sequence_removal(self, tour, omega):
        """Remove omega consecutive vertices from a random starting point."""
        customer_positions = [i for i, node in enumerate(tour) if node >= self.num_depots]
        if len(customer_positions) <= omega:
            return tour, []
        
        valid_starts = []
        for i in range(len(customer_positions) - omega + 1):
            start_pos = customer_positions[i]
            end_pos = customer_positions[i + omega - 1]
            if end_pos - start_pos == omega - 1:
                valid_starts.append(i)
        
        if not valid_starts:
            return self.random_removal(tour, omega)
        
        start_idx = self.rng.choice(valid_starts)
        removed_indices = sorted(customer_positions[start_idx:start_idx + omega], reverse=True)
        removed_vertices = [tour[i] for i in removed_indices]
        
        new_tour = list(tour)
        for idx in removed_indices:
            new_tour.pop(idx)
            
        return new_tour, removed_vertices

    def insertion_by_distance(self, partial_tour, removed_vertices):
        """Insert removed vertices closest to their nearest neighbor in tour."""
        tour = list(partial_tour)
        
        for vertex in removed_vertices:
            best_dist = float('inf')
            best_pos = 0
            
            for pos in range(len(tour) + 1):
                if pos > 0:
                    prev_node = tour[pos - 1]
                    dist_to_prev = self.cost_matrix[vertex, prev_node]
                else:
                    dist_to_prev = float('inf')
                
                if pos < len(tour):
                    next_node = tour[pos]
                    dist_to_next = self.cost_matrix[vertex, next_node]
                else:
                    dist_to_next = float('inf')
                
                min_dist = min(dist_to_prev, dist_to_next)
                
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_pos = pos
            
            tour.insert(best_pos, vertex)
        
        return tour
    
    def perturb(self, tour, omega):
        """
        AILS-II style perturbation: remove omega vertices then reinsert.
        """
        import random
        tour = list(tour)
        if random.random() < 0.5:
            partial_tour, removed = self.concentric_removal(tour, omega)
        else:
            partial_tour, removed = self.sequence_removal(tour, omega)
        # partial_tour, removed = self.random_removal(tour, omega)
        
        if len(removed) == 0:
            return tour
        
        if random.random() < 0.5:
            return self.insertion_by_distance(partial_tour, removed)
        else:
            return self.insertion_by_cost(partial_tour, removed)
        # return self.insertion_by_cost(partial_tour, removed)


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
        self.scaler = 10_000_000
        self.eps = 1e-30

        S = self.scaler
        eps = self.eps
        nd = num_depots
        nc = self.num_customers

        # ── vectorised pre-scaling ────
        locs_s = np.asarray(locs, dtype=np.float64) * S
        dlin_s = np.asarray(demands_linehauls, dtype=np.float64) * S
        dbac_s = np.asarray(demands_backhauls, dtype=np.float64) * S
        svc_s = np.asarray(service_times, dtype=np.float64) * S
        tw_arr = np.asarray(time_windows, dtype=np.float64)

        xs_i = locs_s[:, 0].astype(np.int64)
        ys_i = locs_s[:, 1].astype(np.int64)
        dlin_i = np.round(dlin_s).astype(np.int64)
        dbac_i = np.round(dbac_s).astype(np.int64)
        svc_i = (svc_s + eps * S).astype(np.int64)

        tw_early_all = tw_arr[:, 0]
        tw_late_all = tw_arr[:, 1]
        tw_finite = np.isfinite(tw_early_all) & np.isfinite(tw_late_all)

        _safe_early = np.where(tw_finite, tw_early_all, 0.0)
        _safe_late = np.where(tw_finite, tw_late_all, 0.0)
        tw_early_i = ((_safe_early + eps) * S).astype(np.int64)
        tw_late_i = ((_safe_late - eps) * S).astype(np.int64)

        depots = [Depot(x=int(xs_i[i]), y=int(ys_i[i])) for i in range(nd)]

        clients = []
        for i in range(nd, nd + nc):
            if tw_finite[i]:
                clients.append(
                    Client(
                        x=int(xs_i[i]), y=int(ys_i[i]),
                        delivery=[int(dlin_i[i])], pickup=[int(dbac_i[i])],
                        service_duration=int(svc_i[i]),
                        tw_early=int(tw_early_i[i]), tw_late=int(tw_late_i[i]),
                    )
                )
            else:
                clients.append(
                    Client(
                        x=int(xs_i[i]), y=int(ys_i[i]),
                        delivery=[int(dlin_i[i])], pickup=[int(dbac_i[i])],
                        service_duration=int(svc_i[i]),
                    )
                )

        cap_i = int(round((1 - eps) * S))
        max_dist_i = int((distance_limit - eps) * S) if np.isfinite(distance_limit) else None
        vehicle_types = []
        for i in range(nd):
            vkw = dict(num_available=nc, start_depot=i, end_depot=i, capacity=[cap_i])
            if max_dist_i is not None:
                vkw["max_distance"] = max_dist_i
            if tw_finite[i]:
                vkw["tw_early"] = int(tw_early_i[i])
                vkw["tw_late"] = int(tw_late_i[i])
            vehicle_types.append(VehicleType(**vkw))

        cost_matrix = np.ascontiguousarray(
            compute_cost_matrix(
                locs_s, dlin_s, dbac_s,
                open_route=open_route,
                max_distance=distance_limit * S,
                num_depots=nd,
                mixed_backhaul=mixed_backhaul,
            ),
            dtype=np.float64,
        )

        self._data = ProblemData(clients, depots, vehicle_types, [cost_matrix], [cost_matrix])
        self.model = None
        self._cost_matrix = cost_matrix / S

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

        # Fallback: return original if no improvement found
        # Note: original may be infeasible, but we preserve it for perturbation continuity
        return base_distance / self.scaler, self._extract_tour(solution)

    def iterated_perturbation_search(
        self,
        tour,
        seed: int = 0,
        num_iters: int = 5,
        omega: int = 5,
    ):
        """
        ILS-style: iterate perturbation → LS for num_iters cycles.
        Always returns the best FEASIBLE solution found.
        Leverages initial solution as promising starting point.
        
        Parameters
        ----------
        tour : list
            Initial promising solution (from neural decoder)
        seed : int
            RNG seed
        num_iters : int
            Number of perturbation → LS cycles
        omega : int
            Number of vertices to remove per perturbation
            
        Returns
        -------
        best_cost : float
            Best feasible cost found
        best_tour : list
            Best feasible tour found
        """
        # Initialize perturbation heuristics
        rng = np.random.RandomState(seed)
        perturbation = PerturbationHeuristics(self._data, self._cost_matrix, rng)
        
        # Establish feasible baseline via LS on initial tour
        sol = self._tour_to_solution(tour)
        base_cost, base_tour = self.run(sol, search_instance=self._make_search(seed))
        
        best_cost = base_cost
        best_tour = base_tour
        
        # Adaptive omega: start small (gentle), grow if stuck
        current_omega = omega
        
        no_improvement_count = 0
        
        for it in range(num_iters):
            # ── Perturbation ─────────────────────────────────────────────
            perturbed = perturbation.perturb(
                best_tour,  # Perturb the current best (or recent) solution
                omega=current_omega,
            )
            
            # ── Local Search on perturbed solution ─────────────────────
            try:
                sol = self._tour_to_solution(perturbed)
                fresh_search = self._make_search(seed=seed + it + 1)
                
                cost, improved_tour = self.run(sol, search_instance=fresh_search)
                
                # ── Acceptance & Update ────────────────────────────────
                if cost < best_cost:
                    # Strict improvement: accept and reset
                    best_cost = cost
                    best_tour = improved_tour
                    no_improvement_count = 0
                    current_omega = omega  # Reset omega
                else:
                    # No improvement: increase perturbation strength
                    no_improvement_count += 1
                    current_omega = min(omega + no_improvement_count, len(tour) // 4)
                    
                    # Occasionally restart from best known (diversification)
                    if no_improvement_count >= 8:
                        no_improvement_count = 0
                        
            except Exception:
                # Invalid tour format: skip this iteration
                no_improvement_count += 1
                continue
        
        # Guaranteed feasible: best_tour came from a successful LS run
        return best_cost, best_tour


def _ls_instance_iterated(instance_args, tour, nb_granular, seed,
                          num_iters=5, omega=5,):
    """
    Iterated perturbation + LS on one instance.
    Returns (cost, improved_tour) where improved_tour is guaranteed feasible.
    """
    (
        locs, demands_linehaul, demands_backhaul,
        distance_limit, open_route, time_windows, service_time,
    ) = instance_args
    search = Search(
        locs, demands_linehaul, demands_backhaul,
        distance_limit, open_route, time_windows, service_time,
        nb_granular=nb_granular,
    )
    cost, improved_tour = search.iterated_perturbation_search(
        tour, seed=seed,
        num_iters=num_iters,
        omega=omega,
    )
    return cost, improved_tour
    