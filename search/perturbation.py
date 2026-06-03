import numpy as np

from .cython_heuristics.heuristics import (
    concentric_removal,
    insertion_by_cost,
    insertion_by_distance,
    random_removal,
    sequence_removal,
)


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
