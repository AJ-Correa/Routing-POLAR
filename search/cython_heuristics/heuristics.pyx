# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "math.h":
    double INFINITY


cdef Py_ssize_t _collect_customer_positions(
    cnp.ndarray[cnp.int64_t, ndim=1] tour,
    int num_depots,
    cnp.ndarray[cnp.int64_t, ndim=1] customer_positions,
):
    """Fill customer_positions with tour indices of customers. Returns count."""
    cdef Py_ssize_t n = tour.shape[0]
    cdef Py_ssize_t i, j = 0
    for i in range(n):
        if tour[i] >= num_depots:
            customer_positions[j] = i
            j += 1
    return j


cdef void _sort_indices_desc(
    cnp.ndarray[cnp.int64_t, ndim=1] indices,
    Py_ssize_t count,
):
    """Insertion sort indices descending (for mask removal)."""
    cdef Py_ssize_t i, j
    cdef cnp.int64_t temp
    for i in range(1, count):
        temp = indices[i]
        j = i - 1
        while j >= 0 and indices[j] < temp:
            indices[j + 1] = indices[j]
            j -= 1
        indices[j + 1] = temp


cdef tuple _remove_at_positions(
    cnp.ndarray[cnp.int64_t, ndim=1] tour,
    cnp.ndarray[cnp.int64_t, ndim=1] removed_indices_arr,
    Py_ssize_t omega,
):
    """Remove tour positions in removed_indices_arr (length omega), sorted descending."""
    cdef Py_ssize_t n = tour.shape[0]
    cdef Py_ssize_t i, j
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] mask = np.ones(n, dtype=np.uint8)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] new_tour
    cdef cnp.ndarray[cnp.int64_t, ndim=1] removed_vertices

    for i in range(omega):
        mask[removed_indices_arr[i]] = 0

    new_tour = np.empty(n - omega, dtype=np.int64)
    j = 0
    for i in range(n):
        if mask[i]:
            new_tour[j] = tour[i]
            j += 1

    removed_vertices = np.empty(omega, dtype=np.int64)
    for i in range(omega):
        removed_vertices[i] = tour[removed_indices_arr[i]]

    return new_tour.tolist(), removed_vertices.tolist()


def insertion_by_cost(
    cnp.ndarray[cnp.int64_t, ndim=1] partial_tour,
    cnp.ndarray[cnp.int64_t, ndim=1] removed_vertices,
    cnp.ndarray[cnp.float64_t, ndim=2] cost_matrix,
    int num_depots = 1,
):
    """
    C-level regret-1 insertion. Returns new tour as list (matching Python).

    For multi-depot tours (tour[0] < num_depots), position 0 is skipped so
    customers are never inserted before the first depot marker.
    """
    cdef:
        Py_ssize_t n = partial_tour.shape[0]
        Py_ssize_t k = removed_vertices.shape[0]
        Py_ssize_t current_len = n
        Py_ssize_t v_idx, pos, best_pos, i, start_pos
        Py_ssize_t vertex, prev_node, next_node
        double best_cost, cost
        cnp.ndarray[cnp.int64_t, ndim=1] tour = np.empty(n + k, dtype=np.int64)

    for i in range(n):
        tour[i] = partial_tour[i]

    # Multi-depot tours begin with a depot node; skip inserting before it.
    start_pos = 1 if (n > 0 and partial_tour[0] < num_depots) else 0

    for v_idx in range(k):
        vertex = removed_vertices[v_idx]
        best_cost = INFINITY
        best_pos = start_pos

        for pos in range(start_pos, current_len + 1):
            if pos > 0 and pos < current_len:
                prev_node = tour[pos - 1]
                next_node = tour[pos]
                cost = (
                    cost_matrix[prev_node, vertex]
                    + cost_matrix[vertex, next_node]
                    - cost_matrix[prev_node, next_node]
                )
            elif pos == 0:
                if current_len > 0:
                    cost = (
                        cost_matrix[vertex, tour[0]]
                        - cost_matrix[tour[0], tour[0]]
                    )
                else:
                    cost = 0.0
            else:
                cost = cost_matrix[tour[current_len - 1], vertex]

            if cost < best_cost:
                best_cost = cost
                best_pos = pos

        for i in range(current_len, best_pos, -1):
            tour[i] = tour[i - 1]
        tour[best_pos] = vertex
        current_len += 1

    return tour[:current_len].tolist()


def insertion_by_distance(
    cnp.ndarray[cnp.int64_t, ndim=1] partial_tour,
    cnp.ndarray[cnp.int64_t, ndim=1] removed_vertices,
    cnp.ndarray[cnp.float64_t, ndim=2] cost_matrix,
    int num_depots = 1,
):
    """
    Insert each removed vertex at the position with minimum distance to a neighbor.

    For multi-depot tours (tour[0] < num_depots), position 0 is skipped so
    customers are never inserted before the first depot marker.
    """
    cdef:
        Py_ssize_t n = partial_tour.shape[0]
        Py_ssize_t k = removed_vertices.shape[0]
        Py_ssize_t current_len = n
        Py_ssize_t v_idx, pos, best_pos, i, start_pos
        Py_ssize_t vertex, prev_node, next_node
        double best_dist, dist_to_prev, dist_to_next, min_dist
        cnp.ndarray[cnp.int64_t, ndim=1] tour = np.empty(n + k, dtype=np.int64)

    for i in range(n):
        tour[i] = partial_tour[i]

    # Multi-depot tours begin with a depot node; skip inserting before it.
    start_pos = 1 if (n > 0 and partial_tour[0] < num_depots) else 0

    for v_idx in range(k):
        vertex = removed_vertices[v_idx]
        best_dist = INFINITY
        best_pos = start_pos

        for pos in range(start_pos, current_len + 1):
            if pos > 0:
                prev_node = tour[pos - 1]
                dist_to_prev = cost_matrix[vertex, prev_node]
            else:
                dist_to_prev = INFINITY

            if pos < current_len:
                next_node = tour[pos]
                dist_to_next = cost_matrix[vertex, next_node]
            else:
                dist_to_next = INFINITY

            if dist_to_prev < dist_to_next:
                min_dist = dist_to_prev
            else:
                min_dist = dist_to_next

            if min_dist < best_dist:
                best_dist = min_dist
                best_pos = pos

        for i in range(current_len, best_pos, -1):
            tour[i] = tour[i - 1]
        tour[best_pos] = vertex
        current_len += 1

    return tour[:current_len].tolist()


def random_removal(
    cnp.ndarray[cnp.int64_t, ndim=1] tour,
    int omega,
    int num_depots,
    cnp.ndarray[cnp.int64_t, ndim=1] rng_indices,
):
    """
    C-level random removal. Returns (new_tour_list, removed_vertices_list).

    rng_indices: permutation of range(n_customers); first omega entries select removals.
    """
    cdef:
        Py_ssize_t n = tour.shape[0]
        Py_ssize_t i, j, n_customers
        Py_ssize_t temp
        cnp.ndarray[cnp.int64_t, ndim=1] customer_positions
        cnp.ndarray[cnp.int64_t, ndim=1] removed_indices_arr

    n_customers = 0
    for i in range(n):
        if tour[i] >= num_depots:
            n_customers += 1

    if n_customers <= omega:
        return tour.copy().tolist(), []

    customer_positions = np.empty(n_customers, dtype=np.int64)
    _collect_customer_positions(tour, num_depots, customer_positions)

    removed_indices_arr = np.empty(omega, dtype=np.int64)
    for i in range(omega):
        removed_indices_arr[i] = customer_positions[rng_indices[i]]

    _sort_indices_desc(removed_indices_arr, omega)
    return _remove_at_positions(tour, removed_indices_arr, omega)


def concentric_removal(
    cnp.ndarray[cnp.int64_t, ndim=1] tour,
    int omega,
    int num_depots,
    cnp.ndarray[cnp.float64_t, ndim=2] cost_matrix,
    Py_ssize_t seed_cust_idx,
):
    """
    Remove omega customers closest to a seed customer (by cost_matrix distance).

    Returns (new_tour, removed_vertices, status). status=0 on success, status=1 => fallback.
    """
    cdef:
        Py_ssize_t n = tour.shape[0]
        Py_ssize_t i, j, k, n_customers
        Py_ssize_t seed_pos, seed_node, pos, node
        Py_ssize_t best_j, sel
        double dist, best_dist
        cnp.ndarray[cnp.int64_t, ndim=1] customer_positions
        cnp.ndarray[cnp.int64_t, ndim=1] removed_indices_arr
        cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] selected
        cnp.ndarray[cnp.float64_t, ndim=1] distances

    n_customers = 0
    for i in range(n):
        if tour[i] >= num_depots:
            n_customers += 1

    if n_customers <= omega:
        return tour.copy().tolist(), [], 0

    if seed_cust_idx < 0 or seed_cust_idx >= n_customers:
        return [], [], 1

    customer_positions = np.empty(n_customers, dtype=np.int64)
    _collect_customer_positions(tour, num_depots, customer_positions)

    seed_pos = customer_positions[seed_cust_idx]
    seed_node = tour[seed_pos]

    distances = np.empty(n_customers, dtype=np.float64)
    for j in range(n_customers):
        pos = customer_positions[j]
        node = tour[pos]
        distances[j] = cost_matrix[seed_node, node]

    removed_indices_arr = np.empty(omega, dtype=np.int64)
    selected = np.zeros(n_customers, dtype=np.uint8)

    for sel in range(omega):
        best_dist = INFINITY
        best_j = -1
        for j in range(n_customers):
            if selected[j]:
                continue
            dist = distances[j]
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j < 0:
            return [], [], 1
        selected[best_j] = 1
        removed_indices_arr[sel] = customer_positions[best_j]

    _sort_indices_desc(removed_indices_arr, omega)
    new_tour, removed = _remove_at_positions(tour, removed_indices_arr, omega)
    return new_tour, removed, 0


def sequence_removal(
    cnp.ndarray[cnp.int64_t, ndim=1] tour,
    int omega,
    int num_depots,
    long rng_choice,
):
    """
    Remove omega consecutive customers along the tour.

    rng_choice selects among valid consecutive blocks (modulo count).
    Returns (new_tour, removed_vertices, status). status=1 => fallback.
    """
    cdef:
        Py_ssize_t n = tour.shape[0]
        Py_ssize_t i, j, n_customers, n_valid, pick
        Py_ssize_t start_pos, end_pos
        cnp.ndarray[cnp.int64_t, ndim=1] customer_positions
        cnp.ndarray[cnp.int64_t, ndim=1] valid_starts
        cnp.ndarray[cnp.int64_t, ndim=1] removed_indices_arr

    n_customers = 0
    for i in range(n):
        if tour[i] >= num_depots:
            n_customers += 1

    if n_customers <= omega:
        return tour.copy().tolist(), [], 0

    customer_positions = np.empty(n_customers, dtype=np.int64)
    _collect_customer_positions(tour, num_depots, customer_positions)

    n_valid = 0
    valid_starts = np.empty(max(0, n_customers - omega + 1), dtype=np.int64)
    for i in range(n_customers - omega + 1):
        start_pos = customer_positions[i]
        end_pos = customer_positions[i + omega - 1]
        if end_pos - start_pos == omega - 1:
            valid_starts[n_valid] = i
            n_valid += 1

    if n_valid == 0:
        return [], [], 1

    if rng_choice < 0:
        pick = (-rng_choice) % n_valid
    else:
        pick = rng_choice % n_valid

    start_pos = valid_starts[pick]
    removed_indices_arr = np.empty(omega, dtype=np.int64)
    for j in range(omega):
        removed_indices_arr[j] = customer_positions[start_pos + j]

    _sort_indices_desc(removed_indices_arr, omega)
    new_tour, removed = _remove_at_positions(tour, removed_indices_arr, omega)
    return new_tour, removed, 0
