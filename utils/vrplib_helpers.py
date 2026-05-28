from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from pyvrp.read import ROUND_FUNCS
from scipy.spatial.distance import cdist

# Default rounding used by PyVRP for common CVRPLIB families.
VRPLIB_ROUND_FUNC_BY_FAMILY: Dict[str, str] = {
    "A": "round",
    "B": "round",
    "E": "round",
    "F": "round",
    "M": "exact",
    "P": "round",
    "X": "round",
}

# TensorDict-safe encoding for round_func (strings cannot be stored in TensorDict).
VRPLIB_ROUND_FUNC_NAMES: Tuple[str, ...] = tuple(ROUND_FUNCS.keys())
VRPLIB_ROUND_FUNC_IDS: Dict[str, int] = {
    name: idx for idx, name in enumerate(VRPLIB_ROUND_FUNC_NAMES)
}


def vrplib_round_func_from_id(round_id: int) -> str:
    return VRPLIB_ROUND_FUNC_NAMES[int(round_id)]


def default_vrplib_round_func(dataset: str) -> str:
    family = dataset.split("-")[0]
    return VRPLIB_ROUND_FUNC_BY_FAMILY.get(family, "round")


def resolve_round_func(
    round_func: Union[str, Callable[[np.ndarray], np.ndarray]],
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(round_func, str):
        key = round_func
        if key not in ROUND_FUNCS:
            raise ValueError(
                f"round_func={key!r} is not supported. "
                f"Choose one of {tuple(ROUND_FUNCS)}."
            )
        return ROUND_FUNCS[key]
    if not callable(round_func):
        raise TypeError("round_func must be a string or a callable.")
    return round_func


def compute_vrplib_cost_matrix(
    node_coord: np.ndarray,
    round_func: Union[str, Callable[[np.ndarray], np.ndarray]] = "round",
    edge_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build integer CVRPLIB distance matrix compatible with pyvrp.read(..., round_func=...).
    """
    rf = resolve_round_func(round_func)
    coords = rf(np.asarray(node_coord, dtype=np.float64))

    if edge_weight is not None:
        matrix = rf(np.asarray(edge_weight, dtype=np.float64)).astype(np.int64, copy=False)
    else:
        matrix = np.round(cdist(coords, coords, metric="euclidean")).astype(np.int64)

    return np.ascontiguousarray(matrix), coords
