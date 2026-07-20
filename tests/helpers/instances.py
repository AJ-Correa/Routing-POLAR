"""Hand-built VRP instances for deterministic tests (coords / tags only)."""

import torch
from tensordict import TensorDict


def _base_fields(locs, device="cpu", open_route=False, distance_limit=float("inf")):
    batch = locs.shape[0]
    n = locs.shape[1]
    tw = torch.zeros(batch, n, 2, device=device)
    tw[..., 1] = float("inf")
    demand_linehaul = torch.zeros(batch, n, device=device)
    demand_linehaul[:, 1:] = 1.0
    return {
        "locs": locs,
        "demand_linehaul": demand_linehaul,
        "demand_backhaul": torch.zeros(batch, n, device=device),
        "backhaul_class": torch.ones(batch, 1, device=device),
        "distance_limit": torch.full((batch, 1), distance_limit, device=device),
        "service_time": torch.zeros(batch, n, device=device),
        "open_route": torch.full((batch, 1), open_route, dtype=torch.bool, device=device),
        "time_windows": tw,
        "vehicle_capacity": torch.full((batch, 1), 100.0, device=device),
        "capacity_original": torch.full((batch, 1), 100.0, device=device),
        "speed": torch.ones(batch, 1, device=device),
        "p_s_tag": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 4]], device=device).float(),
    }


def build_cvrp5(device="cpu", open_route=False):
    """1 depot + 4 customers on a unit square."""
    locs = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]],
        device=device,
    )
    fields = _base_fields(locs, device=device, open_route=open_route)
    fields["p_s_tag"] = torch.tensor(
        [[1, int(open_route), 0, 0, 0, 0, 0, 4]], device=device
    ).float()
    return TensorDict(fields, batch_size=[1], device=device)


def build_cvrp5_backhaul(device="cpu", mixed=False):
    locs = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]],
        device=device,
    )
    fields = _base_fields(locs, device=device)
    fields["demand_linehaul"][:] = 0.0
    fields["demand_linehaul"][:, 1:3] = 1.0
    fields["demand_backhaul"][:, 3:5] = 1.0
    fields["backhaul_class"] = torch.full((1, 1), 2 if mixed else 1, device=device)
    fields["p_s_tag"] = torch.tensor(
        [[1, 0, 0, 0, 1, int(mixed), 0, 4]], device=device
    ).float()
    return TensorDict(fields, batch_size=[1], device=device)


def build_cvrp5_tw(device="cpu"):
    locs = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]],
        device=device,
    )
    fields = _base_fields(locs, device=device)
    tw = torch.zeros(1, 5, 2, device=device)
    tw[..., 1] = 10.0
    fields["time_windows"] = tw
    fields["service_time"][:, 1:] = 0.1
    fields["p_s_tag"] = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 4]], device=device).float()
    return TensorDict(fields, batch_size=[1], device=device)


def build_mtd3_c5(device="cpu"):
    """3 depots + 5 customers."""
    locs = torch.tensor(
        [
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [0.2, 0.8],
                [0.5, 0.8],
                [0.8, 0.8],
                [0.5, 0.5],
                [0.2, 0.2],
            ]
        ],
        device=device,
    )
    fields = _base_fields(locs, device=device)
    fields["num_depots"] = torch.tensor([[3]], device=device)
    fields["p_s_tag"] = torch.tensor([[1, 0, 0, 0, 0, 0, 1, 5]], device=device).float()
    return TensorDict(fields, batch_size=[1], device=device)


def manual_tour_cost(locs, actions, open_route=False):
    """Euclidean tour cost matching MTVRPEnv.get_reward (incl. roll wrap)."""
    actions = [int(a) for a in (actions if isinstance(actions, list) else actions.tolist())]
    nodes = [0] + actions
    total = 0.0
    for i in range(len(nodes)):
        a = nodes[i]
        b = nodes[(i + 1) % len(nodes)]
        if open_route and b == 0:
            continue
        total += float(((locs[a] - locs[b]) ** 2).sum().sqrt())
    return total


def actions_to_search_tour(actions):
    """Convert env action sequence to Search.tour_cost node list."""
    seq = [int(x) for x in (actions if isinstance(actions, list) else actions.tolist())]
    if not seq or seq[0] == 0:
        return seq
    return [0] + seq
