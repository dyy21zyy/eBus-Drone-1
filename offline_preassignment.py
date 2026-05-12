from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ModelData:
    buses: List[str]
    stations: List[str]
    customers: List[str]
    q: Dict[str, float]
    d: Dict[str, float]
    c_B: Dict[Tuple[str, str], float]
    c_D: Dict[Tuple[str, str], float]
    H_0: Dict[Tuple[str, str, str], float]
    C_0: Dict[Tuple[str, str, str], float]
    bus_capacity: Dict[str, float]
    unload_capacity: Dict[Tuple[str, str], float]
    bus_station_feasible: Dict[Tuple[str, str], int]
    feasible_customer_station: Dict[Tuple[str, str], int]
    locker_capacity: Dict[Tuple[str, int], float]
    locker_load: Dict[Tuple[str, str, str, int], float]
    time_points: List[int]
    T_rt: Dict[Tuple[str, str], float]
    num_drones: Dict[str, int]
    operating_horizon: float
    beta_H: float
    beta_L: float


def _lateness_plus(data: ModelData, b: str, h: str, i: str) -> float:
    return max(0.0, data.C_0[(b, h, i)] - data.d[i])


def _candidate_keys(data: ModelData) -> List[Tuple[str, str, str]]:
    keys: List[Tuple[str, str, str]] = []
    for b in data.buses:
        for h in data.stations:
            if data.bus_station_feasible.get((b, h), 0) != 1:
                continue
            for i in data.customers:
                if data.feasible_customer_station.get((i, h), 0) == 1:
                    keys.append((b, h, i))
    return keys


def _cost(data: ModelData, b: str, h: str, i: str) -> float:
    return (
        data.c_B[(b, h)] * data.q[i]
        + data.c_D[(h, i)]
        + data.beta_H * data.q[i] * data.H_0[(b, h, i)]
        + data.beta_L * _lateness_plus(data, b, h, i)
    )


def _build_assignment_from_vector(
    keys: List[Tuple[str, str, str]], xvals: Iterable[float]
) -> Dict[Tuple[str, str, str], int]:
    return {k: int(round(v)) for k, v in zip(keys, xvals)}


def _check_feasible(data: ModelData, assign: Dict[Tuple[str, str, str], int]) -> bool:
    for i in data.customers:
        if sum(assign.get((b, h, i), 0) for b in data.buses for h in data.stations) != 1:
            return False

    for b in data.buses:
        used = sum(
            data.q[i] * assign.get((b, h, i), 0)
            for h in data.stations
            for i in data.customers
        )
        if used - data.bus_capacity[b] > 1e-9:
            return False

    for b in data.buses:
        for h in data.stations:
            used = sum(data.q[i] * assign.get((b, h, i), 0) for i in data.customers)
            if used - data.unload_capacity[(b, h)] > 1e-9:
                return False

    for b in data.buses:
        for h in data.stations:
            if data.bus_station_feasible.get((b, h), 0) != 1:
                if any(assign.get((b, h, i), 0) == 1 for i in data.customers):
                    return False

    for h in data.stations:
        for t in data.time_points:
            used = sum(
                data.locker_load.get((b, h, i, t), 0.0) * assign.get((b, h, i), 0)
                for b in data.buses
                for i in data.customers
            )
            if used - data.locker_capacity[(h, t)] > 1e-9:
                return False

    for h in data.stations:
        used = sum(
            data.T_rt[(h, i)] * assign.get((b, h, i), 0)
            for b in data.buses
            for i in data.customers
        )
        if used - data.num_drones[h] * data.operating_horizon > 1e-9:
            return False

    return True


def solve_offline_preassignment(data: ModelData) -> Dict[Tuple[str, str, str], int]:
    keys = _candidate_keys(data)

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp

        n = len(keys)
        c = np.array([_cost(data, b, h, i) for (b, h, i) in keys], dtype=float)
        integrality = np.ones(n, dtype=int)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        rows = []
        lbs = []
        ubs = []

        # 1) Each customer exactly once
        for i in data.customers:
            row = np.array([1.0 if kk[2] == i else 0.0 for kk in keys], dtype=float)
            rows.append(row)
            lbs.append(1.0)
            ubs.append(1.0)

        # 2) Bus freight capacity
        for b in data.buses:
            row = np.array([data.q[i] if bb == b else 0.0 for (bb, hh, i) in keys], dtype=float)
            rows.append(row)
            lbs.append(-np.inf)
            ubs.append(data.bus_capacity[b])

        # 3) Station unloading capacity per bus-station
        for b in data.buses:
            for h in data.stations:
                row = np.array(
                    [data.q[i] if (bb == b and hh == h) else 0.0 for (bb, hh, i) in keys],
                    dtype=float,
                )
                rows.append(row)
                lbs.append(-np.inf)
                ubs.append(data.unload_capacity[(b, h)])

        # 5) Planned locker capacity at time points
        for h in data.stations:
            for t in data.time_points:
                row = np.array(
                    [data.locker_load.get((b, h, i, t), 0.0) for (b, hh, i) in keys], dtype=float
                )
                rows.append(row)
                lbs.append(-np.inf)
                ubs.append(data.locker_capacity[(h, t)])

        # 6) Aggregate drone workload
        for h in data.stations:
            row = np.array(
                [data.T_rt[(h, i)] if hh == h else 0.0 for (b, hh, i) in keys], dtype=float
            )
            rows.append(row)
            lbs.append(-np.inf)
            ubs.append(data.num_drones[h] * data.operating_horizon)

        constraints = LinearConstraint(np.vstack(rows), lb=np.array(lbs), ub=np.array(ubs))
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=[constraints])
        if res.success and res.x is not None:
            assignment = _build_assignment_from_vector(keys, res.x)
            if _check_feasible(data, assignment):
                return assignment
    except Exception:
        pass

    return _greedy_fallback(data, keys)


def _greedy_fallback(
    data: ModelData, keys: List[Tuple[str, str, str]]
) -> Dict[Tuple[str, str, str], int]:
    assignment: Dict[Tuple[str, str, str], int] = {k: 0 for k in keys}

    remaining_bus = {b: data.bus_capacity[b] for b in data.buses}
    remaining_unload = {(b, h): data.unload_capacity[(b, h)] for b in data.buses for h in data.stations}
    remaining_locker = {(h, t): data.locker_capacity[(h, t)] for h in data.stations for t in data.time_points}
    remaining_drone = {h: data.num_drones[h] * data.operating_horizon for h in data.stations}

    # deterministic order by customer id string
    for i in sorted(data.customers):
        options = [(b, h) for (b, h, ii) in keys if ii == i]
        options.sort(key=lambda bh: (_cost(data, bh[0], bh[1], i), bh[0], bh[1]))

        placed = False
        for b, h in options:
            if remaining_bus[b] + 1e-9 < data.q[i]:
                continue
            if remaining_unload[(b, h)] + 1e-9 < data.q[i]:
                continue
            if remaining_drone[h] + 1e-9 < data.T_rt[(h, i)]:
                continue
            locker_ok = True
            for t in data.time_points:
                load = data.locker_load.get((b, h, i, t), 0.0)
                if remaining_locker[(h, t)] + 1e-9 < load:
                    locker_ok = False
                    break
            if not locker_ok:
                continue

            assignment[(b, h, i)] = 1
            remaining_bus[b] -= data.q[i]
            remaining_unload[(b, h)] -= data.q[i]
            remaining_drone[h] -= data.T_rt[(h, i)]
            for t in data.time_points:
                remaining_locker[(h, t)] -= data.locker_load.get((b, h, i, t), 0.0)
            placed = True
            break

        if not placed:
            raise RuntimeError(
                f"Fallback heuristic could not find a feasible assignment for customer '{i}'."
            )

    if not _check_feasible(data, assignment):
        raise RuntimeError("Fallback heuristic produced an infeasible assignment.")

    return assignment


def save_assignment_json(
    assignment: Dict[Tuple[str, str, str], int],
    path: str | Path = "outputs/assignments/offline_assignment_plan.json",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {"bus": b, "station": h, "customer": i, "x": x}
        for (b, h, i), x in sorted(assignment.items())
        if x == 1
    ]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"assignments": records}, f, indent=2)
    return out_path


def load_fixed_assignment_plan(path: str | Path) -> Dict[Tuple[str, str, str], int]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assign: Dict[Tuple[str, str, str], int] = {}
    for row in payload.get("assignments", []):
        assign[(row["bus"], row["station"], row["customer"])] = int(row["x"])
    return assign
