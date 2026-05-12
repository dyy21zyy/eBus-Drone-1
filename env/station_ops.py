from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Dict, List, Tuple


@dataclass
class Parcel:
    parcel_id: str
    customer_id: str
    station_id: str
    release_time: float
    deadline: float


@dataclass
class DroneAssignment:
    station_id: str
    drone_id: str
    parcel_id: str
    customer_id: str
    pickup_time: float
    completion_time: float
    return_time: float


def urgency_score(deadline: float, predicted_completion: float, small_eps: float = 1e-6) -> float:
    return 1.0 / max(deadline - predicted_completion, small_eps)


def rolling_horizon_dispatch(
    station_id: str,
    now: float,
    locker: List[Parcel],
    idle_drones: List[str],
    full_batteries: int,
    t_out: Dict[Tuple[str, str], float],
    t_rt: Dict[Tuple[str, str], float],
    c_d: Dict[Tuple[str, str], float],
    deadlines: Dict[str, float],
    drone_range_feasible: Dict[Tuple[str, str], bool],
    rt_duration_feasible: Dict[Tuple[str, str], bool],
    eta_l_d: float,
    eta_u_d: float,
) -> List[DroneAssignment]:
    feasible_parcels = [
        p for p in locker if drone_range_feasible.get((station_id, p.customer_id), False) and rt_duration_feasible.get((station_id, p.customer_id), False)
    ]

    n_disp = min(len(idle_drones), full_batteries, len(feasible_parcels))
    if n_disp <= 0:
        return []

    scored: List[tuple[float, Parcel]] = []
    for p in feasible_parcels:
        pred_completion = now + t_out[(station_id, p.customer_id)]
        pred_late = max(0.0, pred_completion - deadlines[p.customer_id])
        urg = urgency_score(deadlines[p.customer_id], pred_completion)
        obj = c_d[(station_id, p.customer_id)] + eta_l_d * pred_late - eta_u_d * urg
        scored.append((obj, p))

    scored.sort(key=lambda x: (x[0], x[1].parcel_id))
    chosen = [p for _, p in scored[:n_disp]]

    assignments: List[DroneAssignment] = []
    for drone_id, parcel in zip(idle_drones[:n_disp], chosen):
        assignments.append(
            DroneAssignment(
                station_id=station_id,
                drone_id=drone_id,
                parcel_id=parcel.parcel_id,
                customer_id=parcel.customer_id,
                pickup_time=now,
                completion_time=now + t_out[(station_id, parcel.customer_id)],
                return_time=now + t_rt[(station_id, parcel.customer_id)],
            )
        )
    return assignments


def battery_charging_step(
    b_empty: int,
    g_max: int,
    p_capacity: float,
    p_e: float,
    p_l: float,
    p_bat: float,
) -> tuple[int, float, float, float]:
    residual = max(0.0, p_capacity - p_e - p_l)
    g_h = min(b_empty, g_max, int(floor(residual / p_bat)) if p_bat > 0 else 0)
    p_d = g_h * p_bat
    p_tot = p_e + p_d + p_l
    overload = max(0.0, p_tot - p_capacity)
    return g_h, p_d, p_tot, overload
