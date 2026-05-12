from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from env.action_space import A_FULL, ActionMaskInputs
from env.dwell_time import DwellInputs, compute_dwell_time
from env.station_ops import Parcel, battery_charging_step, rolling_horizon_dispatch
from rl.action_mask import get_action_mask, get_feasible_actions


@dataclass
class Event:
    time: float
    bus_id: str
    station_id: str
    is_integrated_station: bool
    chi: int
    n_al: int
    n_bo0: int
    q_f: float
    onboard_before: int
    parcel_onboard_before: float
    local_urgency: float


class EBusDroneEnv:
    def __init__(self, config: Dict[str, Any], events: List[Event]):
        self.config = config
        self.events = sorted(events, key=lambda e: e.time)
        self.bus_ids = config["bus_ids"]
        self.station_ids = config["station_ids"]
        self.last_time = 0.0
        self._event_idx = 0
        self._current_event = None
        self._done = False

        self.bus_battery = {b: float(config["bus_battery_init"][b]) for b in self.bus_ids}
        self.bus_passenger_load = {b: int(config["bus_passenger_init"].get(b, 0)) for b in self.bus_ids}
        self.bus_parcel_load = {b: float(config["bus_parcel_init"].get(b, 0.0)) for b in self.bus_ids}
        self.available_chargers = {s: int(config["charger_count"][s]) for s in self.station_ids}
        self.station_power_limit = {s: float(config["station_power_limit"][s]) for s in self.station_ids}
        self.queue = {s: int(config["passenger_queue_init"].get(s, 0)) for s in self.station_ids}

        self.locker = {s: [] for s in self.station_ids}
        self.idle_drones = {s: [f"{s}_d{k}" for k in range(int(config["idle_drones_init"].get(s, 0)))] for s in self.station_ids}
        self.active_drones: Dict[str, List[tuple[str, float]]] = {s: [] for s in self.station_ids}
        self.full_batt = {s: int(config["full_battery_init"].get(s, 0)) for s in self.station_ids}
        self.depleted_batt = {s: int(config["depleted_battery_init"].get(s, 0)) for s in self.station_ids}

        self.delivery_completion: Dict[str, float] = {}
        self.pending_parcels: Dict[str, Parcel] = {}

    def reset(self, seed=None, options=None):
        self._done = False
        self._event_idx = 0
        self.last_time = 0.0
        self._current_event = self._find_next_decision_event()
        return self._build_observation(), {"event": self._current_event}

    def get_action_mask(self):
        if self._current_event is None:
            return np.array([1] + [0] * (len(A_FULL) - 1), dtype=np.int8)
        e = self._current_event
        return get_action_mask(ActionMaskInputs(self.available_chargers[e.station_id] > 0, self.bus_battery[e.bus_id], self.config["e_max"], self.config["eta_e"], self.config["p_chg"], max(A_FULL)))

    def get_feasible_actions(self):
        if self._current_event is None:
            return [0]
        e = self._current_event
        return get_feasible_actions(ActionMaskInputs(self.available_chargers[e.station_id] > 0, self.bus_battery[e.bus_id], self.config["e_max"], self.config["eta_e"], self.config["p_chg"], max(A_FULL)))

    def step(self, action):
        if self._done:
            raise RuntimeError("done")
        if action not in self.get_feasible_actions():
            raise ValueError("infeasible action")
        e = self._current_event
        if e is None:
            self._done = True
            return self._build_observation(), 0.0, True, False, {"event": None, "reward_components": {}}

        dwell = compute_dwell_time(DwellInputs(e.n_al, e.n_bo0, self.config["rho_al"], self.config["rho_bo"], self.config.get("tau_q", 0), e.q_f, self.config["rho_f"], float(action), self.config.get("tau_e", 0), (e.n_al > 0 or e.n_bo0 > 0), e.onboard_before))

        dt = dwell.t_s
        self.bus_battery[e.bus_id] = min(self.config["e_max"], self.bus_battery[e.bus_id] + self.config["eta_e"] * self.config["p_chg"] * action / 3600.0)
        min_energy = min(self.bus_battery[e.bus_id], self.bus_battery[e.bus_id] - self.config.get("travel_consumption_per_event", 0.0))
        self.bus_battery[e.bus_id] = max(0.0, self.bus_battery[e.bus_id] - self.config.get("travel_consumption_per_event", 0.0))

        # release parcels -> locker and trigger station dispatch
        for k in range(int(e.q_f)):
            pid = f"{e.bus_id}_{e.station_id}_{int(e.time)}_{k}"
            deadline = e.time + self.config.get("default_deadline_offset", 1800.0)
            p = Parcel(pid, f"c_{pid}", e.station_id, e.time, deadline)
            self.locker[e.station_id].append(p)
            self.pending_parcels[p.parcel_id] = p

        self._run_station_ops(e.station_id, e.time, trigger_reason="release")

        self.last_time = e.time + dt
        self._run_station_ops(e.station_id, self.last_time, trigger_reason="interval")

        d_p = dwell.passenger_delay
        d_l = self._collect_lateness_penalty(until=self.last_time)
        d_e = self.config.get("eta_E_cost", 0.0) * (self.config.get("travel_consumption_per_event", 0.0) + self.config["eta_e"] * self.config["p_chg"] * action / 3600.0)

        p_e = self.config.get("p_chg", 0.0) if action > 0 else 0.0
        g_h, p_d, p_tot, overload = battery_charging_step(self.depleted_batt[e.station_id], self.config.get("G_max", 0), self.station_power_limit[e.station_id], p_e, self.config.get("P_L", 0.0), self.config.get("P_bat", 1.0))
        self.depleted_batt[e.station_id] -= g_h
        self.full_batt[e.station_id] += g_h

        d_pwr = self.config.get("eta_P", 1.0) * overload * dt
        d_b = self.config.get("eta_B", 1.0) * max(0.0, self.config.get("E_min", 0.0) - min_energy)
        d_k = self.config.get("eta_K", 1.0) * max(0.0, len(self.locker[e.station_id]) - self.config.get("locker_capacity", 999999)) * dt

        terminated = self.bus_battery[e.bus_id] <= 0.0
        self._current_event = self._find_next_decision_event()
        if self._current_event is None:
            terminated = True
        terminal_pen = 0.0
        if terminated:
            for parcel in self.pending_parcels.values():
                terminal_pen += self.config.get("eta_L_term", 0.0) * max(0.0, self.config.get("T_end", self.last_time) - parcel.deadline) + self.config.get("eta_U_term", 0.0)

        reward_components = {
            "D_P": d_p,
            "D_L": d_l,
            "D_E": d_e,
            "D_Pwr": d_pwr,
            "D_B": d_b,
            "D_K": d_k,
            "terminal": terminal_pen,
            "P_tot": p_tot,
            "P_D": p_d,
        }
        reward = -(
            self.config.get("alpha_1", 1.0) * d_p
            + self.config.get("alpha_2", 1.0) * d_l
            + self.config.get("alpha_3", 1.0) * d_e
            + self.config.get("alpha_4", 1.0) * d_pwr
            + self.config.get("alpha_5", 1.0) * d_b
            + self.config.get("alpha_6", 1.0) * d_k
            + terminal_pen
        )
        self._done = terminated
        return self._build_observation(), float(reward), terminated, False, {"event": e, "reward_components": reward_components}

    def _run_station_ops(self, station_id: str, now: float, trigger_reason: str):
        # release returns first, can trigger dispatch
        returning = [x for x in self.active_drones[station_id] if x[1] <= now]
        self.active_drones[station_id] = [x for x in self.active_drones[station_id] if x[1] > now]
        for drone_id, _ in returning:
            self.idle_drones[station_id].append(drone_id)
            self.depleted_batt[station_id] += 1

        # dispatch trigger conditions
        should_dispatch = trigger_reason in {"release", "interval"} or len(returning) > 0
        if not should_dispatch:
            return

        # ensure feasibility maps exist for synthetic customers
        for p in self.locker[station_id]:
            self.config.setdefault("drone_range_feasible", {})[(station_id, p.customer_id)] = True
            self.config.setdefault("rt_duration_feasible", {})[(station_id, p.customer_id)] = True
            self.config.setdefault("T_out", {})[(station_id, p.customer_id)] = self.config.get("default_T_out", 300.0)
            self.config.setdefault("T_rt", {})[(station_id, p.customer_id)] = self.config.get("default_T_rt", 900.0)
            self.config.setdefault("c_D", {})[(station_id, p.customer_id)] = self.config.get("default_c_D", 1.0)
            self.config.setdefault("deadlines", {})[p.customer_id] = p.deadline

        assignments = rolling_horizon_dispatch(
            station_id=station_id,
            now=now,
            locker=self.locker[station_id],
            idle_drones=self.idle_drones[station_id],
            full_batteries=self.full_batt[station_id],
            t_out=self.config.get("T_out", {}),
            t_rt=self.config.get("T_rt", {}),
            c_d=self.config.get("c_D", {}),
            deadlines=self.config.get("deadlines", {}),
            drone_range_feasible=self.config.get("drone_range_feasible", {}),
            rt_duration_feasible=self.config.get("rt_duration_feasible", {}),
            eta_l_d=self.config.get("eta_L_D", 1.0),
            eta_u_d=self.config.get("eta_U_D", 1.0),
        )
        if not assignments:
            return
        assigned_pids = {a.parcel_id for a in assignments}
        self.locker[station_id] = [p for p in self.locker[station_id] if p.parcel_id not in assigned_pids]

        for a in assignments:
            self.full_batt[station_id] -= 1
            self.idle_drones[station_id].remove(a.drone_id)
            self.active_drones[station_id].append((a.drone_id, a.return_time))
            self.delivery_completion[a.parcel_id] = a.completion_time
            if a.parcel_id in self.pending_parcels:
                del self.pending_parcels[a.parcel_id]

    def _collect_lateness_penalty(self, until: float) -> float:
        penalty = 0.0
        eta = self.config.get("eta_L_R", 1.0)
        for pid, comp in list(self.delivery_completion.items()):
            if comp <= until:
                cid = pid.replace("", "")
                deadline = self.config.get("deadlines", {}).get(cid, until)
                penalty += eta * max(0.0, comp - deadline)
                del self.delivery_completion[pid]
        return penalty

    def _find_next_decision_event(self):
        while self._event_idx < len(self.events):
            e = self.events[self._event_idx]
            self._event_idx += 1
            if e.is_integrated_station and e.chi == 1:
                return e
        return None

    def _build_observation(self):
        return np.array([self.last_time] + [self.bus_battery[b] for b in self.bus_ids] + [float(len(self.locker[s])) for s in self.station_ids], dtype=np.float32)
