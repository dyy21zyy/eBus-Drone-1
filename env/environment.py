from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.action_space import A_FULL, ActionMaskInputs
from env.dwell_time import DwellInputs, compute_dwell_time
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
        self.rng = np.random.default_rng(config.get("seed", 0))
        self._done = False
        self._event_idx = 0
        self._current_event: Optional[Event] = None
        self._passenger_delay_acc = 0.0

        self.bus_ids: List[str] = config["bus_ids"]
        self.station_ids: List[str] = config["station_ids"]
        self.integrated_stations = set(config["integrated_stations"])

        self.bus_battery = {b: float(config["bus_battery_init"][b]) for b in self.bus_ids}
        self.bus_passenger_load = {b: int(config["bus_passenger_init"].get(b, 0)) for b in self.bus_ids}
        self.bus_parcel_load = {b: float(config["bus_parcel_init"].get(b, 0.0)) for b in self.bus_ids}

        self.queue = {(s): int(config["passenger_queue_init"].get(s, 0)) for s in self.station_ids}
        self.locker = {(s): float(config["locker_inventory_init"].get(s, 0.0)) for s in self.station_ids}

        self.idle_drones = {(s): int(config["idle_drones_init"].get(s, 0)) for s in self.station_ids}
        self.active_drones = {(s): int(config["active_drones_init"].get(s, 0)) for s in self.station_ids}
        self.full_batt = {(s): int(config["full_battery_init"].get(s, 0)) for s in self.station_ids}
        self.depleted_batt = {(s): int(config["depleted_battery_init"].get(s, 0)) for s in self.station_ids}
        self.charging_batt = {(s): int(config["charging_battery_init"].get(s, 0)) for s in self.station_ids}

        self.station_power = {(s): float(config["station_power_init"].get(s, 0.0)) for s in self.station_ids}
        self.station_power_limit = {(s): float(config["station_power_limit"][s]) for s in self.station_ids}
        self.available_chargers = {(s): int(config["charger_count"][s]) for s in self.station_ids}

        self.last_time = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._done = False
        self._event_idx = 0
        self._passenger_delay_acc = 0.0
        self.last_time = 0.0
        self._current_event = self._find_next_decision_event()
        obs = self._build_observation()
        return obs, {"event": self._current_event}

    def get_action_mask(self) -> np.ndarray:
        event = self._current_event
        if event is None:
            return np.array([1] + [0] * (len(A_FULL) - 1), dtype=np.int8)
        inputs = ActionMaskInputs(
            charger_available=self.available_chargers[event.station_id] > 0,
            e_current=self.bus_battery[event.bus_id],
            e_max=self.config["e_max"],
            eta_e=self.config["eta_e"],
            p_chg=self.config["p_chg"],
            u_max=max(A_FULL),
        )
        return get_action_mask(inputs)

    def get_feasible_actions(self) -> list[int]:
        event = self._current_event
        if event is None:
            return [0]
        inputs = ActionMaskInputs(
            charger_available=self.available_chargers[event.station_id] > 0,
            e_current=self.bus_battery[event.bus_id],
            e_max=self.config["e_max"],
            eta_e=self.config["eta_e"],
            p_chg=self.config["p_chg"],
            u_max=max(A_FULL),
        )
        return get_feasible_actions(inputs)

    def step(self, action):
        if self._done:
            raise RuntimeError("Environment is done. Call reset().")
        if self._current_event is None:
            self._done = True
            return self._build_observation(), 0.0, True, False, {"event": None}

        if action not in self.get_feasible_actions():
            raise ValueError(f"Infeasible action {action} at current event.")

        e = self._current_event
        bus = e.bus_id
        station = e.station_id

        # 1-6 dwell and service dynamics
        dwell = compute_dwell_time(
            DwellInputs(
                n_al=e.n_al,
                n_bo0=e.n_bo0,
                rho_al=self.config["rho_al"],
                rho_bo=self.config["rho_bo"],
                tau_q=self.config.get("tau_q", 0.0),
                q_f=e.q_f,
                rho_f=self.config["rho_f"],
                u_r=float(action),
                tau_e=self.config.get("tau_e", 0.0),
                has_passenger_service=(e.n_al > 0 or e.n_bo0 > 0),
                onboard_affected=e.onboard_before,
            )
        )

        # 7 passenger delay
        self._passenger_delay_acc += dwell.passenger_delay

        # 8 departure and downstream propagation (compressed)
        depart_time = e.time + dwell.t_s

        # 9 battery charging + travel consumption
        charged = self.config["eta_e"] * self.config["p_chg"] * float(action) / 3600.0
        self.bus_battery[bus] = min(self.config["e_max"], self.bus_battery[bus] + charged)
        self.bus_battery[bus] = max(0.0, self.bus_battery[bus] - self.config.get("travel_consumption_per_event", 0.0))

        # 10 charger occupancy/release
        if action > 0 and self.available_chargers[station] > 0:
            self.available_chargers[station] -= 1
            self.available_chargers[station] += 1

        # 11 release assigned parcels to locker
        self.bus_parcel_load[bus] = max(0.0, self.bus_parcel_load[bus] - e.q_f)
        self.locker[station] += e.q_f

        # 12-13 low-level station operations (coarse placeholders)
        interval = max(0.0, depart_time - self.last_time)
        dispatched = min(self.idle_drones[station], int(self.locker[station]))
        self.idle_drones[station] -= dispatched
        self.active_drones[station] += dispatched
        self.locker[station] -= dispatched
        returned = min(self.active_drones[station], int(interval // max(1.0, self.config.get("drone_cycle_time", 60.0))))
        self.active_drones[station] -= returned
        self.idle_drones[station] += returned

        power_draw = self.config.get("base_station_power", 0.0) + (self.config.get("p_chg", 0.0) if action > 0 else 0.0)
        self.station_power[station] = power_draw

        # passenger queue updates
        self.bus_passenger_load[bus] = max(0, self.bus_passenger_load[bus] - e.n_al)
        boarded = min(self.queue[station] + e.n_bo0, max(0, self.config.get("bus_capacity_passengers", 80) - self.bus_passenger_load[bus]))
        self.bus_passenger_load[bus] += boarded
        self.queue[station] = max(0, self.queue[station] + int(self.config.get("arrival_rate", 0.0) * dwell.t_s) - boarded)

        # 14 reward over interval
        overload = max(0.0, self.station_power[station] - self.station_power_limit[station])
        reward = -(
            self.config.get("w_delay", 1.0) * dwell.passenger_delay
            + self.config.get("w_dwell", 0.1) * dwell.t_s
            + self.config.get("w_overload", 2.0) * overload
        )

        # 15 next decision event
        self.last_time = depart_time
        self._current_event = self._find_next_decision_event()
        terminated = self._current_event is None
        self._done = terminated

        # 16 return transition
        info = {
            "event": e,
            "depart_time": depart_time,
            "dwell": dwell,
            "feasible_actions": self.get_feasible_actions() if not terminated else [0],
            "passenger_delay_acc": self._passenger_delay_acc,
        }
        return self._build_observation(), float(reward), terminated, False, info

    def _find_next_decision_event(self) -> Optional[Event]:
        while self._event_idx < len(self.events):
            e = self.events[self._event_idx]
            self._event_idx += 1
            if e.is_integrated_station and e.chi == 1:
                return e
        return None

    def _normalize(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    def _build_observation(self) -> np.ndarray:
        t = self._current_event.time if self._current_event else self.last_time
        t_day = max(1.0, self.config.get("t_day", 86400.0))
        cyc = [np.sin(2.0 * np.pi * t / t_day), np.cos(2.0 * np.pi * t / t_day)]

        bus_batt = [self._normalize(self.bus_battery[b], 0.0, self.config["e_max"]) for b in self.bus_ids]
        bus_pax = [self._normalize(float(self.bus_passenger_load[b]), 0.0, float(self.config.get("bus_capacity_passengers", 80))) for b in self.bus_ids]
        bus_parcel = [self._normalize(self.bus_parcel_load[b], 0.0, float(self.config.get("bus_capacity_parcels", 200))) for b in self.bus_ids]
        queues = [self._normalize(float(self.queue[s]), 0.0, float(self.config.get("queue_max", 200))) for s in self.station_ids]
        lockers = [self._normalize(self.locker[s], 0.0, float(self.config.get("locker_max", 500))) for s in self.station_ids]
        idle = [self._normalize(float(self.idle_drones[s]), 0.0, float(self.config.get("drone_max", 20))) for s in self.station_ids]
        active = [self._normalize(float(self.active_drones[s]), 0.0, float(self.config.get("drone_max", 20))) for s in self.station_ids]
        fullb = [self._normalize(float(self.full_batt[s]), 0.0, float(self.config.get("battery_max", 100))) for s in self.station_ids]
        depleted = [self._normalize(float(self.depleted_batt[s]), 0.0, float(self.config.get("battery_max", 100))) for s in self.station_ids]
        charging = [self._normalize(float(self.charging_batt[s]), 0.0, float(self.config.get("battery_max", 100))) for s in self.station_ids]
        power = [self._normalize(self.station_power[s], 0.0, self.station_power_limit[s] * 2.0) for s in self.station_ids]
        margin = [self._normalize(max(0.0, self.station_power_limit[s] - self.station_power[s]), 0.0, self.station_power_limit[s]) for s in self.station_ids]
        chargers = [self._normalize(float(self.available_chargers[s]), 0.0, float(self.config["charger_count"][s])) for s in self.station_ids]

        local = []
        if self._current_event is not None:
            e = self._current_event
            local_station_oh = [1.0 if s == e.station_id else 0.0 for s in self.station_ids]
            local_bus_oh = [1.0 if b == e.bus_id else 0.0 for b in self.bus_ids]
            local = local_station_oh + local_bus_oh + [
                self._normalize(self.bus_battery[e.bus_id], 0.0, self.config["e_max"]),
                self._normalize(float(e.onboard_before), 0.0, float(self.config.get("bus_capacity_passengers", 80))),
                self._normalize(e.parcel_onboard_before, 0.0, float(self.config.get("bus_capacity_parcels", 200))),
                self._normalize(float(e.n_al), 0.0, float(self.config.get("bus_capacity_passengers", 80))),
                self._normalize(float(e.n_bo0), 0.0, float(self.config.get("queue_max", 200))),
                self._normalize(float(e.q_f), 0.0, float(self.config.get("bus_capacity_parcels", 200))),
                self._normalize(float(self.available_chargers[e.station_id]), 0.0, float(self.config["charger_count"][e.station_id])),
                self._normalize(self.locker[e.station_id], 0.0, float(self.config.get("locker_max", 500))),
                self._normalize(float(self.idle_drones[e.station_id]), 0.0, float(self.config.get("drone_max", 20))),
                self._normalize(float(self.full_batt[e.station_id]), 0.0, float(self.config.get("battery_max", 100))),
                self._normalize(max(0.0, self.station_power_limit[e.station_id] - self.station_power[e.station_id]), 0.0, self.station_power_limit[e.station_id]),
                self._normalize(float(e.local_urgency), 0.0, 1.0),
            ]

        vector = np.array(cyc + bus_batt + bus_pax + bus_parcel + queues + lockers + idle + active + fullb + depleted + charging + power + margin + chargers + local, dtype=np.float32)
        return vector
