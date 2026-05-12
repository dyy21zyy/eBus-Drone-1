import numpy as np

from env.action_space import A_FULL, ActionMaskInputs, action_mask, feasible_actions
from env.environment import EBusDroneEnv, Event


def test_action_mask_constraints():
    m = action_mask(ActionMaskInputs(False, 50, 100, 0.95, 300))
    assert m.tolist() == [1] + [0] * (len(A_FULL) - 1)

    f = feasible_actions(ActionMaskInputs(True, 99, 100, 1.0, 3600, 120))
    assert f == [0]


def test_env_event_driven_step_and_reward():
    config = {
        "seed": 0,
        "bus_ids": ["b1"],
        "station_ids": ["h1", "n1"],
        "integrated_stations": ["h1"],
        "bus_battery_init": {"b1": 50.0},
        "bus_passenger_init": {"b1": 20},
        "bus_parcel_init": {"b1": 5.0},
        "passenger_queue_init": {"h1": 5, "n1": 3},
        "locker_inventory_init": {"h1": 0.0, "n1": 0.0},
        "idle_drones_init": {"h1": 1, "n1": 0},
        "active_drones_init": {"h1": 0, "n1": 0},
        "full_battery_init": {"h1": 2, "n1": 0},
        "depleted_battery_init": {"h1": 0, "n1": 0},
        "charging_battery_init": {"h1": 0, "n1": 0},
        "station_power_init": {"h1": 0.0, "n1": 0.0},
        "station_power_limit": {"h1": 100.0, "n1": 100.0},
        "charger_count": {"h1": 1, "n1": 0},
        "e_max": 100.0,
        "eta_e": 1.0,
        "p_chg": 3600.0,
        "rho_al": 1.0,
        "rho_bo": 1.0,
        "rho_f": 2.0,
        "tau_q": 1.0,
        "tau_e": 2.0,
        "travel_consumption_per_event": 5.0,
    }
    events = [
        Event(0, "b1", "n1", False, 1, 2, 2, 0, 20, 5, 0.2),
        Event(100, "b1", "h1", True, 0, 1, 1, 0, 20, 5, 0.2),
        Event(200, "b1", "h1", True, 1, 3, 2, 4, 18, 5, 0.8),
    ]
    env = EBusDroneEnv(config, events)
    obs, info = env.reset(seed=7)
    assert isinstance(obs, np.ndarray)
    assert info["event"].time == 200
    feas = env.get_feasible_actions()
    assert 0 in feas

    obs2, reward, terminated, truncated, info2 = env.step(15)
    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert info2["dwell"].t_s >= 0
