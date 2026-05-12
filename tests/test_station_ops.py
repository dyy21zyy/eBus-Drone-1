from env.station_ops import Parcel, battery_charging_step, rolling_horizon_dispatch


def test_dispatch_respects_capacity_and_exact_n_disp():
    locker = [Parcel("p1", "c1", "h1", 0, 100), Parcel("p2", "c2", "h1", 0, 200)]
    out = rolling_horizon_dispatch(
        "h1", 0, locker, ["d1", "d2", "d3"], 1,
        {("h1", "c1"): 10, ("h1", "c2"): 20},
        {("h1", "c1"): 30, ("h1", "c2"): 40},
        {("h1", "c1"): 1, ("h1", "c2"): 2},
        {"c1": 50, "c2": 50},
        {("h1", "c1"): True, ("h1", "c2"): True},
        {("h1", "c1"): True, ("h1", "c2"): True},
        1.0, 1.0,
    )
    assert len(out) == 1


def test_battery_charge_step():
    g_h, p_d, p_tot, overload = battery_charging_step(10, 3, 100, 30, 20, 10)
    assert g_h == 3
    assert p_d == 30
    assert p_tot == 80
    assert overload == 0
