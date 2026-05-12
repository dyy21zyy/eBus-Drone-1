from env.dwell_time import DwellInputs, compute_dwell_time


def test_parallel_ops_without_excess():
    out = compute_dwell_time(
        DwellInputs(
            n_al=10,
            n_bo0=8,
            rho_al=2,
            rho_bo=1,
            tau_q=3,
            q_f=5,
            rho_f=2,
            u_r=12,
            tau_e=4,
            has_passenger_service=True,
            onboard_affected=30,
        )
    )
    assert out.t_p_hat == 20
    assert out.t_p == 23
    assert out.t_f == 10
    assert out.t_s_hat == 23
    assert out.t_s == 23
    assert out.delta_s == 0
    assert out.passenger_delay == 0


def test_excess_due_to_unloading_or_charging():
    out = compute_dwell_time(
        DwellInputs(
            n_al=1,
            n_bo0=2,
            rho_al=1,
            rho_bo=1,
            tau_q=0,
            q_f=20,
            rho_f=2,
            u_r=15,
            tau_e=5,
            has_passenger_service=True,
            onboard_affected=10,
        )
    )
    assert out.t_p == 2
    assert out.t_f == 40
    assert out.t_s_hat == 40
    assert out.t_s == 45
    assert out.delta_s == 43
    assert out.passenger_delay == 430


def test_parcel_only_stop_delta_equals_ts():
    out = compute_dwell_time(
        DwellInputs(
            n_al=0,
            n_bo0=0,
            rho_al=1,
            rho_bo=1,
            tau_q=0,
            q_f=10,
            rho_f=3,
            u_r=0,
            tau_e=2,
            has_passenger_service=False,
            onboard_affected=12,
        )
    )
    assert out.t_p == 0
    assert out.t_f == 30
    assert out.t_s == 32
    assert out.delta_s == 32
    assert out.passenger_delay == 384
