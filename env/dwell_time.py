from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DwellInputs:
    n_al: float
    n_bo0: float
    rho_al: float
    rho_bo: float
    tau_q: float
    q_f: float
    rho_f: float
    u_r: float
    tau_e: float
    has_passenger_service: bool
    onboard_affected: float


@dataclass(frozen=True)
class DwellOutputs:
    t_p_hat: float
    t_p: float
    t_f: float
    t_s_hat: float
    t_s: float
    delta_s: float
    passenger_delay: float


def compute_dwell_time(inputs: DwellInputs) -> DwellOutputs:
    t_p_hat = max(inputs.rho_al * inputs.n_al, inputs.rho_bo * inputs.n_bo0)
    t_p = t_p_hat + max(0.0, inputs.tau_q)
    t_f = inputs.rho_f * inputs.q_f
    t_s_hat = max(t_p, t_f, inputs.u_r)

    needs_excess = max(t_f, inputs.u_r) > t_p
    t_s = t_s_hat + (max(0.0, inputs.tau_e) if needs_excess else 0.0)

    if inputs.has_passenger_service:
        delta_s = t_s - t_p
    else:
        delta_s = t_s

    passenger_delay = inputs.onboard_affected * delta_s

    return DwellOutputs(
        t_p_hat=t_p_hat,
        t_p=t_p,
        t_f=t_f,
        t_s_hat=t_s_hat,
        t_s=t_s,
        delta_s=delta_s,
        passenger_delay=passenger_delay,
    )
