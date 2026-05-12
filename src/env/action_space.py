from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

A_FULL: List[int] = [0, 15, 30, 45, 60, 75, 90, 105, 120]


@dataclass(frozen=True)
class ActionMaskInputs:
    charger_available: bool
    e_current: float
    e_max: float
    eta_e: float
    p_chg: float
    u_max: int = 120


def floor_to_action_step(value: float, step: int = 15) -> int:
    if value <= 0:
        return 0
    return int(value // step) * step


def compute_u_bar(inputs: ActionMaskInputs) -> int:
    if not inputs.charger_available:
        return 0
    if inputs.eta_e <= 0 or inputs.p_chg <= 0:
        return 0
    max_by_energy = 3600.0 * max(0.0, inputs.e_max - inputs.e_current) / (inputs.eta_e * inputs.p_chg)
    return floor_to_action_step(min(float(inputs.u_max), max_by_energy))


def feasible_actions(inputs: ActionMaskInputs) -> List[int]:
    if not inputs.charger_available:
        return [0]
    u_bar = compute_u_bar(inputs)
    return [a for a in A_FULL if a <= u_bar]


def action_mask(inputs: ActionMaskInputs) -> np.ndarray:
    feasible = set(feasible_actions(inputs))
    return np.array([1 if a in feasible else 0 for a in A_FULL], dtype=np.int8)
