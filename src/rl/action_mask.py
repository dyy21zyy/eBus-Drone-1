from __future__ import annotations

import numpy as np

from env.action_space import ActionMaskInputs, action_mask as env_action_mask, feasible_actions as env_feasible_actions


def get_action_mask(inputs: ActionMaskInputs) -> np.ndarray:
    return env_action_mask(inputs)


def get_feasible_actions(inputs: ActionMaskInputs) -> list[int]:
    return env_feasible_actions(inputs)
