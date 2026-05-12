from __future__ import annotations
import random, numpy as np

def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
