from __future__ import annotations
import numpy as np

class BasePolicy:
    def select_action(self, observation, action_mask, info=None)->int: raise NotImplementedError

class NoChargingPolicy(BasePolicy):
    def select_action(self, observation, action_mask, info=None)->int: return 0

class UniformChargingPolicy(BasePolicy):
    def __init__(self, action_index:int): self.action_index=action_index
    def select_action(self, observation, action_mask, info=None)->int: return self.action_index

class MaxFeasiblePolicy(BasePolicy):
    def select_action(self, observation, action_mask, info=None)->int:
        idx=np.where(np.asarray(action_mask)>0)[0]
        return int(idx.max()) if len(idx) else 0

class DwellGreedyPolicy(BasePolicy):
    def __init__(self, dwell_scale:float=1.0): self.scale=dwell_scale
    def select_action(self, observation, action_mask, info=None)->int:
        est=float((info or {}).get('dwell_est',60.0))*self.scale
        actions=(info or {}).get('action_set',[0,15,30,45,60,75,90,105,120])
        feasible=[i for i,m in enumerate(action_mask) if m>0 and actions[i]<=est]
        return max(feasible) if feasible else 0

class BatteryThresholdPolicy(BasePolicy):
    def __init__(self,e_low:float,e_target:float): self.e_low=e_low; self.e_target=e_target
    def select_action(self, observation, action_mask, info=None)->int:
        e=float((info or {}).get('e_current',0))
        feas=[i for i,m in enumerate(action_mask) if m>0]
        if not feas: return 0
        if e<=self.e_low: return max(feas)
        if e>=self.e_target: return 0
        return min(feas,key=lambda x: abs(x-4))
