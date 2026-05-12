from env.action_space import ACTIONS
from env.dwell_time import compute_dwell
class EBusDroneEnv:
    def __init__(self,instance,scenario):
        self.instance=instance; self.scenario=scenario; self.reset()
    def reset(self):
        self.t=0; self.step_i=0; self.energy=160.0; self.done=False
        return self._obs()
    def _obs(self): return [float(self.t),float(self.energy),float(self.step_i)]
    def feasible_actions(self): return ACTIONS
    def step(self,action_idx:int):
        a=ACTIONS[action_idx]; dwell=compute_dwell(2,1,1.0,a)
        self.energy=max(0.0,self.energy-1.6+0.95*500*(a/3600.0))
        self.t+=1; self.step_i+=1
        r=-0.01*dwell-(5 if self.energy<40 else 0)
        self.done=self.step_i>=self.instance['bus_trips']
        return self._obs(),float(r),self.done,{'action_sec':a}
