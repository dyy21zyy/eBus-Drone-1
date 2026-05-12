from .base_policy import BasePolicy
class BatteryThresholdPolicy(BasePolicy):
 def act(self,obs,feasible): return len(feasible)-1 if obs[1]<80 else 0
