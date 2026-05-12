from .base_policy import BasePolicy
class NoChargingPolicy(BasePolicy):
 def act(self,obs,feasible): return 0
