from .base_policy import BasePolicy
class MaxFeasiblePolicy(BasePolicy):
 def act(self,obs,feasible): return len(feasible)-1
