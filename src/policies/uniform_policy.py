import random
from .base_policy import BasePolicy
class UniformPolicy(BasePolicy):
 def act(self,obs,feasible): return random.randrange(len(feasible))
