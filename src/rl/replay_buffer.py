from __future__ import annotations
from collections import deque
import random
class ReplayBuffer:
    def __init__(self,capacity:int): self.buf=deque(maxlen=capacity)
    def add(self,*transition): self.buf.append(transition)
    def sample(self,batch:int): return random.sample(self.buf,batch)
    def __len__(self): return len(self.buf)
