
import heapq
class EventCalendar:
    def __init__(self): self.q=[]
    def push(self,t,e): heapq.heappush(self.q,(t,e))
    def pop(self): return heapq.heappop(self.q) if self.q else None
