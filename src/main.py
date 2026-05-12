
from pathlib import Path
from utils.config import load_config
from utils.random_seed import set_seed
from data_generation.network_generator import generate_instance
from data_generation.scenario_generator import generate_scenario
from offline.assignment_solver import solve_assignment
from env.ebus_drone_env import EBusDroneEnv
from policies.uniform_policy import UniformPolicy

def run(instance='medium',seed=42):
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root,instance)
    set_seed(seed)
    inst=generate_instance(cfg)
    scn=generate_scenario(inst,seed)
    _=solve_assignment(inst)
    env=EBusDroneEnv(inst,scn)
    pi=UniformPolicy(); obs=env.reset(); total=0
    while True:
        a=pi.act(obs,env.feasible_actions())
        obs,r,d,_=env.step(a); total+=r
        if d: break
    return total

if __name__=='__main__':
    print({'episode_return':run()})
