from data_generation.network_generator import generate_instance
from data_generation.scenario_generator import generate_scenario
from env.ebus_drone_env import EBusDroneEnv
from utils.config import load_config
from pathlib import Path
def test_step():
 c=load_config(Path(__file__).resolve().parents[1]);i=generate_instance(c);s=generate_scenario(i,1);e=EBusDroneEnv(i,s);o=e.reset();o,r,d,info=e.step(0);assert isinstance(r,float) and "action_sec" in info
