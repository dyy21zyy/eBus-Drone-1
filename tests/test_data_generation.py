from utils.config import load_config
from pathlib import Path
from data_generation.network_generator import generate_instance
def test_instance():
 i=generate_instance(load_config(Path(__file__).resolve().parents[1])); assert i["bus_trips"]==36
