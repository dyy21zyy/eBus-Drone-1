from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

@dataclass
class InstanceData:
    customers:int; stations:int; bus_trips:int; headway_min:int; passenger_capacity:int; arrival_rate_per_min:float


def generate_instance(cfg:dict, seed:int)->dict:
    rng=np.random.default_rng(seed)
    n_c=cfg['instance']['customers']; n_s=cfg['instance']['stations']; n_b=cfg['instance']['bus_trips']
    cust=[f'c{i}' for i in range(n_c)]; st=[f'h{i}' for i in range(n_s)]; buses=[f'b{i}' for i in range(n_b)]
    feasible={(c,h): int(rng.random()<0.6) for c in cust for h in st}
    for c in cust:
        if sum(feasible[(c,h)] for h in st)==0: feasible[(c,st[int(rng.integers(0,n_s))])]=1
    return {'customers':cust,'stations':st,'buses':buses,'feasible_customer_station':feasible,'arrival_rate_per_min':cfg['passenger']['arrival_rate_per_min']}


def save_instance(data:dict, path:str)->None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    serial=dict(data); serial['feasible_customer_station']={f'{k[0]}|{k[1]}':v for k,v in data['feasible_customer_station'].items()}
    p.write_text(json.dumps(serial,indent=2),encoding='utf-8')


def load_instance(path:str)->dict:
    raw=json.loads(Path(path).read_text())
    raw['feasible_customer_station']={tuple(k.split('|')):v for k,v in raw['feasible_customer_station'].items()}
    return raw
