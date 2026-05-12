from __future__ import annotations
import yaml
from pathlib import Path

def load_yaml(path:str)->dict:
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def deep_update(a:dict,b:dict)->dict:
    out=dict(a)
    for k,v in b.items():
        if isinstance(v,dict) and isinstance(out.get(k),dict): out[k]=deep_update(out[k],v)
        else: out[k]=v
    return out

def load_config(base_path:str, instance:str|None=None, method_path:str|None=None)->dict:
    cfg=load_yaml(base_path)
    if instance:
        cfg=deep_update(cfg, load_yaml(f'configs/instances/{instance}.yaml'))
    if method_path:
        cfg=deep_update(cfg, load_yaml(method_path))
    return cfg
