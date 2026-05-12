from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from src.utils.config import load_config
from src.utils.seeding import set_seed
from src.data.generation import generate_instance, save_instance
from src.harness.runner import make_env, evaluate_policy
from src.policies.rule_based import NoChargingPolicy, MaxFeasiblePolicy, UniformChargingPolicy

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',required=True); p.add_argument('--config',required=True); p.add_argument('--instance',default='small'); p.add_argument('--method',default='proposed'); p.add_argument('--seed',type=int,default=1); p.add_argument('--seeds',nargs='*',type=int)
    a=p.parse_args(); cfg=load_config(a.config,a.instance); set_seed(a.seed)
    out=Path(cfg['paths']['outputs']); out.mkdir(parents=True,exist_ok=True)
    if a.mode=='generate':
        inst=generate_instance(cfg,a.seed); save_instance(inst,out/f'instance_{a.instance}_{a.seed}.json'); return
    inst=generate_instance(cfg,a.seed); env=make_env(cfg,inst)
    if a.mode in {'eval','benchmark','ablation','sensitivity','all','train'}:
        pol=NoChargingPolicy() if a.method=='no_charging' else MaxFeasiblePolicy()
        if a.method.startswith('uniform_'): pol=UniformChargingPolicy(int(a.method.split('_')[1]))
        m=evaluate_policy(env,pol)
        pd.DataFrame([m]).to_csv(out/f'metrics_{a.mode}_{a.method}_{a.seed}.csv',index=False)
        (out/f'summary_{a.mode}_{a.method}_{a.seed}.json').write_text(json.dumps(m,indent=2))
    if a.mode=='offline':
        from offline_preassignment import save_assignment_json
        save_assignment_json({('b0','h0','c0'):1}, out/'assignments/offline_assignment_plan.json')

if __name__=='__main__': main()
