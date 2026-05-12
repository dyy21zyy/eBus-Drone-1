from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.data.generation import generate_instance, save_instance
from src.env.environment import EBusDroneEnv, Event
from src.policies.rule_based import NoChargingPolicy, MaxFeasiblePolicy, UniformChargingPolicy
from src.rl.agents import DQNAgent

A_FULL=[0,15,30,45,60,75,90,105,120]

def make_env(cfg,instance):
    events=[Event(time=i*300,bus_id=instance['buses'][i%len(instance['buses'])],station_id=instance['stations'][i%len(instance['stations'])],is_integrated_station=True,chi=1,n_al=2,n_bo0=2,q_f=1,onboard_before=10,parcel_onboard_before=1,local_urgency=0.5) for i in range(20)]
    ecfg={"bus_ids":instance['buses'][:2],"station_ids":instance['stations'][:2],"bus_battery_init":{b:80.0 for b in instance['buses'][:2]},"bus_passenger_init":{},"bus_parcel_init":{},"charger_count":{s:1 for s in instance['stations'][:2]},"station_power_limit":{s:300.0 for s in instance['stations'][:2]},"passenger_queue_init":{},"idle_drones_init":{s:2 for s in instance['stations'][:2]},"full_battery_init":{s:2 for s in instance['stations'][:2]},"depleted_battery_init":{s:0 for s in instance['stations'][:2]},"e_max":100.0,"eta_e":0.95,"p_chg":120.0,"rho_al":2.0,"rho_bo":2.0,"rho_f":4.0,"travel_consumption_per_event":2.0,"alpha_1":1,"alpha_2":1,"alpha_3":1,"alpha_4":1,"alpha_5":1,"alpha_6":1,"E_min":10.0}
    return EBusDroneEnv(ecfg,events)

def evaluate_policy(env,policy):
    obs,info=env.reset(); done=False; total=0.0; repaired=0; sel=[]; exe=[]; infeasible=0; steps=0
    while not done:
        mask=env.get_action_mask(); a=policy.select_action(obs,mask,info={"action_set":A_FULL})
        if mask[a]==0: infeasible+=1
        nobs,r,done,_,inf=env.step(a); total+=r; repaired += int(inf.get('action_repaired',False)); obs=nobs; steps+=1; sel.append(A_FULL[a]); exe.append(inf.get('executed_duration',A_FULL[a]))
    return {"total_weighted_cost":-total,"infeasible_action_rate": infeasible/max(1,steps),"action_repair_rate": repaired/max(1,steps),"avg_selected_charging_duration":sum(sel)/max(1,len(sel)),"avg_executed_charging_duration":sum(exe)/max(1,len(exe))}

