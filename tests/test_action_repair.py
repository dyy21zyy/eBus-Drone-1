from src.env.environment import EBusDroneEnv, Event

def make_env():
    cfg={"bus_ids":["b1"],"station_ids":["h1"],"bus_battery_init":{"b1":99.9},"bus_passenger_init":{},"bus_parcel_init":{},"charger_count":{"h1":0},"station_power_limit":{"h1":300},"passenger_queue_init":{},"idle_drones_init":{"h1":0},"full_battery_init":{"h1":0},"depleted_battery_init":{"h1":0},"e_max":100.0,"eta_e":1.0,"p_chg":3600.0,"rho_al":1.0,"rho_bo":1.0,"rho_f":1.0}
    return EBusDroneEnv(cfg,[Event(0,'b1','h1',True,1,1,1,0,10,0,0.1)])

def test_repair_and_mask():
    env=make_env(); env.reset(); m=env.get_action_mask(); assert m[0]==1 and m[1:].sum()==0
    _,_,done,_,info=env.step(8)
    assert info['action_repaired'] is True
    assert info['executed_action_index']==0
