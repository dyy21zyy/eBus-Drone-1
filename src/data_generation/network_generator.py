
def generate_instance(cfg):
    inst=cfg['instance_cfg']
    return {
      'n_stops':cfg['network']['n_stops'],'stations':cfg['network']['stations'][:min(len(cfg['network']['stations']),8 if inst['customers']>30 else 6)],
      'customers':inst['customers'],'bus_trips':inst['bus_trips'],'headway_min':inst['headway_min'],'horizon_min':cfg['horizon_min']
    }
