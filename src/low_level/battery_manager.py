def charge_step(full,depleted,max_simul=3):
 x=min(depleted,max_simul); return full+x,depleted-x
