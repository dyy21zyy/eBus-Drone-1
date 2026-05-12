def compute_dwell(board,alight,unload_kg,charge_sec):
 return max(board*3.0,alight*1.5,unload_kg*6.0,charge_sec)
