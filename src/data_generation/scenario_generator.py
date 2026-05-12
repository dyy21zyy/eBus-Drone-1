import random

def generate_scenario(instance,seed=0):
    rng=random.Random(seed)
    horizon=instance['horizon_min']
    base={h:[rng.uniform(50,150) for _ in range(horizon)] for h in instance['stations']}
    alight_prob={s:rng.uniform(0.05,0.25) for s in range(1,instance['n_stops']+1)}
    return {'base_load_kw':base,'alight_prob':alight_prob,'seed':seed}
