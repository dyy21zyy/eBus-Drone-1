from data_generation.scenario_generator import generate_scenario
def test_repro():
 i={"horizon_min":2,"stations":[1],"n_stops":2};a=generate_scenario(i,7);b=generate_scenario(i,7);assert a==b
