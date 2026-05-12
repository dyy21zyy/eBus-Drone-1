from low_level.battery_manager import charge_step
def test_batt(): assert charge_step(0,2)==(2,0)
