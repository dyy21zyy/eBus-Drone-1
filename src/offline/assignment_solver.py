
def solve_assignment(instance):
    stations=instance['stations']
    return {i:stations[i%len(stations)] for i in range(instance['customers'])}
