from offline.assignment_solver import solve_assignment
def test_assign():
 a=solve_assignment({"customers":5,"stations":[1,2]}); assert len(a)==5
