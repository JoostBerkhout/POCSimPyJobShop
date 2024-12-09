import time

import numpy as np

from pyjobshop import Result
from pyjobshop.simheuristic.DiscreteRV import Constant
from pyjobshop.simheuristic.modeling import fix_solution
from pyjobshop.simheuristic.problems.SingleMachineProblem import (
    SingleMachineProblem,
)
from pyjobshop.simheuristic.Simulator import Simulator

problem = SingleMachineProblem()


def print_results(_result: Result, _duration: float) -> None:
    print(f"\nObjective value = {_result.objective}")
    schedule = list(np.argsort([t.start for t in _result.best.tasks]))
    print(f"Schedule = {schedule}")
    start_times = [_result.best.tasks[t].start for t in schedule]
    print(f"Start times of tasks = {start_times}")
    print(f"The code took {round(_duration, 3)} seconds to run.")


# motivation for having a constant instead of a probability one DiscreteRV
for value in [0, 10, 15]:
    rv = Constant(value)
    assert rv.rvs()[0] == value
    assert rv.mean() == value
    assert rv.var() == 0
start = time.time()
for _ in range(1000):
    rv.rvs()
print("Time taken for 1000 rvs: ", time.time() - start)
start = time.time()
for _ in range(1000):
    b = value
print("Time taken for 1000 value: ", time.time() - start)


# test the model_builder (without solution given)
start_time = time.time()
data_generator = problem.build_data_generator()
data = data_generator.int_mean()
model = problem.concrete_model(data)
result = model.solve(display=False)
duration = time.time() - start_time
print_results(result, duration)

# test the model_builder (with solution given)
start_time = time.time()
data = data_generator.int_mean()
model = problem.concrete_model(data)
model = fix_solution(result.best, model)
result2 = model.solve(display=False)
duration = time.time() - start_time
assert result.best == result2.best
assert result.objective == result2.objective
print_results(result, duration)

# simulate solution
simulator = Simulator(problem, result.best)
start_time = time.time()
num_sims = 10
simulator.simulate(num_sims)
duration = time.time() - start_time
print(f"\nThe {num_sims} sims. took {round(duration, 3)} seconds to run.")
print(f"Number of simulation runs = {simulator.num_sims}")
print(f"Mean of simulation results = {simulator.mean}")
print(f"Variance of simulation results = {simulator.variance}")
print(f"Simulation results = {simulator.results}")
