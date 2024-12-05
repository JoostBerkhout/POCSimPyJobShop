import time

import numpy as np
from DataGeneratorBuilder import DataGeneratorBuilder

from pyjobshop import Result
from pyjobshop.simheuristic.DiscreteRV import Constant
from pyjobshop.simheuristic.EliteSolutions import EliteSolutions
from pyjobshop.simheuristic.evaluator import evaluator
from pyjobshop.simheuristic.modeling import (
    fix_solution,
    model_builder,
    model_builder_fix_sol,
)
from pyjobshop.simheuristic.problems.single_machine import (
    single_machine_data,
    single_machine_model,
)
from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback

problem_data = single_machine_data
problem_model = single_machine_model


def print_results(_result: Result, _duration: float) -> None:
    print(f"\nObjective value = {_result.objective}")
    schedule = list(np.argsort([t.start for t in _result.best.tasks]))
    print(f"Schedule = {schedule}")
    start_times = [_result.best.tasks[t].start for t in schedule]
    print(f"Start times of tasks = {start_times}")
    print(f"The code took {round(_duration, 3)} seconds to run.")


# test DiscreteRV
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

# test the DataGeneratorBuilder
builder = DataGeneratorBuilder(problem_data)
data_generator = builder.build()
print("Random data: ", data_generator.random())
print("Mean data: ", data_generator.int_mean())
data_generator = builder.build()
print("Random data: ", data_generator.random())
print("Mean data: ", data_generator.int_mean())

# test the model_builder (without solution given)
start_time = time.time()
data = data_generator.int_mean()
model = model_builder_fix_sol(data)
model.set_objective(weight_total_tardiness=1000, weight_makespan=1)
result = model.solve(display=False)
duration = time.time() - start_time
print_results(result, duration)

# test the model_builder (with solution given)
start_time = time.time()
data = data_generator.int_mean()
model = model_builder_fix_sol(data, solution=result.best)
model.set_objective(weight_total_earliness=1000, weight_makespan=1)
result = model.solve(display=False)
duration = time.time() - start_time
print_results(result, duration)

# test the model_builder (without solution given)
start_time = time.time()
data = data_generator.int_mean()
model = model_builder(problem_model, data)
model.set_objective(weight_total_tardiness=1000, weight_makespan=1)
result = model.solve(display=False)
duration = time.time() - start_time
print_results(result, duration)

# test the model_builder (with solution given)
start_time = time.time()
data = data_generator.int_mean()
model = model_builder(problem_model, data)
model = fix_solution(result.best, model)
model.set_objective(weight_total_earliness=1000, weight_makespan=1)
result = model.solve(display=False)
duration = time.time() - start_time
print_results(result, duration)

# simulate solution
simulator = Simulator(problem_model, data_generator, result.best, evaluator)
start_time = time.time()
simulator.simulate(10)
duration = time.time() - start_time
print(f"\nThe simulation took {round(duration, 3)} seconds to run.")
print(f"Number of simulation runs = {simulator.num_sims}")
print(f"Mean of simulation results = {simulator.mean}")
print(f"Variance of simulation results = {simulator.variance}")
print(f"Simulation results = {simulator.results}")

# test EliteSolution
solutions = EliteSolutions()
solutions.add(result.best, simulator, result.objective)
print(solutions.get_best_solution())

print("do I come here?")

# test callback
start_time = time.time()
data = data_generator.int_mean()
model = model_builder(problem_model, data)
model.set_objective(weight_total_earliness=1000, weight_makespan=1)
num_sims = 7
callback = SolutionCallback(problem_model, builder, evaluator, num_sims)
result = model.solve(callback=callback, display=False)
duration = time.time() - start_time
print_results(result, duration)
callback.solutions.print_summary()

# test whether simulation results coincide
for solution, simulator, objective in callback.solutions.solutions.values():
    data = data_generator.int_mean()
    model = model_builder(problem_model, data)
    model = fix_solution(solution, model)
    model.set_objective(weight_total_earliness=1000, weight_makespan=1)
    result = model.solve(display=False)
    assert result.objective == objective

    data_generator = builder.build()
    new_simulator = Simulator(
        problem_model, data_generator, solution, evaluator
    )
    new_simulator.simulate(num_sims)
    assert new_simulator.num_sims == simulator.num_sims
    assert new_simulator.mean == simulator.mean
    assert new_simulator.variance == simulator.variance

    new_simulator.simulate(3)
    assert new_simulator.num_sims != simulator.num_sims
    assert new_simulator.mean != simulator.mean
    assert new_simulator.variance != simulator.variance

    simulator.simulate(3)
    assert new_simulator.num_sims == simulator.num_sims
    assert new_simulator.mean == simulator.mean
    assert new_simulator.variance == simulator.variance

print("Test passed: All simulation results coincide.")
