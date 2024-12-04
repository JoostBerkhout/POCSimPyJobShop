import time

import numpy as np
from DataGeneratorBuilder import DataGeneratorBuilder

from pyjobshop import Result
from pyjobshop.simheuristic.modeling import (
    fix_solution,
    model_builder,
    model_builder_fix_sol,
)


def print_results(result: Result, duration: float) -> None:
    print(f"\nObjective value = {result.objective}")
    schedule = list(np.argsort([t.start for t in result.best.tasks]))
    print(f"Schedule = {schedule}")
    start_times = [result.best.tasks[t].start for t in schedule]
    print(f"Start times of tasks = {start_times}")
    print(f"The code took {round(duration, 3)} seconds to run.")


# test the DataGeneratorBuilder
builder = DataGeneratorBuilder()
data_generator = builder.build()
print(data_generator.random())
print(data_generator.int_mean())

# test the model_builder (without solution given)
# time the following code snippet
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

# test the model_builder (with solution given)
start_time = time.time()
data = data_generator.int_mean()
model = model_builder(data)
model = fix_solution(result.best, model)
model.set_objective(weight_total_earliness=1000, weight_makespan=1)
result = model.solve(display=False)
duration = time.time() - start_time
print_results(result, duration)
