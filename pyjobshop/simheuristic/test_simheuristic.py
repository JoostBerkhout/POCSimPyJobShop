from pyjobshop.simheuristic.problems.SingleMachineProblem import (
    SingleMachineProblem,
)
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback

# Create a model
problem = SingleMachineProblem()
data_generator = problem.build_data_generator()
data = data_generator.quantile(0.6)
model = problem.concrete_model(data)

# Solve the problem with a callback
callback = SolutionCallback(problem, num_sims=25)
result = model.solve(
    callback=callback,
    display=False,
    enumerate_all_solutions=True,  # stimulates finding more solutions
    time_limit=360,
)
print(result.status)
print(result.best.tasks)

callback.solutions.print_summary()
