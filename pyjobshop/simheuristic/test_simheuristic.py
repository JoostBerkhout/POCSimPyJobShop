import matplotlib.pyplot as plt
import wandb

from pyjobshop.plot import plot_machine_gantt
from pyjobshop.simheuristic.problems.HybridFlowShop import HybridFlowShop
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback
from pyjobshop.simheuristic.utils import save_elite_solutions_to_csv

use_wandb = False
save_to_csv = False
project_name = "static-simheuristics-pyjobshop"
problem_name = "HybridFlowShop"
config: dict[str, int] = {
    "num_sims": 25,
    "time_limit": 60,
}

if use_wandb:
    wandb.init(
        project=project_name,  # where it will be logged
        name=problem_name,  # name of the run
        config=config,  # log config
    )

# Create a model
problem = HybridFlowShop(seed=3)
assert problem.__class__.__name__ == problem_name, "Set correct problem"
data_generator = problem.build_data_generator()
data = data_generator.int_mean()
model = problem.concrete_model(data)

# Solve the problem with a callback
callback = SolutionCallback(problem, num_sims=config["num_sims"])
result = model.solve(
    callback=callback,
    display=False,
    enumerate_all_solutions=True,  # stimulates finding more solutions
    time_limit=config["time_limit"],
)
print("Solver status:", result.status)
print("Best objective value:", result.objective)
callback.solutions.print_summary()
if save_to_csv:
    save_elite_solutions_to_csv(callback.solutions, problem_name)

if use_wandb:
    wandb.finish()

# Plot solutions
best_stoch_solution = callback.solutions.get_best_elite_solution().solution
plot_solutions = {
    "Best solution for deterministic problem": result.best,
    "Best solution for stochastic problem": best_stoch_solution,
}
for title, solution in plot_solutions.items():
    plot_machine_gantt(solution, model.data(), title=title, plot_labels=True)
    plt.show()
