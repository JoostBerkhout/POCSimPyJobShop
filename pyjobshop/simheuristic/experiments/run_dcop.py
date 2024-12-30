import matplotlib.pyplot as plt
import wandb

from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.plot import plot_machine_gantt
from pyjobshop.simheuristic.problems.HybridFlowShopLarge import HybridFlowShopLarge as HybridFlowShop
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback
from pyjobshop.simheuristic.utils import save_elite_solutions_to_csv

"""
This is a static / standard SimHeuristic implementation
"""

use_wandb = True
save_to_csv = False
project_name = "simheuristics-pyjobshop"
method_name = "dcop"
problem_name = "HybridFlowShopLarge"

config: dict[str, int] = {
    "num_sims": 25,
    "time_limit": 300,
    "num_sims_long": 5000
}

if use_wandb:
    wandb.init(
        project=project_name,  # where it will be logged
        name=method_name,  # name of the run
        config=config,  # log config
    )

# Create a model
problem = HybridFlowShop()
assert problem.__class__.__name__ == problem_name, "Set correct problem"
data_generator = problem.build_data_generator()
data = data_generator.int_mean()
model = problem.concrete_model(data)

# Solve the problem with a callback
callback = None
result = model.solve(
    callback=callback,
    display=False,
    enumerate_all_solutions=True,  # stimulates finding more solutions
    time_limit=config["time_limit"],
)
print("Solver status:", result.status)
print("Best objective value:", result.objective)

if save_to_csv:
    save_elite_solutions_to_csv(callback.solutions, problem_name)

# Plot solutions, for DCOP SimHeuristic we use the best deterministic solution
plot_solutions = {
    "Best solution for deterministic problem": result.best,
    "Best solution for stochastic problem": result.best,
}
simulator_best_sol = Simulator(problem, result.best)
simulator_best_sol.simulate(config["num_sims_long"])

print(f'Mean value after long simulation {simulator_best_sol.mean}')
# TODO: simulate for 5000 simulations to obtain true value
for title, solution in plot_solutions.items():
    plot_machine_gantt(solution, model.data(), title=title, plot_labels=True)
    plt.show()

if use_wandb:
    wandb.log({"final_best": simulator_best_sol.mean})

if use_wandb:
    wandb.finish()