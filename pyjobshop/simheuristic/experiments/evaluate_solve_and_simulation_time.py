import matplotlib.pyplot as plt
import wandb
import time
import pandas as pd

from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.plot import plot_machine_gantt
from pyjobshop.simheuristic.problems.HybridFlowShopGeneric import HybridFlowShop
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback
from pyjobshop.simheuristic.utils import save_elite_solutions_to_csv

"""
This is a static / standard SimHeuristic implementation
"""

use_wandb = False
data_list = []
save_to_csv = False
project_name = "simheuristics-pyjobshop"
method_name = "dcop"
problem_name = "HybridFlowShop"

config: dict[str, int] = {
    "num_sims": 25,
    "time_limit": 3600,
    "num_sims_long": 1
}

for (j, k) in [(5, 5), (10, 5), (10, 10), (20, 10), (20, 20),
               (30, 15), (30, 30), (40, 20), (40, 40),
               (50, 25), (50, 50)]:

    print(f"\nStart with solving for (num_jobs, num_stages)={(j, k)}")
    if use_wandb:
        config["num_jobs"] = j
        config["num_stages"] = k
        wandb.init(
            project=project_name,  # where it will be logged
            name=method_name,  # name of the run
            config=config,  # log config
        )

    # Create a model
    problem = HybridFlowShop(num_jobs=j, num_stages=k)
    assert problem.__class__.__name__ == problem_name, "Set correct problem"
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)

    # Solve the problem with a callback
    callback = None
    start_solving = time.time()
    result = model.solve(
        callback=callback,
        display=False,
        enumerate_all_solutions=True,  # stimulates finding more solutions
        time_limit=config["time_limit"],
    )
    finish_solving = time.time()
    print("Solver status:", result.status)
    print("Best objective value:", result.objective)
    print("Solver time:", finish_solving - start_solving)

    if save_to_csv:
        save_elite_solutions_to_csv(callback.solutions, problem_name)

    # Plot solutions, for DCOP SimHeuristic we use the best deterministic solution
    plot_solutions = {
        "Best solution for deterministic problem": result.best,
        "Best solution for stochastic problem": result.best,
    }
    simulator_best_sol = Simulator(problem, result.best)
    start_simulation = time.time()
    simulator_best_sol.simulate(config["num_sims_long"])
    finish_simulation = time.time()

    print(f'Mean value after long simulation: {simulator_best_sol.mean}')
    print(f'Simulation time: {finish_simulation - start_simulation}')
    # TODO: simulate for 5000 simulations to obtain true value
    for title, solution in plot_solutions.items():
        plot_machine_gantt(solution, model.data(), title=title, plot_labels=True)
        plt.show()

    if use_wandb:
        wandb.log({"final_best": simulator_best_sol.mean})

    if use_wandb:
        wandb.finish()

    data_list.append({
                      "num_jobs": j, "num_stages": k,
                      "simulation_time": finish_simulation - start_simulation,
                      "solve_time": finish_solving - start_solving,
                      "solve_status": result.status
    })
    data_df = pd.DataFrame(data_list)
    data_df.to_csv("results/solve_and_simulation_time.csv")
