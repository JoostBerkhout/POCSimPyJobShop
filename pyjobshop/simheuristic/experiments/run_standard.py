import matplotlib.pyplot as plt
import wandb
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
save_to_csv = False
data_list = []
project_name = "simheuristics-sensitivity"
method_name = "standard"
problem_name = "HybridFlowShop"

for (j, k) in [(30, 30)]:
    for beta in [60, 300, 600]:
        for eta in [25, 50, 100]:
            config: dict[str, int] = {
                "num_sims": eta,
                "time_limit": beta,
                "num_sims_long": 5000,
                "enumerate": 0,
                "num_jobs": j,
                "num_stages": k
            }

            if use_wandb:
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
            callback = SolutionCallback(problem, num_sims=config["num_sims"])
            result = model.solve(
                callback=callback,
                display=False,
                enumerate_all_solutions=config["enumerate"],  # stimulates finding more solutions
                time_limit=config["time_limit"],
            )
            print("Solver status:", result.status)
            print("Best objective value:", result.objective)
            callback.solutions.print_summary()
            if save_to_csv:
                save_elite_solutions_to_csv(callback.solutions, problem_name)

            # Plot solutions
            best_stoch_solution = callback.solutions.get_best_elite_solution().solution
            plot_solutions = {
                "Best solution for deterministic problem": result.best,
                "Best solution for stochastic problem": best_stoch_solution,
            }
            simulator_best_sol = Simulator(problem, best_stoch_solution)
            simulator_best_sol.simulate(config["num_sims_long"])
            print(f'Mean value after long simulation {simulator_best_sol.mean}')

            if use_wandb:
                wandb.log({"final_best": simulator_best_sol.mean})

            if use_wandb:
                wandb.finish()

            data_list.append({ "num_sims": eta,
                "time_limit": beta,
                "num_jobs": j,
                "num_stages": k,
                "final_best": simulator_best_sol.mean,
                "method": "standard"})

            data_df = pd.DataFrame(data_list)
            data_df.to_csv("results/sensitivity_standard.csv")