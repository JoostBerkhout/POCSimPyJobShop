import wandb
import numpy as np
import time

from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.problems.HybridFlowShopGeneric import HybridFlowShop
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback
from pyjobshop.simheuristic.utils import save_elite_solutions_to_csv

"""
This is a static / standard SimHeuristic implementation
"""


def run_standard(config, use_wandb=False):

    print(f'Start standard simheuristic at time {time.time()}')
    if use_wandb:
        wandb.init(
            project=config["project_name"],  # where it will be logged
            config=config,  # log config
        )
    # Technical init
    save_to_csv = False
    data_list = []
    np.random.seed(config["seed"])

    # Set problem
    if config["problem_name"] == "HybridFlowShop":
        problem = HybridFlowShop(num_jobs=config["num_jobs"], num_stages=config["num_stages"], seed=config["seed"])
    else:
        ValueError(f'Unknown problem: {config["problem_name"]}')

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
    #callback.solutions.print_summary()
    if save_to_csv:
        save_elite_solutions_to_csv(callback.solutions, problem_name)

    # Plot solutions
    best_stoch_solution = callback.solutions.get_best_elite_solution().solution

    simulator_best_sol = Simulator(problem, best_stoch_solution)
    simulator_best_sol.simulate(config["num_sims_long"])
    print(f'Mean value after long simulation {simulator_best_sol.mean}')

    if use_wandb:
        wandb.log({"final_best": simulator_best_sol.mean})

    if use_wandb:
        wandb.finish()

    data_list.append({"num_sims": config["num_sims"],
        "time_limit": config["time_limit"],
        "num_jobs": config["num_jobs"],
        "num_stages": config["num_stages"],
        "num_sims_long": config["num_sims_long"],
        "best_scop": simulator_best_sol.mean,
        "method": "standard",
        "seed": config["seed"]
                      })

    return data_list