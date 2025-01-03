import wandb
import numpy as np

from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.problems.HybridFlowShopGeneric import HybridFlowShop
from pyjobshop.simheuristic.utils import save_elite_solutions_to_csv

"""
This is a static / standard SimHeuristic implementation
"""


def run_dcop(config):
    np.random.seed(config["seed"])
    data_list = []
    use_wandb = False
    save_to_csv = False
    project_name = "simheuristics-pyjobshop"
    method_name = "dcop"
    problem_name = "HybridFlowShop"

    if use_wandb:
        wandb.init(
            project=project_name,  # where it will be logged
            name=method_name,  # name of the run
            config=config,  # log config
        )

    # Create a model
    problem = HybridFlowShop(num_jobs=config["num_jobs"], num_stages=config["num_stages"], seed=config["seed"])
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

    simulator_best_sol = Simulator(problem, result.best)
    simulator_best_sol.simulate(config["num_sims_long"])

    print(f'Mean value after long simulation {simulator_best_sol.mean}')

    if use_wandb:
        wandb.log({"final_best": simulator_best_sol.mean})

    if use_wandb:
        wandb.finish()

    data_list.append({
        "time_limit": config["time_limit"],
        "num_jobs": config["num_jobs"],
        "num_stages": config["num_stages"],
        "num_sims_long": config["num_sims_long"],
        "num_jobs": config["num_jobs"],
        "num_stages": config["num_stages"],
        "best_scop": simulator_best_sol.mean,
        "method": "dcop",
        "seed": config["seed"]
    })
    return data_list