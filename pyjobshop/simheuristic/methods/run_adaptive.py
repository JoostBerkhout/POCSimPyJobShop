import time
import pandas as pd
import numpy as np
import wandb

from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.modeling import find_solution_for_other_data
from pyjobshop.simheuristic.problems.HybridFlowShopGeneric import HybridFlowShop
from pyjobshop.simheuristic.EliteSolutions import EliteSolutions

"""
An adaptive SimHeuristic implementation will generate new solutions with different p-quantile settings
and simulate them given a fixed simulation budget
"""

# TODO: do we still want to implement a RouletteWheel like implementation?


def run_adaptive(config):
    # Set seed for reproduciblity
    np.random.seed(config['seed'])
    use_wandb = False
    data_list = []
    project_name = "simheuristics-sensitivity"
    problem_name = "HybridFlowShop"

    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if use_wandb:
        # Init wandb
        wandb.init(
            project=project_name,  # where it will be logged
            name="adaptive",  # name of the run
            config=config,  # log config
        )

    # Set problem
    problem = HybridFlowShop(num_jobs=config["num_jobs"], num_stages=config["num_stages"], seed=config["seed"])
    assert problem.__class__.__name__ == problem_name, "Set correct problem"
    data_generator = problem.build_data_generator()

    # Set concrete data to consider
    concrete_data = {}
    if config["consider_mean"]:
        concrete_data["mean"] = data_generator.int_mean()
    for quantile in quantiles:
        concrete_data[f"quantile_{quantile}"] = data_generator.quantile(quantile)

    # Set init scores
    scores = {k: config["init_score"] for k in concrete_data.keys()}

    # Technical init
    best_objective_elite = np.inf
    worst_objective_elite = np.inf
    elite_set = EliteSolutions()
    start_time = time.time()
    time_spend = time.time() - start_time
    current_solution = None
    old_data_key = None
    data_keys = []
    while time_spend < config["time_limit"]:
        # Randomly select data based on scores
        probs = np.array(list(scores.values())) / sum(scores.values())
        data_key = np.random.choice(list(scores.keys()), p=list(probs))
        #print(f'data key chosen {data_key}')
        data_keys.append(data_key)

        # Update data if needed
        new_data = data_key != old_data_key
        if new_data:
            old_data_key = data_key
            data = concrete_data[data_key]

            if current_solution is not None:
                # Update current solution for new data
                current_solution = find_solution_for_other_data(
                    current_solution, problem, data
                )

        # Solve the problem while warmstarting with the current solution
        model = problem.concrete_model(data)
        """
        Perhaps it is faster to store the concrete models instead of the data
        and creating the concrete models every time?
        """
        time_limit = min(
            config["time_limit"] - time_spend,
            config["max_time_per_cp_solve"],
        )
        result = model.solve(
            callback=None,
            display=False,
            initial_solution=current_solution,
            enumerate_all_solutions=config["enumerate"],  # stimulates finding more solutions
            time_limit=time_limit,
        )

        candidate_solution = result.best
        candidate_objective = result.objective
        candidate_simulator = Simulator(problem, candidate_solution)
        candidate_simulator.simulate(config["num_sims"])

        #print(f'The mean of the candidate solution is {candidate_simulator.mean}')

        # Update elite set
        elite_set.add(candidate_solution, candidate_simulator, candidate_objective)

        # Update scores
        best_obj = elite_set.get_best_elite_solution().simulator.mean
        #print(f'Best obj {best_obj}')
        worst_obj = elite_set.get_worst_elite_solution().simulator.mean
        #print(f'Worst obj {worst_obj}')

        if best_obj < best_objective_elite:
            scores[data_key] += config["score_finding_new_best"]
            best_objective_elite = best_obj
            worst_objective_elite = worst_obj  # by definition new worst
        elif worst_obj < worst_objective_elite:
            scores[data_key] += config["score_finding_new_elite"]
            worst_objective_elite = worst_obj
        #print(f'New scores {scores}')
        time_spend = time.time() - start_time

    best_stoch_solution = elite_set.get_best_elite_solution().solution
    simulator_best_sol = Simulator(problem, best_stoch_solution)
    simulator_best_sol.simulate(config["num_sims_long"])

    print(f'Mean value after long simulation {simulator_best_sol.mean}')
    print(f'Used data keys {data_keys}')
    # TODO: now we need to select the best solution from the elite set and then we take the best one at the end
    if use_wandb:
        wandb.log({"final_best": simulator_best_sol.mean})
        wandb.finish()

    data_list.append({
            "num_jobs": config["num_jobs"],
            "num_stages": config["num_stages"],
            "num_sims": config["num_sims"],
            "max_time_per_cp_solve": config["max_time_per_cp_solve"],
            "time_limit": config["time_limit"],
            "consider_mean": int(True),
            "num_sims_long": config["num_sims_long"],
            "method": "adaptive",
             "seed": config["seed"]
        })
    return data_list