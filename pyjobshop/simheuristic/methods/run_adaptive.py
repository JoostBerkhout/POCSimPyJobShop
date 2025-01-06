import time
import pandas as pd
import numpy as np
import wandb
import numpy.random as rnd

from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.modeling import find_solution_for_other_data
from pyjobshop.simheuristic.problems.HybridFlowShopGeneric import HybridFlowShop
from pyjobshop.simheuristic.EliteSolutions import EliteSolutions
from pyjobshop.simheuristic.RouletteWheel import RouletteWheel
from pyjobshop.simheuristic.Outcome import Outcome

"""
An adaptive SimHeuristic implementation will generate new solutions with different p-quantile settings
and simulate them given a fixed simulation budget
"""


def run_adaptive(config, use_wandb=False):

    print(f'Start adaptive simheuristic at {time.time()}')
    # Set seed for reproduciblity
    rnd_state = rnd.RandomState(config["seed"])

    if use_wandb:
        # Init wandb
        wandb.init(
            project=config["project_name"],  # where it will be logged
            config=config,  # log config
        )

    # Set problem
    if config["problem_name"] == "HybridFlowShop":
        problem = HybridFlowShop(num_jobs=config["num_jobs"], num_stages=config["num_stages"], seed=config["seed"])
    else:
        ValueError(f'Unknown problem: {config["problem_name"]}')

    data_generator = problem.build_data_generator()

    # Set concrete data to consider
    strategies = config["strategies"]
    concrete_data = {}
    if config["consider_mean"]:
        concrete_data["mean"] = data_generator.int_mean()
    for quantile in strategies:
        concrete_data[f"quantile_{quantile}"] = data_generator.quantile(quantile)
    strategies = [k for k in concrete_data.keys()]

    # Technical init
    select = RouletteWheel(scores=[20, 5, 0.5], decay=0.8, num_strategies=len(strategies))
    current_objective = np.inf
    best_objective_elite = np.inf
    worst_objective_elite = np.inf
    elite_set = EliteSolutions()
    start_time = time.time()
    time_spend = time.time() - start_time
    current_solution = None
    old_data_key = None
    strategy_history = []
    data_list = []

    while time_spend < config["time_limit"]:

        # Select a strategy
        s_idx = select(rnd_state)
        data_key = strategies[s_idx]
        strategy_history.append(data_key)

        # Update data if needed
        new_data = data_key != old_data_key
        if new_data:
            old_data_key = data_key
            data = concrete_data[data_key]

            if current_solution is not None:
                # Update current solution for new data
                current_solution, current_objective = find_solution_for_other_data(
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

        if current_solution is None:
            current_solution = candidate_solution
            current_objective = result.objective
            # Update elite set
            elite_set.add(candidate_solution, candidate_simulator, candidate_objective)
        elif candidate_objective < current_objective:
            current_solution = candidate_solution
            current_objective = candidate_objective

            # Update elite set
            elite_set.add(candidate_solution, candidate_simulator, candidate_objective)
            outcome = Outcome.BETTER
        else:
            outcome = Outcome.REJECT

        # Update scores
        best_obj = elite_set.get_best_elite_solution().simulator.mean
        worst_obj = elite_set.get_worst_elite_solution().simulator.mean

        if best_obj < best_objective_elite:
            best_objective_elite = best_obj
            worst_objective_elite = worst_obj  # by definition new worst
            outcome = Outcome.BEST
        elif worst_obj < worst_objective_elite:
            worst_objective_elite = worst_obj

        select.update(s_idx, outcome)
        time_spend = time.time() - start_time

    best_stoch_solution = elite_set.get_best_elite_solution().solution
    simulator_best_sol = Simulator(problem, best_stoch_solution)
    simulator_best_sol.simulate(config["num_sims_long"])

    print(f'Mean value after long simulation {simulator_best_sol.mean}')
    print(f'Used strategies {strategy_history}')
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
            "seed": config["seed"],
            "best_scop": simulator_best_sol.mean
        })
    return data_list