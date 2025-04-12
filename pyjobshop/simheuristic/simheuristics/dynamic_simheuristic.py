import time
from typing import List, Tuple, TypedDict

import numpy as np
import wandb

from pyjobshop import Result
from pyjobshop.simheuristic.modeling import (
    find_solution_for_other_concrete_model,
)
from pyjobshop.simheuristic.problems.Problem import Problem
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback
from pyjobshop.simheuristic.utils import init_wandb


class DynamicSimheuristicConfig(TypedDict):
    num_sims: int
    max_size_elite_set: int
    score_finding_new_elite: int
    score_finding_new_best: int
    init_score: int
    max_time_per_cp_solve: int
    consider_mean: bool
    quantiles: List[float]
    frac_budget_before_sims: float
    frac_budget_final_elites_sim: float


def parse_model_key(model_key: str) -> float:
    if "_" in model_key:
        try:
            return float(model_key.split("_")[-1])
        except ValueError:
            pass  # fallback to default below if conversion fails
    return 0.0


def log_model(callback: SolutionCallback, model_key: str) -> None:
    wandb.log(
        {
            "Time (in seconds)": callback.time_spent,
            "Deterministic model": parse_model_key(model_key),
        }
    )


def select_model_key(scores):
    keys, values = zip(*scores.items())
    probs = np.array(values) / sum(values)
    return np.random.choice(keys, p=probs)


def dynamic_simheuristic(
    problem: Problem,
    simh_config: DynamicSimheuristicConfig,
    exp_config: dict[str, int],
    use_wandb: bool = False,
    wandb_config: dict[str, str] | None = None,
) -> Tuple[SolutionCallback, Result, dict[str, float]]:
    """
    Runs a dynamic simheuristic on the given problem.

    Parameters
    ----------
    problem : Problem
        The optimization problem instance (e.g., HybridFlowShop).
    simh_config : dict[str, int]
        Configuration for the simheuristic (e.g., number of simulations).
    exp_config : dict[str, int]
        Configuration for the experiment (e.g., time limit).
    use_wandb : bool, optional
        Whether to log progress to Weights & Biases, by default False.
    wandb_config : dict[str, str], optional
        Configuration for Weights & Biases logging, by default None.

    Returns
    -------
    callback : SolutionCallback
        A callback object that contains the found elite solutions.
    result : Result
        A Result object containing the best found solution and additional
        information about the solver run.
    durations : dict[str, float]
        The durations of the experiment phases in seconds. It does not include
        the loading and closing duration of wandb if used.
    """

    if use_wandb:
        problem_name = problem.__class__.__name__
        save_config = {"exp_config": exp_config, "simh_config": simh_config}
        init_wandb(problem_name, save_config, wandb_config)

    start_time_exp = time.time()
    durations = {}

    # Set concrete models
    concrete_models = {}
    data_generator = problem.build_data_generator()
    if simh_config["consider_mean"]:
        mean_data = data_generator.int_mean()
        concrete_models["mean"] = problem.concrete_model(mean_data)
    for quantile in simh_config["quantiles"]:
        model_name = f"quantile_{quantile}"
        quant_data = data_generator.quantile(quantile)
        concrete_models[model_name] = problem.concrete_model(quant_data)

    # Set init scores
    scores = {k: simh_config["init_score"] for k in concrete_models.keys()}

    # Init
    time_limit = exp_config["time_limit"]
    num_sims = simh_config["num_sims"]
    start_time_sims = simh_config["frac_budget_before_sims"] * time_limit
    max_size_elite_set = simh_config["max_size_elite_set"]
    time_final_sims = simh_config["frac_budget_final_elites_sim"] * time_limit
    solve_time_limit = time_limit - time_final_sims

    # Technical init
    best_objective_elite = np.inf
    worst_objective_elite = np.inf
    callback = SolutionCallback(
        problem=problem,
        num_sims=num_sims,
        start_time_exp=start_time_exp,
        start_time_sims=time.time() + start_time_sims,
        max_size_elite_set=max_size_elite_set,
        stop_time=None,  # set later
    )
    current_solution = None
    old_model_key = None
    time_spend = time.time() - start_time_exp

    while time_spend < solve_time_limit:
        # Randomly select model based on scores
        probs = np.array(list(scores.values())) / sum(scores.values())
        model_key = np.random.choice(list(scores.keys()), p=list(probs))

        # Update model if needed
        new_model = model_key != old_model_key
        if new_model:
            old_model_key = model_key
            model = concrete_models[model_key]

            if current_solution is not None:
                # Update current solution for new model
                current_solution = find_solution_for_other_concrete_model(
                    current_solution, model
                )
        log_model(callback, model_key)

        # Solve the problem using a callback for stochastic evaluations
        cp_time_limit = min(
            solve_time_limit - time_spend,
            simh_config["max_time_per_cp_solve"],
        )
        callback.stop_time = time.time() + cp_time_limit
        callback._log_event(f"Starting CP solver with {model_key}.")
        result = model.solve(
            callback=callback,
            display=False,
            initial_solution=current_solution,
            time_limit=cp_time_limit,
            num_workers=exp_config["num_workers"],
        )

        # Update scores
        solutions = callback.solutions
        best_obj = solutions.get_best_mean_solution().simulator.mean
        worst_obj = solutions.get_worst_mean_solution().simulator.mean
        if best_obj < best_objective_elite:
            scores[model_key] += simh_config["score_finding_new_best"]
            best_objective_elite = best_obj
            worst_objective_elite = worst_obj  # by definition new worst
        elif worst_obj < worst_objective_elite:
            scores[model_key] += simh_config["score_finding_new_elite"]
            worst_objective_elite = worst_obj

        time_spend = time.time() - start_time_exp

    callback.print_log()

    # Log times
    time_spent = time.time() - start_time_exp
    durations["solver_phase_dur"] = time_spent

    # Simulate the elite solutions for the remaining time
    callback.solutions.simulate_to_time_limit(time_limit - time_spent)

    # Log times
    exp_duration = time.time() - start_time_exp
    durations["final_sim_phase_dur"] = exp_duration - time_spent
    durations["total_exp_dur"] = exp_duration

    if use_wandb:
        wandb.finish()

    return callback, result, durations
