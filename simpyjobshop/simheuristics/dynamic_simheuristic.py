import time
from typing import List, Tuple, TypedDict

import numpy as np
import wandb

from pyjobshop import Result
from simpyjobshop.modeling import find_solution_for_other_data
from simpyjobshop.problems import Problem
from simpyjobshop.SolutionCallback import SolutionCallback
from simpyjobshop.utils import init_wandb


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


def dynamic_simheuristic(
    problem: Problem,
    simh_config: DynamicSimheuristicConfig,
    exp_config: dict[str, int],
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

    if exp_config["use_wandb"]:
        problem_name = problem.__class__.__name__
        save_config = {"exp_config": exp_config, "simh_config": simh_config}
        init_wandb(problem_name, save_config, wandb_config)

    start_time_exp = time.time()
    durations = {}

    # Set concrete models
    concrete_models = {}
    data = {}
    data_generator = problem.build_data_generator()
    if simh_config["consider_mean"]:
        data["mean"] = data_generator.int_mean()
        concrete_models["mean"] = problem.concrete_model(data["mean"])
    for quantile in simh_config["quantiles"]:
        key = f"quantile_{quantile}"
        data[key] = data_generator.quantile(quantile)
        concrete_models[key] = problem.concrete_model(data[key])

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
    elite_tracker = {"worst mean": float("inf"), "best mean": float("inf")}
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
        model_key = select_weighted_random_key(scores)

        # Set model and update current solution if needed
        model = concrete_models[model_key]
        new_model = model_key != old_model_key
        old_model_used = old_model_key is not None
        update_current_solution = new_model and old_model_used
        old_model_key = model_key
        new_obj_val = None
        if update_current_solution:
            assert current_solution is not None
            current_solution, new_obj_val = find_solution_for_other_data(
                current_solution, problem, data[model_key]
            )
        if exp_config["use_wandb"]:
            log_model(callback, model_key, new_obj_val)

        # Solve the problem using a callback for stochastic evaluations
        cp_time_limit = min(
            solve_time_limit - time_spend,
            simh_config["max_time_per_cp_solve"],
        )
        callback.stop_time = time.time() + cp_time_limit
        callback._log_event(f"Starting CP solver with {model_key} ({scores}).")
        result = model.solve(
            callback=callback,
            display=False,
            initial_solution=current_solution,
            time_limit=cp_time_limit,
            num_workers=exp_config["num_workers"],
        )
        log_solve_status(callback, result)

        # Update current solution
        current_solution = result.best

        # Update scores
        update_scores(callback, elite_tracker, scores, model_key, simh_config)

        time_spend = time.time() - start_time_exp

    # Log times
    time_spent = time.time() - start_time_exp
    durations["solver_phase_dur"] = time_spent

    # Simulate the elite solutions for the remaining time
    callback.solutions.simulate_to_time_limit(time_limit - time_spent)

    # Log times
    exp_duration = time.time() - start_time_exp
    durations["final_sim_phase_dur"] = exp_duration - time_spent
    durations["total_exp_dur"] = exp_duration

    if exp_config["use_wandb"]:
        wandb.finish()

    return callback, result, durations


def parse_solve_status(solve_status: str) -> int:
    """
    Helper function to change status to an integer for wandb plot.
    """
    if solve_status == "Optimal":
        return 0
    elif solve_status == "Feasible":
        return 1
    elif solve_status == "Infeasible":
        return 2
    elif solve_status == "Time-limit":
        return 3
    elif solve_status == "Unknown":
        return 4
    else:
        raise ValueError(f"Unknown solve status: {solve_status}")


def parse_model_key(model_key: str) -> float:
    """
    Helper function to parse the model key to extract a float value.
    """
    if "_" in model_key:
        try:
            return float(model_key.split("_")[-1])
        except ValueError:
            pass  # fallback to default below if conversion fails
    return 0.0


def log_model(
    callback: SolutionCallback, model_key: str, objective_value: float | None
):
    """
    Log the used model and the objective of a current solution in this model.
    """
    log_data = {
        "Time (in seconds)": callback.time_spent,
        "Deterministic model": parse_model_key(model_key),
    }

    if objective_value is not None:
        log_data.update(
            {
                "Objective new candidate": objective_value,
            }
        )

    wandb.log(log_data)


def log_solve_status(callback: SolutionCallback, result: Result):
    """
    Log the solve status of the last CP solve.
    """
    log_data = {
        "Time (in seconds)": callback.time_spent,
        "Solve status": parse_solve_status(result.status),
    }
    wandb.log(log_data)


def select_weighted_random_key(scores: dict[str, int]) -> str:
    """
    Randomly select a key with selection probability proportional to its score.

    Parameters
    ----------
    scores : dict[str, int]
        Dictionary mapping strings to non-negative integer scores. Scores
        are normalized to form a probability distribution.

    Returns
    -------
    str
        A randomly selected key, weighted by the corresponding score.
    """

    keys, values = zip(*scores.items(), strict=True)
    total = sum(values)
    if total == 0:
        raise ValueError("Ensure not all scores are zero.")
    probs = np.array(values) / total
    return np.random.choice(keys, p=probs)


def update_scores(
    callback: SolutionCallback,
    elite_tracker: dict[str, float],
    scores: dict[str, int],
    model_key: str,
    simh_config: DynamicSimheuristicConfig,
):
    """Helper function to update the scores in-place."""

    # Init
    solutions = callback.solutions
    if solutions.none_simulated():
        callback._log_event("No solutions simulated yet. Simulating now...")
        solutions.simulate_to_num_sims(simh_config["num_sims"])
        callback._log_event("Simulation finished.")
    best_obj = solutions.get_best_mean_solution().simulator.mean
    worst_obj = solutions.get_worst_mean_solution().simulator.mean

    # Find score change and update best and worst of elite tracker
    if best_obj < elite_tracker["best mean"]:
        score_change = simh_config["score_finding_new_best"]
        elite_tracker["best mean"] = best_obj
        elite_tracker["worst mean"] = worst_obj  # by definition <= old worst
    elif worst_obj < elite_tracker["worst mean"]:
        score_change = simh_config["score_finding_new_elite"]
        elite_tracker["worst mean"] = worst_obj
    else:
        score_change = -simh_config["init_score"]

    # Apply score change
    scores[model_key] = max(
        scores[model_key] + score_change, simh_config["init_score"]
    )
    wandb.log(
        {
            "Time (in seconds)": callback.time_spent,
            model_key + "_score_change": score_change,
        }
    )
