import time
from typing import Tuple, TypedDict

import wandb

from pyjobshop import Result
from simpyjobshop.problems.Problem import Problem
from simpyjobshop.SolutionCallback import SolutionCallback
from simpyjobshop.utils import init_wandb


class StandardSimheuristicConfig(TypedDict):
    det_repr: float | str  # float in (0, 1) for a quantile or str "mean"
    num_sims: int
    max_size_elite_set: int
    frac_budget_before_sims: float
    frac_budget_final_elites_sim: float


def standard_simheuristic(
    problem: Problem,
    simh_config: StandardSimheuristicConfig,
    exp_config: dict[str, int],
    wandb_config: dict[str, str] | None = None,
) -> Tuple[SolutionCallback, Result, dict[str, float]]:
    """
    Runs a standard simheuristic on the given problem.

    Parameters
    ----------
    problem : Problem
        The optimization problem instance (e.g., HybridFlowShop).
    simh_config : StandardSimheuristicConfig
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

    # Init
    time_limit = exp_config["time_limit"]
    det_repr = simh_config["det_repr"]
    num_sims = simh_config["num_sims"]
    start_time_sims = simh_config["frac_budget_before_sims"] * time_limit
    max_size_elite_set = simh_config["max_size_elite_set"]
    time_final_sims = simh_config["frac_budget_final_elites_sim"] * time_limit

    # Generate problem data and build model
    data_generator = problem.build_data_generator()
    data = data_generator.by_key(det_repr)
    model = problem.concrete_model(data)

    # Solve the problem using a callback for stochastic evaluations
    callback = SolutionCallback(
        problem=problem,
        num_sims=num_sims,
        start_time_exp=start_time_exp,
        start_time_sims=start_time_exp + start_time_sims,
        max_size_elite_set=max_size_elite_set,
        stop_time=start_time_exp + time_limit - time_final_sims,
    )
    result = model.solve(
        callback=callback,
        display=False,
        time_limit=time_limit - time_final_sims,
        num_workers=exp_config["num_workers"],
    )

    # Log times
    time_spent = time.time() - start_time_exp
    durations["solver_phase_dur"] = time_spent

    # Final simulation of the elite solutions for the remaining time
    rem_time_final_sims = min(time_final_sims, time_limit - time_spent)
    callback.solutions.simulate_to_time_limit(rem_time_final_sims)

    # Log times
    exp_duration = time.time() - start_time_exp
    durations["final_sim_phase_dur"] = exp_duration - time_spent
    durations["total_exp_dur"] = exp_duration

    if exp_config["use_wandb"]:
        wandb.finish()

    return callback, result, durations
