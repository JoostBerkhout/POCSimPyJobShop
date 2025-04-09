import time
from typing import Tuple

import wandb

from pyjobshop import Result
from pyjobshop.simheuristic.EliteSolutions import EliteSolutions
from pyjobshop.simheuristic.problems.Problem import Problem
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback
from pyjobshop.simheuristic.utils import init_wandb


def standard_simheuristic(
    problem: Problem,
    simh_config: dict[str, int],
    exp_config: dict[str, int],
    use_wandb: bool = False,
    wandb_config: dict[str, str] | None = None,
) -> Tuple[EliteSolutions, Result, float]:
    """
    Runs a standard simheuristic on the given problem.

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
    elite_solutions : EliteSolutions
        The elite solutions found during the simheuristic.
    result : Result
        A Result object containing the best found solution and additional
        information about the solver run.
    exp_duration : float
        The duration of the experiment in seconds. It does not include the
        loading duration of wandb if used.
    """

    if use_wandb:
        problem_name = problem.__class__.__name__
        save_config = {"exp_config": exp_config, "simh_config": simh_config}
        init_wandb(problem_name, save_config, wandb_config)

    start_time_exp = time.time()

    # Init
    time_limit = exp_config["time_limit"]
    num_sims = simh_config["num_sims"]
    start_time_sims = simh_config["%_budget_before_sims"] * time_limit
    max_size_elite_set = simh_config["max_size_elite_set"]
    remaining_time_limit = time_limit - (time.time() - start_time_exp)
    assert remaining_time_limit > 0, "Time limit reached before experiment..."

    # Generate problem data and build model
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)

    # Solve the problem using a callback for stochastic evaluations
    current_time = time.time()
    callback = SolutionCallback(
        problem=problem,
        num_sims=num_sims,
        start_time_exp=start_time_exp,
        start_time_sims=current_time + start_time_sims,
        max_size_elite_set=max_size_elite_set,
        stop_time=current_time + remaining_time_limit,
    )
    result = model.solve(
        callback=callback,
        display=False,
        time_limit=remaining_time_limit,
        num_workers=exp_config["num_workers"],
    )

    exp_duration = time.time() - start_time_exp

    if use_wandb:
        wandb.finish()

    return callback.solutions, result, exp_duration
