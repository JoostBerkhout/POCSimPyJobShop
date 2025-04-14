from typing import Tuple

from pyjobshop import Result
from simpyjobshop.problems.Problem import Problem
from simpyjobshop.simheuristics.standard_simheuristic import (
    StandardSimheuristicConfig,
    standard_simheuristic,
)
from simpyjobshop.SolutionCallback import SolutionCallback


def deterministic_optimization(
    problem: Problem,
    simh_config: dict,
    exp_config: dict[str, int],
    use_wandb: bool = False,
    wandb_config: dict[str, str] | None = None,
) -> Tuple[SolutionCallback, Result, dict[str, float]]:
    """
    Runs a deterministic optimization on the given problem.

    It uses the standard simheuristic approach with no simulation for
    convenience.

    Parameters
    ----------
    problem : Problem
        The optimization problem instance (e.g., HybridFlowShop).
    simh_config : dict
        To ensure a similar signature as a simheuristic. It is not used.
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

    assert len(simh_config) == 0, "simh_config should be empty."

    _simh_config: StandardSimheuristicConfig = {
        "num_sims": 0,
        "max_size_elite_set": 1,
        "frac_budget_before_sims": 1.1,
        "frac_budget_final_elites_sim": 0.0,
    }

    return standard_simheuristic(
        problem,
        simh_config=_simh_config,
        exp_config=exp_config,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
    )
