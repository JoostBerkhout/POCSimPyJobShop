from typing import Tuple, TypedDict

from pyjobshop import Result
from simpyjobshop.problems import Problem
from simpyjobshop.simheuristics.standard_simheuristic import (
    StandardSimheuristicConfig,
    standard_simheuristic,
)
from simpyjobshop.SolutionCallback import SolutionCallback


class SimulateLastSolutionsConfig(TypedDict):
    det_repr: str | float  # float in (0, 1) for a quantile or str "mean"
    max_size_elite_set: int
    frac_budget_final_elites_sim: float


def simulate_last_solutions(
    problem: Problem,
    simh_config: SimulateLastSolutionsConfig,
    exp_config: dict[str, int],
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
    simh_config : SimulateLastSolutionsConfig
        Configuration for the simheuristic (e.g., elite set size).
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

    frac_budget_final_elites_sim = simh_config["frac_budget_final_elites_sim"]
    _simh_config: StandardSimheuristicConfig = {
        "det_repr": simh_config["det_repr"],
        "num_sims": 0,  # no simulation during optimization
        "max_size_elite_set": simh_config["max_size_elite_set"],
        "frac_budget_before_sims": 1.1,  # no simulation during optimization
        "frac_budget_final_elites_sim": frac_budget_final_elites_sim,
    }

    return standard_simheuristic(
        problem,
        simh_config=_simh_config,
        exp_config=exp_config,
        wandb_config=wandb_config,
    )
