from typing import Any, Type

from experiments.utils.SimheuristicSpec import SimheuristicSpec
from simpyjobshop.modeling import find_result_for_expected_data
from simpyjobshop.problems import Problem


def run_experiment(
    project_name: str,
    GenericProblem: Type[Problem],
    problem_seed: int,
    simheuristic: SimheuristicSpec,
    exp_config: dict[str, int],
) -> dict[str, Any]:
    """
    Runs a single simheuristic experiment on a specified problem.

    Parameters
    ----------
    project_name : str
        Name of the project (used for logging and WandB).
    GenericProblem : Type[Problem]
        Problem class to instantiate with the provided seed.
    problem_seed : int
        Random seed for reproducibility.
    simheuristic : SimheuristicSpec
        Simheuristic to use, with function and config bundled.
    exp_config : dict of str to int
        Configuration for the experiment (e.g., time limit).

    Returns
    -------
    dict of str to Any
        Dictionary with experiment metadata, stats, and logs.
    """

    simh_name = simheuristic.name
    simh_fun = simheuristic.fun
    simh_config = simheuristic.config

    problem = GenericProblem(seed=problem_seed)
    problem_name = GenericProblem.__name__
    short_name = "".join([c for c in problem_name if c.isupper()])

    wandb_config = (
        {
            "project_name": project_name,
            "job_name": simh_name,
            "run_name": f"{short_name}_{simh_name}_seed_{problem_seed}",
        }
        if exp_config["use_wandb"]
        else None
    )

    callback, last_CP_results, durations = simh_fun(
        problem,
        simh_config,
        exp_config,
        wandb_config=wandb_config,
    )

    # Find elite
    elite_solutions = callback.solutions
    if elite_solutions.all_simulated():
        elite = elite_solutions.get_best_mean_solution()
    elif elite_solutions.none_simulated():
        elite = elite_solutions.get_best_deterministic_solution()
    else:
        raise Exception(
            "Some elites are simulated and some not. This is not expected."
        )

    # Simulate elite
    cur_num_sims = elite.simulator.num_sims
    extra_num_sims = max(
        exp_config["num_sims_for_true_expec_objective"] - cur_num_sims, 0
    )
    elite.simulator.simulate(num_sims=extra_num_sims)

    # Find results for elite with expected data
    exp_data_result = find_result_for_expected_data(elite.solution, problem)

    # Save results
    exp_run_results = {
        "project_name": project_name,
        "problem_type": problem_name,
        "simheuristic": simh_name,
        "seed": problem_seed,
        "mean": elite.simulator.mean,
        "std": elite.simulator.variance**0.5,
        "num_sims": elite.simulator.num_sims,
        "DCOP_objective": exp_data_result.objective,
        "DCOP_solution": exp_data_result.best.to_json_str(),
        "callback_log": callback.get_log_str(),
        "solve status": last_CP_results.status,
    }
    exp_run_results.update(durations)

    return exp_run_results


def run_experiment_unpack(args):
    """Helper function to unpack arguments for parallel processing."""
    return run_experiment(*args)
