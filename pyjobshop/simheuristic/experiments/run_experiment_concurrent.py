import time
from typing import Any, Type

import pandas as pd
from tqdm.contrib.concurrent import process_map

from pyjobshop.simheuristic.experiments.SimheuristicSpec import (
    SimheuristicSpec,
)
from pyjobshop.simheuristic.problems.ParallelMachineProblem import (
    ParallelMachineProblem,
)
from pyjobshop.simheuristic.problems.Problem import Problem
from pyjobshop.simheuristic.simheuristics import (
    DynamicSimheuristicConfig,
    SimulateLastSolutionsConfig,
    StandardSimheuristicConfig,
    deterministic_optimization,
    dynamic_simheuristic,
    simulate_last_solutions,
    standard_simheuristic,
)


def run_experiment(
    project_name: str,
    GenericProblem: Type[Problem],
    problem_seed: int,
    simheuristic: SimheuristicSpec,
    exp_config: dict[str, int],
    use_wandb: bool = False,
) -> dict[str, Any]:
    simh_name = simheuristic.name
    simh_fun = simheuristic.fun
    simh_config = simheuristic.config

    problem = GenericProblem(seed=problem_seed)
    problem_name = GenericProblem.__name__
    problem_name_short = "".join([c for c in problem_name if c.isupper()])

    wandb_config = {
        "project_name": project_name,
        "job_name": simh_name,
        "run_name": f"{problem_name_short}_{simh_name}_seed_{problem_seed}",
    }

    callback, last_CP_results, durations = simh_fun(
        problem,
        simh_config,
        exp_config,
        use_wandb=use_wandb,
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
            "Some elites are simulated and some not. " "This is not expected."
        )

    # Simulate elite
    cur_num_sims = elite.simulator.num_sims
    extra_num_sims = max(
        exp_config["num_sims_for_truth_expec_objective"] - cur_num_sims, 0
    )
    elite.simulator.simulate(num_sims=extra_num_sims)

    # Save results
    exp_run_results = {
        "project_name": project_name,
        "problem_type": problem_name,
        "simheuristic": simh_name,
        "seed": problem_seed,
        "mean": elite.simulator.mean,
        "std": elite.simulator.variance**0.5,
        "num_sims": elite.simulator.num_sims,
        "DCOP_objective": elite.objective,
        "callback_log": callback.get_log_str(),
    }
    exp_run_results.update(durations)

    return exp_run_results


def run_experiment_unpack(args):
    return run_experiment(*args)


if __name__ == "__main__":
    start_time = time.time()

    # init
    project_name = "simpyjobshop-PMP"
    GenericProblem = ParallelMachineProblem
    problem_name = GenericProblem.__name__
    problem_name_short = "".join([c for c in problem_name if c.isupper()])
    use_wandb = True
    exp_config = {
        "time_limit": 30,
        "num_rand_experiments": 1,
        "num_workers": 1,
        "num_sims_for_truth_expec_objective": 100,
    }
    stand_simh_config: StandardSimheuristicConfig = {
        "num_sims": 25,
        "max_size_elite_set": 5,
        "frac_budget_before_sims": 0.5,
        "frac_budget_final_elites_sim": 0.1,
    }
    dyn_simh_config: DynamicSimheuristicConfig = {
        "num_sims": 25,
        "max_size_elite_set": 5,
        "score_finding_new_elite": 1,
        "score_finding_new_best": 2,
        "init_score": 1,
        "max_time_per_cp_solve": 10,
        "consider_mean": True,
        "quantiles": [0.5, 0.6, 0.7],
        "frac_budget_before_sims": 2 / exp_config["time_limit"],
        "frac_budget_final_elites_sim": 5 / exp_config["time_limit"],
        # frac_budget_final_elites_sim should be enough to sim. final
    }
    sim_last_config: SimulateLastSolutionsConfig = {
        "max_size_elite_set": 10,
        "frac_budget_final_elites_sim": 0.4,  # should be enough to sim. final
    }
    simheuristics: list[SimheuristicSpec] = [
        SimheuristicSpec("std_simh", standard_simheuristic, stand_simh_config),
        SimheuristicSpec("det_opt", deterministic_optimization, {}),
        SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config),
        SimheuristicSpec("dyn_simh", dynamic_simheuristic, dyn_simh_config),
    ]

    # results = []
    # for seed in range(exp_config["num_rand_experiments"]):
    #     for simh in simheuristics:
    #         exp_result = run_experiment(
    #             project_name=project_name,
    #             GenericProblem=GenericProblem,
    #             problem_seed=seed,
    #             simheuristic=simh,
    #             exp_config=exp_config,
    #             use_wandb=use_wandb,
    #         )
    #         results.append(exp_result)

    # Build all combinations of (seed, simheuristic)
    combinations = [
        (project_name, GenericProblem, seed, simh, exp_config, use_wandb)
        for seed in range(exp_config["num_rand_experiments"])
        for simh in simheuristics
    ]

    # Use process_map directly without extra functions
    results = process_map(
        run_experiment_unpack,
        combinations,
        max_workers=4,  # Adjust to your machine
        unit="experiment",
    )

    # load as pandas df
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)

    duration = time.time() - start_time
    print(f"Finished running the experiments concurrently in {duration} sec.")
