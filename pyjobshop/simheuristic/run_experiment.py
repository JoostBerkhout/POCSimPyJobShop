from typing import Any

import pandas as pd

from pyjobshop.simheuristic.problems.ParallelMachineProblem import (
    ParallelMachineProblem,
)
from pyjobshop.simheuristic.simheuristics import (
    DynamicSimheuristicConfig,
    StandardSimheuristicConfig,
    dynamic_simheuristic,
)

# init
project_name = "simpyjobshop-PMP"
GenericProblem = ParallelMachineProblem
problem_name = GenericProblem.__name__
problem_name_short = "".join([c for c in problem_name if c.isupper()])
exp_config = {
    "time_limit": 60,
    "num_rand_experiments": 1,
    "num_workers": 1,
    "num_sims_for_truth_expec_objective": 100,
}
stand_simh_config = {
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
sim_last_config: StandardSimheuristicConfig = {
    "num_sims": 0,
    "max_size_elite_set": 10,
    "frac_budget_before_sims": 1.1,
    "frac_budget_final_elites_sim": 0.4,  # should be enough to sim. final
}
simheuristics: dict[str, dict[str, Any]] = {
    # "stand_simh": {
    #     "simheuristic": standard_simheuristic,
    #     "simh_config": stand_simh_config,
    #     },
    # "deterministic_opt": {
    #     "simheuristic": deterministic_optimization,
    #     "simh_config": dict(),
    #     },
    # "sim_last": {
    #     "simheuristic": standard_simheuristic,
    #     "simh_config": sim_last_config,
    #     },
    "dynamic_simh": {
        "simheuristic": dynamic_simheuristic,
        "simh_config": dyn_simh_config,
    },
}
results = []

for simh_name, simh in simheuristics.items():
    for seed in range(exp_config["num_rand_experiments"]):
        problem = GenericProblem(seed=seed)

        wandb_config = {
            "project_name": project_name,
            "job_name": simh_name,
            "run_name": f"{problem_name_short}_{simh_name}_seed_{seed}",
        }

        callback, last_CP_results, durations = simh["simheuristic"](
            problem,
            simh["simh_config"],
            exp_config,
            use_wandb=True,
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
                "Some elites are simulated and some not. "
                "This is not expected."
            )

        # Simulate elite
        cur_num_sims = elite.simulator.num_sims
        extra_num_sims = max(
            exp_config["num_sims_for_truth_expec_objective"] - cur_num_sims, 0
        )
        elite.simulator.simulate(num_sims=extra_num_sims)

        # Save results
        results.append(
            {
                "project_name": project_name,
                "problem_type": problem_name,
                "simheuristic": simh_name,
                "seed": seed,
                "mean": elite.simulator.mean,
                "std": elite.simulator.variance**0.5,
                "num_sims": elite.simulator.num_sims,
                "DCOP_objective": elite.objective,
                "callback_log": callback.get_log_str(),
            }
            | durations
        )

        callback.print_log()

# load as pandas df
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
