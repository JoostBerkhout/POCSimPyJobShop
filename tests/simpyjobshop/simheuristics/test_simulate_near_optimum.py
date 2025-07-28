import heapq

from simpyjobshop.simheuristics import (
    SimulateNearOptimumConfig,
    simulate_near_optimum,
)
from tests.simpyjobshop.problems.OneMachineTwoJobs import OneMachineTwoJobs
from tests.simpyjobshop.problems.TwoMachineTwoJobs import TwoMachinesTwoJobs


def test_simulate_near_optimum():
    """
    Tests that simulate_near_optimum finds all solutions.
    """

    experiments = [
        {
            "Problem": OneMachineTwoJobs(),
            "num_solutions": 2,
            "objective_values": [2.0, 12.0],
        },
        {
            "Problem": TwoMachinesTwoJobs(),
            "num_solutions": 6,
            "objective_values": [1.0, 1.0, 2.0, 2.0, 12.0, 12.0],
        },
    ]

    # Init experiment and simheuristic config
    exp_config = {
        "time_limit": 10**10,
        "num_workers": 8,  # (per instance)
        "use_wandb": False,
        # only relevant for cli_run_experiments.py and submit_slurm_job.py:
        "num_rand_experiments": None,
        "num_parallel_instances": None,
        "num_sims_for_true_expec_objective": None,
    }
    near_opt_sim_config: SimulateNearOptimumConfig = {
        "det_repr": "mean",
        "num_sims": 0,
        "max_size_elite_set": 10**10,
        "frac_budget_final_elites_sim": 0,
    }

    for max_size_elite_set in [10**10, 3, 2, 1]:
        near_opt_sim_config["max_size_elite_set"] = max_size_elite_set
        for exp_data in experiments:
            # Init problem
            problem = exp_data["Problem"]

            # Find all solutions
            callback, last_CP_results, durations = simulate_near_optimum(
                problem,
                near_opt_sim_config,
                exp_config,
            )

            # Check results
            exp_num_sols = min(exp_data["num_solutions"], max_size_elite_set)
            assert len(callback.solutions) == exp_num_sols
            all_objectives = [es.objective for es in callback.solutions]
            exp_obj_vals = heapq.nsmallest(
                max_size_elite_set, exp_data["objective_values"]
            )
            assert sorted(all_objectives) == sorted(exp_obj_vals)
