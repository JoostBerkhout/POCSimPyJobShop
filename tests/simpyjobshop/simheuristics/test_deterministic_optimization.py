from simpyjobshop.simheuristics import (
    DeterministicOptimizationConfig,
    deterministic_optimization,
)
from tests.simpyjobshop.problems.OneMachineTwoJobsSpecificRV import (
    OneMachineTwoJobsSpecificRV,
)


def test_det_opt_one_machine():
    """
    Tests whether solving a one-machine problem with two jobs works.
    """

    problem = OneMachineTwoJobsSpecificRV()

    exp_config = {
        "time_limit": 10**8,
        "num_workers": 1,
        "use_wandb": False,
        # only relevant for cli_run_experiments.py and submit_slurm_job.py:
        "num_rand_experiments": None,
        "num_parallel_instances": None,
        "num_sims_for_true_expec_objective": None,
    }

    # Solve with deterministic optimization using the mean
    det_opt_config: DeterministicOptimizationConfig = {
        "det_repr": "mean",
    }
    callback, last_CP_results, durations = deterministic_optimization(
        problem,
        det_opt_config,
        exp_config,
    )

    elites = callback.solutions
    assert len(elites) == 1
    assert elites[0].objective == 4
    assert elites[0].schedule[0] == [0, 1]

    # Solve with deterministic optimization using the 0.75 quantile
    det_opt_config: DeterministicOptimizationConfig = {
        "det_repr": 0.75,
    }
    callback, last_CP_results, durations = deterministic_optimization(
        problem,
        det_opt_config,
        exp_config,
    )

    elites = callback.solutions
    assert len(elites) == 1
    assert elites[0].objective == 7
    assert elites[0].schedule[0] == [1, 0]

    # Solve with deterministic optimization using the 0.66 quantile
    det_opt_config: DeterministicOptimizationConfig = {
        "det_repr": 0.66,
    }
    callback, last_CP_results, durations = deterministic_optimization(
        problem,
        det_opt_config,
        exp_config,
    )

    elites = callback.solutions
    assert len(elites) == 1
    assert elites[0].objective == 2
    assert elites[0].schedule[0] == [0, 1]
