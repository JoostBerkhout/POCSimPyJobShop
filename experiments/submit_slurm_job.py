"""
One-off script for submitting a bunch of jobs to a SLURM cluster.

Specifically, one job is submitted for each problem by executing the
``cli_run_experiments.py`` script.
"""

import argparse
from math import ceil
from subprocess import run

from experiments.utils.check_cpu_usage import check_cpu_use
from experiments.utils.configs_loader import get_configs
from simpyjobshop import problems

JOBSCRIPT = """#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH --time={job_time_limit}
#SBATCH --nodes=1
#SBATCH --partition=genoa
#SBATCH --array=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=joost.berkhout@vu.nl
#SBATCH --out=slurm/{job_name}-%A_%a.out

uv run cli_run_experiments.py \
--problem {problem} \
"""


MIN_NUM_CPUS = 24
MAX_NUM_CPUS = 192


def main(problem: str, mock: bool):
    """
    Submits a job to the SLURM cluster for the specified problem.

    This function calculates the required job runtime and other configurations
    and formats a job script to submit to SLURM.

    Parameters
    ----------
    problem : str
        The name of the problem class to run in the experiment.
    """

    job_name = problem
    exp_config, simheuristics = get_configs(problem)
    num_simh = len(simheuristics)
    time_per_sim = 0.05 * 3  # times 3 to be safe
    sim_time = exp_config["num_sims_for_true_expec_objective"] * time_per_sim
    run_time = exp_config["time_limit"] + sim_time
    num_exps = exp_config["num_rand_experiments"] * num_simh
    num_workers = exp_config["num_workers"]
    num_parallel_instances = exp_config["num_parallel_instances"]
    num_consec_proc = ceil(num_exps / num_parallel_instances)
    total_time = num_consec_proc * run_time
    buffer = 5 * 60  # 5 minutes buffer
    job_time_limit = seconds_to_string(int(total_time + buffer))
    check_cpu_use(num_parallel_instances, num_workers, MAX_NUM_CPUS)
    num_cpus = min(MAX_NUM_CPUS, num_parallel_instances * num_workers)
    num_cpus = max(num_cpus, MIN_NUM_CPUS)

    jobscript = JOBSCRIPT.format(
        job_name=job_name,
        job_time_limit=job_time_limit,
        problem=problem,
        num_cpus=num_cpus,
    )

    if mock:
        print(jobscript)
    else:
        run(["sbatch"], input=jobscript.encode())


def seconds_to_string(seconds: int) -> str:
    """
    Converts number of seconds into a time string in the format HH:MM:SS.

    Parameters
    ----------
    seconds : int
        The number of seconds to convert.

    Returns
    -------
    str
        A string in the format 'HH:MM:SS'.
    """
    mins, seconds = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{seconds:02d}"


def get_available_problems():
    """
    Retrieve a list of available problem class names from the problems module.

    Returns
    -------
    list of str
        A list of problem class names (strings).
    """
    return [
        name for name, cls in vars(problems).items() if isinstance(cls, type)
    ]


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    Namespace
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--problem",
        required=True,
        help="Problem class name",
        choices=get_available_problems(),
    )
    parser.add_argument(
        "--mock",
        required=False,
        help="For testing purposes",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.problem, args.mock)
