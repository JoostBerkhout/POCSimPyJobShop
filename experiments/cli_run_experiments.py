import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Type

import pandas as pd
from tqdm.contrib.concurrent import process_map

from experiments.run_experiment import run_experiment_unpack
from experiments.utils.check_cpu_usage import check_cpu_use
from experiments.utils.configs_loader import get_configs
from simpyjobshop import problems
from simpyjobshop.problems import Problem


def main(problem: str, problem_cls: Type[Problem]):
    """
    Run a batch of simheuristic experiments for a given problem class.

    Parameters
    ----------
    problem : str
        Name of the problem class (e.g., ParallelMachineProblem).
    problem_cls : Type[problems.Problem]
        Problem class object from the problems module.
    """
    start_time = time.time()

    # Load problem-specific or fallback experiment configuration
    exp_config, simheuristics = get_configs(problem)
    num_parallel_instances = exp_config["num_parallel_instances"]
    check_cpu_use(num_parallel_instances, exp_config["num_workers"])

    # Set project name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = problem + "_" + timestamp

    # Prepare all combinations of seeds and simheuristics
    combinations = [
        (project_name, problem_cls, seed, simh, exp_config)
        for seed in range(exp_config["num_rand_experiments"])
        for simh in simheuristics
    ]

    # Run experiments in parallel
    results = process_map(
        run_experiment_unpack,
        combinations,
        max_workers=num_parallel_instances,
        unit="experiment_run",
    )

    # Load as pandas df
    df = pd.DataFrame(results)
    results_dir = Path(__file__).resolve().parent / "results"
    df.to_csv(results_dir / f"{project_name}.csv", index=False)

    duration = time.time() - start_time
    print(f"Finished experiments in {duration:.2f} seconds.")


def maybe_mkdir(where: str):
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    where : str
        Path to the directory to create.
    """
    if where:
        dir_loc = Path(where)
        dir_loc.mkdir(parents=True, exist_ok=True)


def parse_args():
    """
    Parse CLI arguments and fetch the corresponding problem class.

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    problem_cls : Type[problems.Problem]
        Problem class object based on user input.

    Raises
    ------
    ValueError
        If the problem name does not correspond to a known class.
    """
    parser = argparse.ArgumentParser(
        description="Run simheuristic experiments."
    )
    parser.add_argument("--problem", required=True, help="Problem class name")
    args = parser.parse_args()

    # Get the problem class from the problems module
    problem_cls = getattr(problems, args.problem, None)
    if problem_cls is None:
        raise ValueError(f"Unknown problem: {args.problem}")

    return args, problem_cls


if __name__ == "__main__":
    args, problem_cls = parse_args()
    main(args.problem, problem_cls)
