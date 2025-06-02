import time
from typing import List

import numpy as np
from tqdm.contrib.concurrent import process_map

from pyjobshop import Solution
from simpyjobshop.evaluator import evaluator
from simpyjobshop.problems.Problem import Problem


class Simulator:
    """
    Simulates a solution's performance for a problem.

    Attributes
    ----------
    problem : Problem
        Problem representation.
    solution : Solution
        Representing the solution to be evaluated.
    data_generator : DataGenerator
        Instance of DataGenerator to generate random data.
    results : List[float]
        List of results from simulation runs.
    """

    def __init__(self, problem: Problem, solution: Solution):
        self.problem = problem
        self.solution = solution
        self.data_generator = problem.build_data_generator()
        self.results: List[float] = []

    def parallel_simulate(
        self,
        num_sims: int,
        num_workers: int | None = None,
    ) -> None:
        """
        Performs `num_sims` simulation runs in parallel using process_map.

        Parameters
        ----------
        num_sims : int
            The number of simulations to run.
        num_workers : int, optional
            Number of parallel workers to use, by default None.

        Notes
        -----
        - Results are appended to self.results in order of input.
        """

        data_list = [self.data_generator.random() for _ in range(num_sims)]
        args_list = [
            (self.solution, self.problem, data, 1) for data in data_list
        ]

        results = process_map(
            Simulator._single_sim,
            args_list,
            max_workers=num_workers,
        )

        self.results.extend(results)

    @staticmethod
    def _single_sim(args: tuple) -> float:
        solution, problem, data, num_workers = args
        return evaluator(solution, problem, data, None)

    def simulate(
        self,
        num_sims: int,
        time_limit: float | None = None,
        num_workers: int | None = None,
    ) -> None:
        """
        Performs num_sims simulation runs and stores the results.

        Parameters
        ----------
        num_sims : int
            The number of simulations to run.
        time_limit : float, optional
            Time limit (in seconds) for simulate(), by default None.
        num_workers : int, optional
            Number of workers in evaluator, by default None.
        """

        start_time = None if time_limit is None else time.time()

        for _ in range(num_sims):
            if time_limit is not None:
                assert start_time is not None
                if time.time() - start_time >= time_limit:
                    break

            data = self.data_generator.random()
            result = evaluator(self.solution, self.problem, data, num_workers)
            self.results.append(result)

    @property
    def num_sims(self) -> int:
        """
        Returns the number of simulation runs.

        Returns
        -------
        int
            The number of simulation runs.
        """
        return len(self.results)

    @property
    def mean(self) -> float:
        """
        Computes the mean of the simulation results.

        Returns
        -------
        float
            The mean of simulation results, or None if no simulations were run.
        """
        return np.mean(self.results) if len(self.results) > 0 else None

    @property
    def variance(self) -> float:
        """
        Computes the variance of the simulation results.

        Returns
        -------
        float
            The variance of simulation results, or None if no sims. were run.
        """
        return np.var(self.results) if len(self.results) > 0 else None
