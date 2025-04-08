import time
from typing import List

import numpy as np

from pyjobshop import Solution
from pyjobshop.simheuristic.evaluator import evaluator
from pyjobshop.simheuristic.problems.Problem import Problem


class Simulator:
    """
    Simulates a solution's performance for a problem.

    Attributes
    ----------
    problem
        Problem representation.
    solution : Solution
        A callable representing the solution to be evaluated.
    data_generator : DataGenerator
        Instance of DataGenerator to generate input data.
    results : List[float]
        List of results from simulation runs.
    """

    def __init__(self, problem: Problem, solution: Solution):
        self.problem = problem
        self.solution = solution
        self.data_generator = problem.build_data_generator()
        self.results: List[float] = []

    def simulate(self, num_sims: int, time_limit: float | None = None):
        """
        Performs num_sims simulation runs and stores the results.

        Parameters
        ----------
        num_sims : int
            The number of simulations to run.
        time_limit : float, optional
            Time limit (in seconds) for simulate(), by default None.
        """

        start_time = None if time_limit is None else time.time()

        for _ in range(num_sims):
            if time_limit is not None:
                assert start_time is not None
                if time.time() - start_time >= time_limit:
                    break

            data = self.data_generator.random()
            result = evaluator(self.solution, self.problem, data)
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
