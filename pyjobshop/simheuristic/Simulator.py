from typing import Callable, Dict, List

import numpy as np

from pyjobshop import Solution
from pyjobshop.simheuristic.DataGenerator import DataGenerator


class Simulator:
    """
    Simulates a solution's performance using data from a generator.

    Attributes
    ----------
    data_generator : DataGenerator
        Instance of DataGenerator to generate input data.
    solution : Solution
        A callable representing the solution to be evaluated.
    evaluator : Callable[[Solution, Dict], float]
        A function to evaluate the solution on generated data.
    results : List[float]
        List of results from simulation runs.
    """

    def __init__(
        self,
        data_generator: DataGenerator,
        solution: Solution,
        evaluator: Callable[[Solution, Dict], float],
    ) -> None:
        """
        Initializes the Simulator.

        Parameters
        ----------
        data_generator : DataGenerator
            The data generator to produce input data.
        solution : Solution
            The solution to be evaluated.
        evaluator : Callable[[Solution, Dict], float]
            The evaluation function that computes the objective value.
        """
        self.data_generator = data_generator
        self.solution = solution
        self.evaluator = evaluator
        self.results: List[float] = []

    def simulate(self, num_sims: int) -> None:
        """
        Performs multiple simulation runs and stores the results.

        Parameters
        ----------
        num_sims : int
            The number of simulations to run.
        """
        for _ in range(num_sims):
            data = self.data_generator.random()
            result = self.evaluator(self.solution, data)
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
