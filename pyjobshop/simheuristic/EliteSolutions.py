from typing import Dict

from pyjobshop import Solution
from pyjobshop.simheuristic.Simulator import Simulator


class EliteSolutions:
    """
    Manages solutions and their associated simulators.

    Attributes
    ----------
    solutions : Dict[Solution, Simulator]
        A mapping from solutions to their simulators.
    """

    def __init__(self) -> None:
        """Initializes an empty dictionary for solutions and simulators."""
        self.solutions: Dict[Solution, Simulator] = {}

    def add(self, solution: Solution, simulator: Simulator) -> None:
        """
        Adds a solution and its simulator to the storage.

        Parameters
        ----------
        solution : Solution
            The solution to add.
        simulator : Simulator
            The simulator associated with the solution.
        """
        self.solutions[solution] = simulator

    def get_best_solution(self) -> Solution:
        """
        Returns the best-performing solution based on mean objective value.

        Returns
        -------
        Solution
            The best-performing solution.
        """
        return min(self.solutions, key=lambda sol: self.solutions[sol].mean)
