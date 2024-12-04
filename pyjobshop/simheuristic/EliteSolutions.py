from typing import Dict, Tuple

from pyjobshop import Solution
from pyjobshop.simheuristic.Simulator import Simulator


class EliteSolutions:
    """
    Manages solutions and their associated simulators.

    Attributes
    ----------
    solutions : Dict[int, Tuple[Solution, Simulator, float]]
        A mapping from solution ids to their solution, simulator & objective.
    """

    def __init__(self) -> None:
        self.solutions: Dict[int, Tuple[Solution, Simulator, float]] = {}

    def add(self, solution: Solution, simulator: Simulator, objective: float):
        """
        Adds a solution and its simulator to the storage.

        Parameters
        ----------
        solution : Solution
            The solution to add.
        simulator : Simulator
            The simulator associated with the solution.
        objective : float
            The objective value of the solution when found.
        """
        self.solutions[id(solution)] = (solution, simulator, objective)

    def get_best_solution(self) -> Solution:
        """
        Returns the best-performing solution based on mean objective value.

        Returns
        -------
        Solution
            The best-performing solution.
        """
        return min(self.solutions.values(), key=lambda val: val[1].mean)[0]

    def print_summary(self) -> None:
        """Prints a summary of the elite solutions."""

        print("\nElite Solutions:")
        for solution, simulator, objective in self.solutions.values():
            print(
                f"Solution id: {id(solution)} | "
                f"Objective: {objective:.2f} | "
                f"Mean: {simulator.mean:.2f} | "
                f"Var: {simulator.variance:.2f}"
            )
