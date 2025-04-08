import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pyjobshop import Solution
from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.utils import find_schedule_per_resource


@dataclass
class EliteSolution:
    """
    Represents a solution with associated simulator, objective, and metadata.

    Attributes
    ----------
    solution : Solution
        The solution object.
    simulator : Simulator
        The simulator associated with the solution.
    objective : float
        The objective value of the solution.
    schedule : dict[int, list[int]]
        A dictionary mapping resource indices to scheduled task indices, resp.
    metadata : Dict[str, Any]
        Additional attributes of the solution (e.g., lower bounds).
    """

    solution: Solution
    simulator: Simulator
    objective: float
    schedule: dict[int, list[int]]
    metadata: Dict[str, Any]


class EliteSolutions:
    """
    Manages elite solutions.

    Attributes
    ----------
    elite_solutions : Dict[int, EliteSolution]
        A mapping from solution IDs to EliteSolution objects.
    """

    def __init__(self) -> None:
        self.elite_solutions: Dict[int, EliteSolution] = {}

    def add(
        self,
        solution: Solution,
        simulator: Simulator,
        objective: float,
        schedule: Optional[dict[int, list[int]]] = None,
        **metadata: Any,
    ):
        """
        Adds a new solution.

        Parameters
        ----------
        solution : Solution
            The solution object to add.
        simulator : Simulator
            The simulator associated with the solution.
        objective : float
            The objective value of the solution.
        schedule : Optional[dict[int, list[int]]]
            Maps resource indices to scheduled task indices.
        **metadata : Any
            Additional metadata for the solution.
        """
        if schedule is None:
            schedule = find_schedule_per_resource(solution)

        self.elite_solutions[id(solution)] = EliteSolution(
            solution=solution,
            simulator=simulator,
            objective=objective,
            schedule=schedule,
            metadata=metadata,
        )

    def is_new_schedule(self, schedule: Dict[int, list[int]]) -> bool:
        """
        Checks if schedule is new.

        Parameters
        ----------
        schedule : Dict[int, list[int]]
            Maps resource indices to scheduled task indices.

        Returns
        -------
        bool
            True if the schedule is new, otherwise False.
        """
        for elite_solution in self.elite_solutions.values():
            if schedule == elite_solution.schedule:
                return False
        return True

    def keep_top_n(self, n: int):
        """
        Keeps only the top n solutions with the lowest mean objective if all
        are simulated, else it will keep the best deterministic ones.

        Parameters
        ----------
        n : int
            The maximum number of solutions to retain.
        """
        if len(self.elite_solutions) <= n:
            return

        if self.all_simulated():
            # Sort solutions by mean objective value
            sorted_solutions = sorted(
                self.elite_solutions.values(),
                key=lambda elite_solution: elite_solution.simulator.mean,
            )
        else:
            # Sort solutions by objective value
            sorted_solutions = sorted(
                self.elite_solutions.values(),
                key=lambda elite_solution: elite_solution.objective,
            )

        # Retain only the top n solutions
        self.elite_solutions = {
            id(elite_solution.solution): elite_solution
            for elite_solution in sorted_solutions[:n]
        }

    def get_best_elite_solution(self) -> EliteSolution:
        """
        Returns the best-performing solution based on mean objective value.

        Only works if all elite solutions have been simulated.

        Returns
        -------
        EliteSolution
            The best-performing elite solution.
        """
        best_elite_solution = min(
            self.elite_solutions.values(),
            key=lambda elite_solution: elite_solution.simulator.mean,
        )
        return best_elite_solution

    def get_worst_elite_solution(self) -> EliteSolution:
        """
        Returns the worst-performing solution based on mean objective value.

        Only works if all elite solutions have been simulated.

        Returns
        -------
        EliteSolution
            The worst-performing elite solution.
        """
        worst_elite_solution = max(
            self.elite_solutions.values(),
            key=lambda elite_solution: elite_solution.simulator.mean,
        )
        return worst_elite_solution

    def print_summary(self):
        """Prints a summary of the elite solutions."""
        print("\nElite Solutions:")
        for elite_solution in self.elite_solutions.values():
            print(
                f"Solution id: {id(elite_solution.solution)} | "
                f"Objective: {elite_solution.objective:.2f} | "
                f"Num sims: {elite_solution.simulator.num_sims:.2f} | "
                f"Mean: {elite_solution.simulator.mean:.2f} | "
                f"Var: {elite_solution.simulator.variance:.2f} | "
                f"Metadata: {elite_solution.metadata}"
            )

    def all_simulated(self) -> bool:
        """Returns True if all elite solutions are simulated, False else."""
        for elite_solution in self.elite_solutions.values():
            if elite_solution.simulator.num_sims == 0:
                return False
        return True

    def simulate_to_num_sims(
        self, num_sims: int, time_limit: float | None = None
    ):
        """
        Ensures all elite solutions are simulated up to the specified number of
        simulations.

        Simulation stops early if the optional time limit is exceeded.

        Parameters
        ----------
        num_sims : int
            The total number of simulations each elite solution should have.

        time_limit : float, optional
            The maximum amount of time allowed for simulation (in seconds).
            If None, there is no time constraint.

        Notes
        -----
        - Only solutions with fewer than `num_sims` current simulations will be
          updated.
        - The remaining time is passed to each simulator; it's assumed that the
          simulator can respect this time constraint internally.
        """
        start_time = time.time()
        remaining_time = None

        for elite_solution in self.elite_solutions.values():
            current_num_sims = elite_solution.simulator.num_sims

            if current_num_sims < num_sims:
                extra_sims = num_sims - current_num_sims

                if time_limit is not None:
                    time_spent = time.time() - start_time
                    if time_spent >= time_limit:
                        return
                    remaining_time = max(time_limit - time_spent, 0.0)

                elite_solution.simulator.simulate(extra_sims, remaining_time)
