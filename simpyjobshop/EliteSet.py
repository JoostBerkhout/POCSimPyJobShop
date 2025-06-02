import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from tqdm.contrib.concurrent import process_map

from pyjobshop import Solution
from simpyjobshop.Simulator import Simulator
from simpyjobshop.utils import find_schedule_per_resource


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


class EliteSet:
    """
    Manages a set of elite solutions.

    Notes
    -----
    - Solutions are not required to be unique.
    - Insertion order may change (e.g., after keep_top_n).

    Attributes
    ----------
    _solutions : List[EliteSolution]
        A list of elite solutions.
    """

    def __init__(self) -> None:
        self._solutions: List[EliteSolution] = []

    def __iter__(self) -> Iterator[EliteSolution]:
        return iter(self._solutions)

    def __len__(self) -> int:
        return len(self._solutions)

    def __getitem__(self, idx: int) -> EliteSolution:
        return self._solutions[idx]

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

        self._solutions.append(
            EliteSolution(
                solution=solution,
                simulator=simulator,
                objective=objective,
                schedule=schedule,
                metadata=metadata,
            )
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
        return all(schedule != sol.schedule for sol in self._solutions)

    def _get_sort_key(self) -> Callable[[EliteSolution], float]:
        """
        Returns a key function to rank solutions based on objective type.

        Comparing is only meaningful if all solutions are simulated or when
        none of them are simulated.

        Returns
        -------
        Callable
            A function used to compare solutions.
        """
        if self.all_simulated():

            def _sort_key(sol: EliteSolution) -> float:
                return sol.simulator.mean
        elif self.none_simulated():

            def _sort_key(sol: EliteSolution) -> float:
                return sol.objective
        else:
            raise Exception("All or none of the solutions must be simulated.")
        return _sort_key

    def keep_top_n(self, n: int):
        """
        Keeps only the top-n elite solutions with the lowest mean or
        deterministic objective.

        Parameters
        ----------
        n : int
            The maximum number of solutions to retain.
        """
        if len(self._solutions) <= n:
            return

        sort_key = self._get_sort_key()
        if len(self._solutions) == n + 1:
            worst_sol = max(self._solutions, key=sort_key)
            self._solutions.remove(worst_sol)

        self._solutions.sort(key=sort_key)
        self._solutions = self._solutions[:n]

    def get_best_mean_solution(self) -> EliteSolution:
        """
        Returns best solution based on mean objective value.

        Only works if all elite solutions have been simulated.

        Returns
        -------
        EliteSolution
            The best-performing elite solution.
        """
        assert self.all_simulated()
        return min(self._solutions, key=lambda sol: sol.simulator.mean)

    def get_worst_mean_solution(self) -> EliteSolution:
        """
        Returns worst solution based on mean objective value.

        Only works if all elite solutions have been simulated.

        Returns
        -------
        EliteSolution
            The worst-performing elite solution.
        """
        assert self.all_simulated()
        return max(self._solutions, key=lambda sol: sol.simulator.mean)

    def get_best_deterministic_solution(self) -> EliteSolution:
        """
        Returns best solution based on deterministic objective value.

        Returns
        -------
        EliteSolution
            The best-performing elite solution.
        """
        return min(self._solutions, key=lambda sol: sol.objective)

    def print_summary(self):
        """Prints a summary of the elite solutions."""
        print("\nElite Solutions:")
        for sol in self._solutions:
            print(
                f"Solution id: {id(sol.solution)} | "
                f"Objective: {sol.objective:.2f} | "
                f"Num sims: {sol.simulator.num_sims} | "
                f"Mean: {sol.simulator.mean:.2f} | "
                f"Var: {sol.simulator.variance:.2f} | "
                f"Metadata: {sol.metadata}"
            )

    def all_simulated(self) -> bool:
        """Returns True if all solutions are simulated."""
        return all(sol.simulator.num_sims > 0 for sol in self._solutions)

    def none_simulated(self) -> bool:
        """Returns True if none of the solutions are simulated."""
        return all(sol.simulator.num_sims == 0 for sol in self._solutions)

    def simulate_to_num_sims(
        self, num_sims: int, time_limit: Optional[float] = None
    ):
        """
        Ensures all solutions are simulated up to the given number.

        Parameters
        ----------
        num_sims : int
            The number of simulations each solution should have.
        time_limit : float, optional
            Max time allowed for simulation (in seconds).
            If None, there is no time constraint.

        Notes
        -----
        - Only solutions with < `num_sims` simulations will be simulated.
        - The remaining time is passed to each simulator.
        """
        start_time = time.time()
        remaining_time = None

        for sol in self._solutions:
            current_num_sims = sol.simulator.num_sims
            if current_num_sims < num_sims:
                if time_limit is not None:
                    time_spent = time.time() - start_time
                    if time_spent >= time_limit:
                        return
                    remaining_time = max(time_limit - time_spent, 0.0)
                extra_sims = num_sims - current_num_sims
                sol.simulator.simulate(extra_sims, remaining_time)

    def simulate_to_time_limit(self, time_limit: float):
        """
        Simulates solutions until the time limit is reached.

        Parameters
        ----------
        time_limit : float
            Max time allowed for simulation (in seconds).

        Notes
        -----
        - Adds one simulation at a time in round-robin fashion.
        - Remaining time is passed to each simulator.
        """

        start_time = time.time()
        max_num_sims = max(sol.simulator.num_sims for sol in self._solutions)
        time_spent = time.time() - start_time

        while time_spent < time_limit:
            remaining_time = max(time_limit - time_spent, 0.0)
            self.simulate_to_num_sims(max_num_sims, remaining_time)
            max_num_sims += 1
            time_spent = time.time() - start_time

    def parallel_simulate_to_num_sims(
        self,
        num_sims: int,
        num_workers: int,
        time_limit: float | None = None,
    ) -> None:
        """
        Ensures all solutions are simulated in parallel up to the given number.

        Parameters
        ----------
        num_sims : int
            The number of simulations each solution should have.
        num_workers : int
            Number of parallel workers.
        time_limit : float, optional
            Max time allowed for simulation (in seconds).

        Notes
        -----
        - Only solutions with < `num_sims` simulations are processed.
        - Remaining time is fixed before dispatch, not updated during
        execution. All workers receive the same time_limit, so if tasks run in
        multiple rounds (due to limited num_workers), later tasks may exceed
        the intended limit.
        """

        args_list = []
        sol_list = []

        for sol in self._solutions:
            current_num_sims = sol.simulator.num_sims
            if current_num_sims < num_sims:
                extra_sims = num_sims - current_num_sims
                args = (sol.simulator, extra_sims, time_limit)
                args_list.append(args)
                sol_list.append(sol)  # corresponding solution

        simulators = process_map(
            self._simulate_wrapper,
            args_list,
            max_workers=num_workers,
        )

        for sol, sim in zip(sol_list, simulators, strict=True):
            sol.simulator = sim

    def parallel_simulate_to_time_limit(
        self,
        time_limit: float,
        num_workers: int,
        setup_time_process_map: Optional[float] = None,
    ) -> None:
        """
        All solutions are simulated in parallel till given time limit.

        Parameters
        ----------
        time_limit : float
            Max time allowed for simulation (in seconds).
        num_workers : int
            Number of parallel workers.
        setup_time_process_map : float, optional
            Time to set up process_map (in seconds).
        """

        args_list = []
        sol_list = []
        num_map_loops = math.ceil(len(self._solutions) / num_workers)
        if setup_time_process_map is None:
            setup_time_process_map = 0.0
        net_time_limit = time_limit - setup_time_process_map
        worker_time_limit = net_time_limit / num_map_loops
        inf = 10**18

        for sol in self._solutions:
            args = (sol.simulator, inf, worker_time_limit)
            args_list.append(args)
            sol_list.append(sol)  # corresponding solution

        simulators = process_map(
            self._simulate_wrapper,
            args_list,
            max_workers=num_workers,
        )

        for sol, sim in zip(sol_list, simulators, strict=False):
            sol.simulator = sim

    @staticmethod
    def _simulate_wrapper(args: tuple) -> Simulator:
        """
        Helper for parallel simulation of a single simulator.

        Parameters
        ----------
        args : tuple
            A tuple of (simulator, extra_sims, time_limit).

        Notes
        -----
        - Since process_map makes a copy of simulator, the original simulator
        is not modified. Therefore, the simulator needs to be returned.
        """
        simulator, extra_sims, time_limit = args
        simulator.simulate(extra_sims, time_limit=time_limit)
        return simulator
