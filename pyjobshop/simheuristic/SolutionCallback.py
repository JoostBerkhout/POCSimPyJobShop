import time

import wandb
from ortools.sat.python import cp_model

from pyjobshop.simheuristic.EliteSolutions import EliteSolutions
from pyjobshop.simheuristic.problems.Problem import Problem
from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.utils import find_schedule_per_resource
from pyjobshop.solvers.ortools.Solver import Solver


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """
    Callback for CP-SAT solver to collect and simulate elite solutions.

    Parameters
    ----------
    problem : Problem
        The problem instance being solved.
    num_sims : int
        Number of simulation replications to perform for each new elite
        solution (if simulation is started).
    start_time_exp : float, optional
        Time when the experiment started, used to compute elapsed time. If
        None, the current time is used.
    start_time_sims : float, optional
        Time after which simulation of solutions is allowed. If None,
        simulations are triggered automatically.
    max_size_elite_set : int, optional
        Maximum number of elite solutions to retain. If None, all unique
        solutions are kept.
    stop_time : float, optional
        Time after which the solver should stop execution.
    """

    def __init__(
        self,
        problem: Problem,
        num_sims: int,
        start_time_exp: float | None = None,
        start_time_sims: float | None = None,
        max_size_elite_set: int | None = None,
        stop_time: float | None = None,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.problem = problem
        self.num_sims = num_sims
        if start_time_exp is None:
            self.start_time_exp = time.time()
        else:
            self.start_time_exp = start_time_exp
        self.start_time_sims = start_time_sims
        self.max_size_elite_set = max_size_elite_set
        self.stop_time = stop_time
        self.solutions = EliteSolutions()
        self.solver: Solver | None = None

    def set_solver(self, solver: Solver):
        """Can be used to set the solver that is using this callback."""
        self.solver = solver

    @property
    def time_spent(self) -> float:
        """Elapsed time since experiment started."""
        return time.time() - self.start_time_exp

    @property
    def simulation_started(self) -> bool:
        """Returns True if enough time has passed to begin simulations."""
        if self.start_time_sims is None:
            return True
        return time.time() >= self.start_time_sims

    @property
    def remaining_time(self) -> float | None:
        """Returns remaining time till stop_time, or None if not set."""
        if self.stop_time is None:
            return None
        else:
            return max(self.stop_time - time.time(), 0.0)

    def on_solution_callback(self):
        """Method that is called by ortools when a new solution is found."""

        try:
            solution = self.solver._convert_to_solution(self)
            schedule = find_schedule_per_resource(solution)

            if self.solutions.is_new_schedule(schedule):
                simulator = Simulator(self.problem, solution)
                if self.simulation_started:
                    simulator.simulate(self.num_sims, self.remaining_time)

                self.solutions.add(
                    solution=solution,
                    simulator=simulator,
                    objective=self.objective_value,
                    schedule=schedule,
                    metadata={
                        "current_time": time.time() - self.start_time_exp,
                        "current_bound": self.best_objective_bound,
                    },
                )

                self._log_to_wandb(simulator)

                if self.max_size_elite_set is not None:
                    self.solutions.keep_top_n(self.max_size_elite_set)

        except Exception as e:
            print(
                f"An error occured in on_solution_callback(). We catch it"
                f" because else it will continue indefinitely. Msg: {e}"
            )
            return

    def _log_to_wandb(self, simulator: Simulator):
        # Logs relevant metrics to Weights & Biases.

        if wandb.run is None:
            return

        log_data = {
            "Time (in seconds)": self.time_spent,
            "Objective new candidate": self.objective_value,
            "Current bound": self.best_objective_bound,
        }

        if self.simulation_started:
            self.solutions.simulate_to_num_sims(
                self.num_sims,
                self.remaining_time,
            )
            if self.solutions.all_simulated():
                best_elite = self.solutions.get_best_elite_solution()
                log_data.update(
                    {
                        "Mean objective new candidate": simulator.mean,
                        "Best mean objective": best_elite.simulator.mean,
                    }
                )
            elif simulator.num_sims > 0:
                log_data.update(
                    {
                        "Mean objective new candidate": simulator.mean,
                    }
                )

        wandb.log(log_data)
