import time

import wandb
from ortools.sat.python import cp_model

from pyjobshop.solvers.ortools.Solver import Solver
from simpyjobshop.EliteSolutions import EliteSolutions
from simpyjobshop.problems.Problem import Problem
from simpyjobshop.Simulator import Simulator
from simpyjobshop.utils import find_schedule_per_resource


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
        if start_time_sims is None:
            self.start_time_sims = self.start_time_exp
        else:
            self.start_time_sims = start_time_sims
        self.max_size_elite_set = max_size_elite_set
        if stop_time is None:
            self.stop_time = float("inf")
        else:
            self.stop_time = stop_time
        self.solutions = EliteSolutions()
        self.solver: Solver | None = None

        self.callback_log: list[str] = []

    def _log_event(self, message: str):
        timestamp = time.time() - self.start_time_exp
        self.callback_log.append(f"[{timestamp:.2f}s] {message}")

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
        return time.time() >= self.start_time_sims

    @property
    def remaining_time(self) -> float | None:
        """Returns remaining time till stop_time, or None if not set."""
        return max(self.stop_time - time.time(), 0.0)

    def on_solution_callback(self):
        """Method that is called by ortools when a new solution is found."""

        try:
            self._log_event("Solution callback triggered.")
            solution = self.solver._convert_to_solution(self)
            schedule = find_schedule_per_resource(solution)
            time_left = self.remaining_time > 0
            new_schedule = self.solutions.is_new_schedule(schedule)

            if time_left and new_schedule:
                self._log_event("New unique schedule and time left.")
                simulator = Simulator(self.problem, solution)
                self._log_event("Simulator loaded.")
                if self.simulation_started:
                    self._log_event("Simulation started.")
                    simulator.simulate(self.num_sims)
                else:
                    self._log_event(
                        f"Simulation skipped, not yet allowed. Start time "
                        f"sims. = {self.start_time_sims - self.start_time_exp}"
                    )

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
                    self._log_event("Elite set trimmed.")

            elif not new_schedule:
                self._log_event("Duplicate schedule. Ignored.")
            elif not time_left:
                self._log_event("Time limit reached. Solution ignored.")

        except Exception as e:
            print(
                f"An error occurred in on_solution_callback(). We catch it"
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
            self._log_event(
                f"wandb log: Starting to simulate. "
                f"all_simulated = {self.solutions.all_simulated()}"
            )
            self.solutions.simulate_to_num_sims(self.num_sims)
            self._log_event("wandb log: Simulation ended.")
            best_elite = self.solutions.get_best_mean_solution()
            log_data.update(
                {
                    "Mean objective new candidate": simulator.mean,
                    "Best mean objective": best_elite.simulator.mean,
                }
            )

        wandb.log(log_data)

    def get_log_str(self) -> str:
        log_lines = ["\nCallback Log:", "-" * 40]
        log_lines.extend(self.callback_log)
        log_lines.append("-" * 40)
        return "\n".join(log_lines)

    def print_log(self):
        print(self.get_log_str())
