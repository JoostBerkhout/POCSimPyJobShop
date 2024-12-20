import wandb
from ortools.sat.python import cp_model

from pyjobshop.simheuristic.EliteSolutions import EliteSolutions
from pyjobshop.simheuristic.problems.Problem import Problem
from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.utils import find_schedule_per_resource
from pyjobshop.solvers.ortools.Solver import Solver


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """
    Callback class that stores solutions found by ortools solver.
    """

    def __init__(self, problem: Problem, num_sims: int):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.problem = problem
        self.num_sims = num_sims
        self.solutions = EliteSolutions()
        self.solver: Solver | None = None

    def set_solver(self, solver: Solver):
        """
        Can be used to set the solver that is using this callback.
        """

        self.solver = solver

    def on_solution_callback(self):
        """
        Callback method that is called by ortools when a new solution is found.
        """

        solution = self.solver._convert_to_solution(self)
        schedule = find_schedule_per_resource(solution)
        if wandb.run is not None:
            wandb.log(
                {
                    "Objective new candidate": self.objective_value,
                    "Current bound": self.best_objective_bound,
                }
            )
        if self.solutions.is_new_schedule(schedule):
            simulator = Simulator(self.problem, solution)
            simulator.simulate(self.num_sims)
            metadata = {
                "current_time": self.WallTime(),
                "current_bound": self.best_objective_bound,
            }
            self.solutions.add(
                solution=solution,
                simulator=simulator,
                objective=self.objective_value,
                schedule=schedule,
                metadata=metadata,
            )
            if wandb.run is not None:
                best_elite_sol = self.solutions.get_best_elite_solution()
                wandb.log(
                    {
                        "Mean objective new candidate": simulator.mean,
                        "Best mean objective": best_elite_sol.simulator.mean,
                    }
                )
