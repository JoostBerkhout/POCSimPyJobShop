from typing import Callable, Dict

from ortools.sat.python import cp_model

from pyjobshop import Solution
from pyjobshop.simheuristic.DataGeneratorBuilder import DataGeneratorBuilder
from pyjobshop.simheuristic.EliteSolutions import EliteSolutions
from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.solvers.ortools.Solver import Solver


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """
    Callback class that stores solutions found by ortools solver.
    """

    def __init__(
        self,
        data_generator_builder: DataGeneratorBuilder,
        evaluator: Callable[[Solution, Dict], float],
        num_sims: int,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.data_generator_builder = data_generator_builder
        self.evaluator = evaluator
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
        data_generator = self.data_generator_builder.build()  # CRN
        simulator = Simulator(data_generator, solution, self.evaluator)
        simulator.simulate(self.num_sims)
        self.solutions.add(solution, simulator, self.objective_value)
