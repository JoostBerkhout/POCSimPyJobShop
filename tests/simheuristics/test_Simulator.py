from pyjobshop.simheuristic.Simulator import Simulator
from pyjobshop.simheuristic.SolutionCallback import SolutionCallback
from tests.simheuristics.problems.OneMachineTenJobs import OneMachineTenJobs


def test_simulator():
    """
    Tests whether solving a one-machine problem with two jobs works.
    """

    # Create a model
    problem = OneMachineTenJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.quantile(0.7)
    model = problem.concrete_model(data)

    # Solve the problem with a callback for solution generation
    num_sims = 2
    callback = SolutionCallback(problem, num_sims=num_sims)
    model.solve(callback=callback, display=False)

    for elite_solution in callback.solutions.elite_solutions.values():
        solution = elite_solution.solution
        simulator = elite_solution.simulator

        # Make a new simulator and simulate the solution
        new_simulator = Simulator(problem, solution)
        new_simulator.simulate(num_sims)

        # Test whether simulation results coincide
        assert new_simulator.num_sims == simulator.num_sims
        assert new_simulator.mean == simulator.mean
        assert new_simulator.variance == simulator.variance

        # Test whether simulation results change after simulating
        new_simulator.simulate(3)
        assert new_simulator.num_sims == simulator.num_sims + 3
        assert new_simulator.mean != simulator.mean
        assert new_simulator.variance != simulator.variance

        # Test whether simulation results become the same again
        simulator.simulate(3)
        assert new_simulator.num_sims == simulator.num_sims
        assert new_simulator.mean == simulator.mean
        assert new_simulator.variance == simulator.variance
