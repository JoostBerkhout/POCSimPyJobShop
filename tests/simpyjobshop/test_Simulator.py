import time

from simpyjobshop.Simulator import Simulator
from simpyjobshop.SolutionCallback import SolutionCallback
from tests.simpyjobshop.problems.OneMachineTenJobs import OneMachineTenJobs


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

    elite_solution = callback.solutions.get_best_mean_solution()

    solution = elite_solution.solution
    simulator = elite_solution.simulator

    # Make a new simulator and simulate the solution
    new_simulator = Simulator(problem, solution)
    new_simulator.simulate(num_sims)

    # Test whether simulation results coincide
    assert num_sims == simulator.num_sims
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

    # Test simulation with a time_limit
    current_num_sims = simulator.num_sims
    time_limit = 0.0
    simulator.simulate(num_sims, time_limit)
    assert simulator.num_sims == current_num_sims

    # Test simulation with a very small time limit
    current_num_sims = simulator.num_sims
    time_limit = 0.0001  # allows only for 1 simulation
    simulator.simulate(100000000, time_limit)
    assert simulator.num_sims == current_num_sims + 1

    # Test simulation with a small time limit
    time_limit = 0.01
    start_time = time.time()
    simulator.simulate(100000000, time_limit)
    elapsed_time = time.time() - start_time
    assert elapsed_time > time_limit
    assert elapsed_time < time_limit + 1.0
