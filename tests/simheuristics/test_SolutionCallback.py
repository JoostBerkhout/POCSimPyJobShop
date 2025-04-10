import time

from pyjobshop.simheuristic.SolutionCallback import SolutionCallback
from tests.simheuristics.problems.OneMachineTenJobs import OneMachineTenJobs


def test_SolutionCallback():
    """
    Tests SolutionCallback.
    """

    # Create a model
    problem = OneMachineTenJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.quantile(0.7)
    model = problem.concrete_model(data)

    # Solve the problem with a callback for solution generation
    num_sims = 10000000
    time_limit = 0.1
    callback = SolutionCallback(
        problem,
        num_sims=num_sims,
        stop_time=time.time() + time_limit,
    )
    start_time = time.time()
    model.solve(
        callback=callback,
        display=True,
        time_limit=time_limit,
        num_workers=1,
    )

    # Tests whether SolutionCallback stops in time and all simulated
    assert 0 < time.time() - start_time < time_limit + 0.1
    assert len(callback.solutions.elite_solutions) == 1
    assert callback.solutions.all_simulated()

    # Solve the problem again with a different callback
    time_limit = 0.1
    start_time_exp = time.time()
    callback = SolutionCallback(
        problem,
        num_sims=num_sims,
        start_time_exp=start_time_exp,
        stop_time=start_time_exp + time_limit,
        start_time_sims=start_time_exp + 1.5 * time_limit,  # no simulation
        max_size_elite_set=3,
    )
    start_time = time.time()
    model.solve(
        callback=callback, display=True, time_limit=time_limit, num_workers=1
    )

    # Tests whether SolutionCallback stops in time, non are simulated and size
    assert time.time() - start_time < time_limit + 0.1
    elite_sols = callback.solutions.elite_solutions.values()
    num_sol_simulated = sum([sol.simulator.num_sims > 0 for sol in elite_sols])
    assert num_sol_simulated == 0
    assert len(callback.solutions.elite_solutions) == 3
