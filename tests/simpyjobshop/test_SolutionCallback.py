import time

from simpyjobshop.SolutionCallback import SolutionCallback
from tests.simpyjobshop.problems.OneMachineTenJobs import OneMachineTenJobs
from tests.simpyjobshop.problems.OneMachineTwoJobs import OneMachineTwoJobs


def test_solution_callback():
    """
    Tests SolutionCallback.
    """

    # Create a model
    problem = OneMachineTenJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.quantile(0.7)
    model = problem.concrete_model(data)

    # Solve the problem
    num_sims = 3
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
    assert callback.solutions.all_simulated()
    assert not callback.solutions.none_simulated()

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
        callback=callback, display=False, time_limit=time_limit, num_workers=1
    )

    # Tests whether SolutionCallback stops in time, non are simulated and size
    assert 0 < time.time() - start_time < time_limit + 0.1
    assert callback.solutions.none_simulated()
    assert not callback.solutions.all_simulated()
    assert len(callback.solutions) == 3


def test_solution_callback_finds_all():
    """
    Tests to see whether all solutions are found by solution callback.
    """

    # Create a problem
    problem = OneMachineTwoJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)

    # Find all solutions with makespan <= 12
    model.set_objective_bound(12)
    callback = SolutionCallback(problem, num_sims=1)
    result = model.solve(
        callback=callback,
        display=True,
        num_workers=1,
        enumerate_all_solutions=True,  # OR-tools will enumerate all solutions
    )

    # Check if the result is optimal and solution correct
    assert result.status.value == "Optimal"
    elites = callback.solutions
    assert len(elites) == 2
    schedules = {tuple(es.schedule[0]) for es in elites}
    assert schedules == {(0, 1), (1, 0)}

    # Find all solutions with makespan <= 11
    model.set_objective_bound(11)
    callback = SolutionCallback(problem, num_sims=1)
    result = model.solve(
        callback=callback,
        display=True,
        num_workers=1,
        enumerate_all_solutions=True,  # OR-tools will enumerate all solutions
    )

    # Check if the result is optimal and solution correct
    assert result.status.value == "Optimal"
    elites = callback.solutions
    assert len(elites) == 1
    schedule = elites[0].schedule[0]
    assert schedule == [0, 1]
