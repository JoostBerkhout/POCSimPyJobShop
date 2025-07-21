import pytest
from numpy.testing import assert_, assert_equal

from pyjobshop import Model, solve
from pyjobshop.Solution import Solution, TaskData


def test_solve(small, solver):
    """
    Tests that solve returns the expected result.
    """
    result = solve(small, solver)

    assert_equal(result.status.value, "Optimal")
    assert_(result.runtime < 1)
    assert_equal(result.objective, 3)


def test_solve_unknown_solver(small):
    """
    Tests that an unknown solver raises a ValueError.
    """
    with pytest.raises(ValueError):
        solve(small, "unknown")


def test_solve_log(small, solver, capfd):
    """
    Tests that setting the log flag correctly show solver output.
    """
    solve(small, solver, display=True)
    printed = capfd.readouterr().out
    assert_(printed != "")

    solve(small, solver, display=False)
    printed = capfd.readouterr().out
    assert_equal(printed, "")


def test_solve_time_limit(small, capfd):
    """
    Tests the log that the time limit is set. No test for CP Optimizer
    because it does not log this setting.
    """
    solve(small, "ortools", time_limit=1.2, display=True)
    printed = capfd.readouterr().out
    assert_("max_time_in_seconds: 1.2" in printed)


def test_solve_num_workers(small, solver, capfd):
    """
    Tests the log that the ``num_workers`` parameter is correctly set.
    """
    solver2msg = {
        "ortools": "num_workers: 2",
        "cpoptimizer": "Using parallel search with 2 workers.",
    }
    msg = solver2msg[solver]

    solve(small, solver, num_workers=2, display=True)
    printed = capfd.readouterr().out
    assert_(msg in printed)


def test_solve_initial_solution(small, solver, capfd):
    """
    Tests that the log message is correct when an initial solution is provided.
    """
    solver2msg = {
        # Not all variables are hinted so this message is correct.
        "ortools": "The solution hint is incomplete",
        "cpoptimizer": "Starting point is complete and consistent with constraints.",  # noqa
    }
    msg = solver2msg[solver]

    init = Solution([TaskData(0, [0], 0, 1), TaskData(1, [0], 1, 3)])
    solve(small, solver, display=True, initial_solution=init)
    printed = capfd.readouterr().out
    assert_(msg in printed)


def test_solve_exclude_solutions_or_tools(small):
    """
    Tests that the exclude_solutions parameter works with OR-Tools.
    """
    solver = "ortools"

    result1 = solve(small, solver)
    assert_equal(result1.status.value, "Optimal")
    assert_(result1.runtime < 1)
    assert_equal(result1.objective, 3)

    # Now exclude the previous solution
    result2 = solve(small, solver, exclude_solutions=[result1.best])
    assert_equal(result2.status.value, "Optimal")
    assert_(result2.runtime < 1)
    assert_equal(result2.objective, 3)
    assert result1.best != result2.best, "The solutions should be different."

    # Since there are only two solutions, we can do some tests:

    # i) Excluding the second solution should return the first one
    result3 = solve(small, solver, exclude_solutions=[result2.best])
    assert_equal(result3.status.value, "Optimal")
    assert_(result3.runtime < 1)
    assert_equal(result3.objective, 3)
    assert result3.best == result1.best, "The solutions should be equal."

    # ii) Excluding both solutions should lead to infeasibility
    excl_sols = [result1.best, result2.best]
    result4 = solve(small, solver, exclude_solutions=excl_sols)
    assert_equal(result4.status.value, "Infeasible")


def test_solve_exclude_solutions_or_tools_two_machines(small_two_machines):
    """
    Tests that the exclude_solutions parameter works with OR-Tools in case
    of multiple machines.
    """
    solver = "ortools"

    result1 = solve(small_two_machines, solver)
    assert_equal(result1.status.value, "Optimal")
    assert_(result1.runtime < 1)
    assert_equal(result1.objective, 1)
    exp_opt1 = Solution([TaskData(0, [0], 0, 1), TaskData(3, [1], 0, 1)])
    assert result1.best == exp_opt1

    # Exclude the previous solution
    result2 = solve(
        small_two_machines, solver, exclude_solutions=[result1.best]
    )
    assert_equal(result2.status.value, "Optimal")
    assert_(result2.runtime < 1)
    assert_equal(result2.objective, 2)
    exp_opt2 = Solution([TaskData(1, [1], 0, 2), TaskData(2, [0], 0, 2)])
    assert result2.best == exp_opt2

    # Exclude one-by-one the remaining solutions
    sols_on_same_machine = [
        Solution([TaskData(0, [0], 0, 1), TaskData(2, [0], 1, 3)]),
        Solution([TaskData(0, [0], 2, 3), TaskData(2, [0], 0, 2)]),
        Solution([TaskData(1, [1], 0, 2), TaskData(3, [1], 2, 3)]),
        Solution([TaskData(1, [1], 1, 3), TaskData(3, [1], 0, 1)]),
    ]
    for idx, sol in enumerate(sols_on_same_machine):
        copy_sols_on_same_machine = sols_on_same_machine.copy()
        copy_sols_on_same_machine.pop(idx)
        # Exclude all solutions except for the last solution sol
        excl_sols = copy_sols_on_same_machine + [result1.best, result2.best]
        result = solve(small_two_machines, solver, exclude_solutions=excl_sols)
        assert_equal(result.status.value, "Optimal")
        assert_(result.runtime < 1)
        assert_equal(result.objective, 3)
        assert result.best == sol

    # Lastly, excluding all solutions should lead to infeasibility
    excl_sols = sols_on_same_machine + [result1.best, result2.best]
    result = solve(small_two_machines, solver, exclude_solutions=excl_sols)
    assert_equal(result.status.value, "Infeasible")


def test_solve_exclude_solutions_or_tools_0_durations():
    """
    Tests exclude_solutions in solve with OR-Tools in case durations of 0.
    """

    # Specify model for one machine and 2 tasks of 0 durations.
    model = Model()
    job = model.add_job()
    machine = model.add_machine()
    tasks = [model.add_task(job=job) for _ in range(2)]
    for task, duration in zip(tasks, [0, 0], strict=False):
        model.add_mode(task, machine, duration)

    # Find the optimal solution
    result = solve(model.data(), "ortools")
    assert_equal(result.status.value, "Optimal")
    assert_(result.runtime < 1)
    assert_equal(result.objective, 0)

    # Excluding previous solution makes it infeasible since durations are 0
    result2 = solve(model.data(), "ortools", exclude_solutions=[result.best])
    assert_equal(result2.status.value, "Infeasible")


def test_solve_exclude_solutions_or_tools_0_duration():
    """
    Tests exclude_solutions in solve with OR-Tools in case of duration 0.
    """

    # Specify model for one machine and 2 tasks of 0 and 1 durations, resp.
    model = Model()
    job = model.add_job()
    machine = model.add_machine()
    tasks = [model.add_task(job=job) for _ in range(2)]
    for task, duration in zip(tasks, [0, 1], strict=False):
        model.add_mode(task, machine, duration)

    # State all solutions
    sol1 = Solution([TaskData(0, [0], 0, 0), TaskData(1, [0], 0, 1)])
    sol2 = Solution([TaskData(0, [0], 1, 1), TaskData(1, [0], 0, 1)])

    # Solve the model ruling out one solution -> should return the other
    for s1, s2 in [(sol1, sol2), (sol2, sol1)]:
        result = solve(model.data(), "ortools", exclude_solutions=[s1])
        assert_equal(result.status.value, "Optimal")
        assert_(result.runtime < 1)
        assert_equal(result.objective, 1)
        assert result.best == s2, f"{result.best.tasks} and {s2.tasks}"

    # Excluding both solutions should make it infeasible
    result = solve(model.data(), "ortools", exclude_solutions=[sol1, sol2])
    assert_equal(result.status.value, "Infeasible")


def test_solve_additional_params(small, solver, capfd):
    """
    Tests the solve method with additional parameters can override the
    parameters supported by ``solve``.
    """
    solver2param_value = {
        "ortools": ("log_search_progress", True),
        "cpoptimizer": ("LogVerbosity", "Terse"),
    }
    param, value = solver2param_value[solver]

    # Let's test that setting log to False will not print anything.
    solve(small, solver, display=False)
    printed = capfd.readouterr().out
    assert_equal(printed, "")

    # Now we set the corresponding log parameter using the additional keyword
    # argument, which will override the earlier setting of log being False.
    solve(small, solver, display=False, **{param: value})
    printed = capfd.readouterr().out
    assert_(printed != "")
