from typing import Dict

import pytest

from pyjobshop import Solution, TaskData
from simpyjobshop.modeling import fix_solution
from simpyjobshop.SolutionCallback import SolutionCallback
from simpyjobshop.utils import find_schedule_per_resource
from tests.simpyjobshop.problems.OneMachineTenJobs import OneMachineTenJobs
from tests.simpyjobshop.problems.OneMachineTwoJobs import OneMachineTwoJobs
from tests.simpyjobshop.problems.TwoMachineTwoJobs import TwoMachinesTwoJobs


@pytest.fixture(params=[OneMachineTwoJobs(), TwoMachinesTwoJobs()])
def problem(request):
    """
    Fixture that provides individual problem instances for testing.

    The `params` argument cycles through instances of specific problems.
    """
    return request.param


def equal_solutions(
    solution1: Solution, solution2: Solution, map_to_sol1_modes: Dict
) -> bool:
    """
    Compares two solutions to see if they are equal. The mode indices in
    solution2 are mapped to the mode indices in solution1 using the given
    dictionary. Useful when fix_solution() is used to fix a solution and
    it changes the modes indices used (in solution2).
    """

    for task1, task2 in zip(solution1.tasks, solution2.tasks, strict=True):
        if task1.resources != task2.resources:
            return False
        if task1.start != task2.start:
            return False
        if task1.end != task2.end:
            return False
        if task1.mode != map_to_sol1_modes[task2.mode]:
            return False

    return True


def test_fix_one_machine_two_jobs():
    """
    Tests whether fix_solution() correctly fixes solutions for a one-machine
    """

    # Define two tasks
    task1 = TaskData(
        mode=0,
        resources=[0],
        start=1,
        end=2,  # irrelevant since fix_solution() only looks at start
    )
    task2 = TaskData(
        mode=1,
        resources=[0],
        start=0,
        end=1,  # irrelevant since fix_solution() only looks at start
    )

    # Create a solution with these tasks
    solution = Solution(tasks=[task1, task2])

    # Create a problem
    problem = OneMachineTwoJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)

    # Fix the solution
    model = fix_solution(solution, model)
    result = model.solve(display=False)

    # Check if the solution is correct
    assert result.objective == 12
    assert result.best.tasks[0].start == 11
    assert result.best.tasks[1].start == 0


def test_fix_two_machines_two_jobs():
    """
    Tests whether fix_solution() correctly fixes solutions for two-machines.
    """

    # Define two tasks on same machine, resp.
    for machine in [0, 1]:
        task1 = TaskData(
            mode=0 + machine,
            resources=[machine],
            start=0,
            end=-1,  # irrelevant since fix_solution() only looks at start
        )
        task2 = TaskData(
            mode=2 + machine,
            resources=[machine],
            start=1,
            end=-1,  # irrelevant since fix_solution() only looks at start
        )

        # Create a solution with these tasks
        solution = Solution(tasks=[task1, task2])

        # Create a problem
        problem = TwoMachinesTwoJobs()
        data_generator = problem.build_data_generator()
        data = data_generator.int_mean()
        model = problem.concrete_model(data)

        # Fix the solution
        model = fix_solution(solution, model)
        result = model.solve(display=False)

        # Check if the solution is correct
        assert result.objective == 2
        assert result.best.tasks[0].start == 0
        assert result.best.tasks[1].start == 1
        assert result.best.tasks[0].resources == result.best.tasks[1].resources
        assert result.best.tasks[0].resources == [machine]

        # Now define two tasks on machine 1 in reverse order, resp.
        task1 = TaskData(
            mode=0 + machine,
            resources=[machine],
            start=1,
            end=-1,  # irrelevant since fix_solution() only looks at start
        )
        task2 = TaskData(
            mode=2 + machine,
            resources=[machine],
            start=0,
            end=-1,  # irrelevant since fix_solution() only looks at start
        )

        # Create a solution with these tasks
        solution = Solution(tasks=[task1, task2])

        # Create a problem
        problem = TwoMachinesTwoJobs()
        data_generator = problem.build_data_generator()
        data = data_generator.int_mean()
        model = problem.concrete_model(data)

        # Fix the solution
        model = fix_solution(solution, model)
        result = model.solve(display=False)

        # Check if the solution is correct
        assert result.objective == 12
        assert result.best.tasks[0].start == 11
        assert result.best.tasks[1].start == 0
        assert result.best.tasks[0].resources == result.best.tasks[1].resources
        assert result.best.tasks[0].resources == [machine]

    # Define two tasks on different machines
    for machine_task1 in [0, 1]:
        task1 = TaskData(
            mode=machine_task1,
            resources=[machine_task1],
            start=100,  # irrelevant since there will be only 1 job on machine
            end=-1,  # irrelevant since fix_solution() only looks at start
        )
        task2 = TaskData(
            mode=1 - machine_task1,
            resources=[1 - machine_task1],
            start=100,  # irrelevant since there will be only 1 job on machine
            end=-1,  # irrelevant since fix_solution() only looks at start
        )

        # Create a solution with these tasks
        solution = Solution(tasks=[task1, task2])

        # Create a problem
        problem = TwoMachinesTwoJobs()
        data_generator = problem.build_data_generator()
        data = data_generator.int_mean()
        model = problem.concrete_model(data)

        # Fix the solution
        model = fix_solution(solution, model)
        result = model.solve(display=False)

        # Check if the solution is correct
        assert result.objective == 1
        assert result.best.tasks[0].start == 0
        assert result.best.tasks[1].start == 0
        assert result.best.tasks[0].resources == [machine_task1]
        assert result.best.tasks[1].resources == [1 - machine_task1]


def test_fix_solution(problem):
    """
    Generic tests if the fix_solution function correctly fixes solutions.
    """

    # Solve a problem to get a solution
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)
    result = model.solve(display=False)

    # Solve the same problem with the solution given
    data = data_generator.int_mean()
    model = problem.concrete_model(data)
    model = fix_solution(result.best, model)
    result2 = model.solve(display=False)
    assert equal_solutions(result.best, result2.best, model._map_to_old_mode)
    assert result.objective == result2.objective

    # Solve problem with no objective and fixing solution
    data = data_generator.int_mean()
    model = problem.concrete_model(data)
    model = fix_solution(result.best, model)
    model.set_objective()
    result2 = model.solve(display=False)

    assert equal_solutions(result.best, result2.best, model._map_to_old_mode)

    # Solve problem with no objective to get some random solution
    data = data_generator.int_mean()
    model = problem.concrete_model(data)
    model.set_objective()
    result = model.solve(display=False)

    # Solve problem (with original objective) with previous solution fixed
    data = data_generator.int_mean()
    model = problem.concrete_model(data)
    model = fix_solution(result.best, model)
    result2 = model.solve(display=False)

    assert equal_solutions(result.best, result2.best, model._map_to_old_mode)


def test_fix_callback_solutions():
    """
    Test to check whether callback solutions are correctly fixed.
    """

    # Create a model
    problem = OneMachineTenJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.quantile(0.7)
    model = problem.concrete_model(data)

    # Solve the problem with a callback
    callback = SolutionCallback(problem, num_sims=0)
    result = model.solve(callback=callback, display=False)

    # Test whether intermediate results are correct
    objective = None
    for elite_solution in callback.solutions:
        solution = elite_solution.solution
        simulator = elite_solution.simulator
        objective = elite_solution.objective
        model = problem.concrete_model(data)
        model = fix_solution(solution, model)
        result = model.solve(display=False)
        sol_schedule = find_schedule_per_resource(solution)[0]
        res_schedule = find_schedule_per_resource(result.best)[0]

        assert sol_schedule == res_schedule
        # solution is an intermediate solution so its objective can be worse
        assert result.objective <= objective
        assert simulator.num_sims == 0
