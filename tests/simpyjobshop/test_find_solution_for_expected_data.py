from pyjobshop import Solution, TaskData
from simpyjobshop.modeling import (
    find_result_for_expected_data,
)
from tests.simpyjobshop.problems.TwoMachineTwoJobs import TwoMachinesTwoJobs


def test_find_solution_for_expected_data_two_machine_two_jobs():
    """
    Tests whether find_solution_for_expected_data() correctly finds new
    solution for expected data for a two-machines two-jobs problem.
    """

    # Create a problem
    problem = TwoMachinesTwoJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)

    # Find best solution
    result = model.solve(display=False)

    # Check whether it finds the same solution
    exp_result = find_result_for_expected_data(result.best, problem)
    assert exp_result.best == result.best
    assert exp_result.objective == result.objective

    # Define solution with task 2 -> 1 on second machine
    task1 = TaskData(
        mode=1,
        resources=[1],
        start=1,
        end=-1,  # irrelevant since fix_solution() only looks at start
    )
    task2 = TaskData(
        mode=3,
        resources=[1],
        start=0,
        end=-1,  # irrelevant since fix_solution() only looks at start
    )
    solution = Solution(tasks=[task1, task2])
    orig_modes = [task.mode for task in solution.tasks]

    # Check whether it finds the same solution
    exp_result_2 = find_result_for_expected_data(solution, problem)
    exp_solution = exp_result_2.best
    assert exp_result_2.objective == 12
    assert exp_solution.tasks[0].start == 1 + 10
    assert exp_solution.tasks[0].end == 1 + 1 + 10
    assert exp_solution.tasks[0].mode == orig_modes[0]
    assert exp_solution.tasks[1].start == 0
    assert exp_solution.tasks[1].end == 1
    assert exp_solution.tasks[1].mode == orig_modes[1]
