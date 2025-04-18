from pyjobshop import Solution, TaskData
from simpyjobshop.modeling import find_solution_for_other_data
from simpyjobshop.utils import find_schedule_per_resource
from tests.simpyjobshop.problems.OneMachineTwoJobs import OneMachineTwoJobs
from tests.simpyjobshop.problems.TwoMachineTwoJobs import TwoMachinesTwoJobs


def test_find_solution_for_other_data_one_machine_two_jobs():
    """
    Tests whether find_schedule_per_resource() correctly finds new solution
    for different data for a one-machine two-jobs problem.
    """

    # Create a problem
    problem = OneMachineTwoJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)

    # Find best solution
    result = model.solve(display=False)
    solution = result.best
    schedule = find_schedule_per_resource(solution)[0]
    first = schedule[0]
    second = schedule[1]

    # Now change the data and find new solution
    duration1 = 21
    duration2 = 11
    data[f"duration_{first}"] = duration1
    data[f"duration_{second}"] = duration2
    new_solution, obj = find_solution_for_other_data(solution, problem, data)
    new_schedule = find_schedule_per_resource(new_solution)[0]

    # Check if the new solution is correct
    assert schedule == new_schedule
    assert obj == 32
    assert new_solution.tasks[first].start == 0
    assert new_solution.tasks[first].end == duration1
    assert new_solution.tasks[second].start == duration1
    assert new_solution.tasks[second].end == duration1 + duration2


def test_find_solution_for_other_data_two_machines_two_jobs():
    """
    Tests whether find_schedule_per_resource() correctly finds new solution
    for different data for a two-machines two-jobs problem.
    """

    # Create a problem
    problem = TwoMachinesTwoJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)

    # Find best solution
    result = model.solve(display=False)
    assert result.objective == 1
    solution = result.best
    schedule = find_schedule_per_resource(solution)

    # Now change the data and find new solution
    duration1 = 21
    duration2 = 11
    data["duration_0"] = duration1
    data["duration_1"] = duration2
    new_solution, obj = find_solution_for_other_data(solution, problem, data)
    new_schedule = find_schedule_per_resource(new_solution)

    # Check if the new solution is correct
    assert schedule == new_schedule
    assert obj == 21
    assert new_solution.tasks[0].start == 0
    assert new_solution.tasks[0].end == duration1
    assert new_solution.tasks[1].start == 0
    assert new_solution.tasks[1].end == duration2

    # Define solution with task 1 -> 2 on second machine
    task1 = TaskData(
        mode=1,
        resources=[1],
        start=0,
        end=-1,  # irrelevant since fix_solution() only looks at start
    )
    task2 = TaskData(
        mode=3,
        resources=[1],
        start=1,
        end=-1,  # irrelevant since fix_solution() only looks at start
    )
    solution = Solution(tasks=[task1, task2])
    schedule = find_schedule_per_resource(solution)

    # Find solution for changed data
    new_solution, obj = find_solution_for_other_data(solution, problem, data)
    new_schedule = find_schedule_per_resource(new_solution)

    # Check if the new solution is correct
    assert schedule == new_schedule
    assert obj == 32
    assert new_solution.tasks[0].start == 0
    assert new_solution.tasks[0].end == duration1
    assert new_solution.tasks[1].start == duration1
    assert new_solution.tasks[1].end == duration1 + duration2

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
    schedule = find_schedule_per_resource(solution)

    # Find solution for changed data
    new_solution, obj = find_solution_for_other_data(solution, problem, data)
    new_schedule = find_schedule_per_resource(new_solution)

    # Check if the new solution is correct
    assert schedule == new_schedule
    assert obj == 32 + 10
    assert new_solution.tasks[0].start == duration2 + 10
    assert new_solution.tasks[0].end == duration1 + duration2 + 10
    assert new_solution.tasks[1].start == 0
    assert new_solution.tasks[1].end == duration2
