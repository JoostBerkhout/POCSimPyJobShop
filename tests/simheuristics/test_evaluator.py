from pyjobshop import Solution, TaskData
from simpyjobshop.evaluator import evaluator
from tests.simheuristics.problems.OneMachineTwoJobs import OneMachineTwoJobs
from tests.simheuristics.problems.TwoMachineTwoJobs import TwoMachinesTwoJobs


def test_evaluator_machine_two_jobs():
    """
    Tests whether evaluator() finds correct objective for 1 machine, 2 jobs.
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

    solution = Solution(tasks=[task1, task2])
    problem = OneMachineTwoJobs()
    data = problem.build_data_generator().int_mean()
    objective = evaluator(solution, problem, data)

    assert objective == 12


def test_evaluator_two_machines_two_jobs():
    """
    Tests whether evaluator() finds correct objective for 2 machines, 2 jobs.
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

        solution = Solution(tasks=[task1, task2])
        problem = TwoMachinesTwoJobs()
        data = problem.build_data_generator().int_mean()
        objective = evaluator(solution, problem, data)

        assert objective == 2

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

        solution = Solution(tasks=[task1, task2])
        problem = TwoMachinesTwoJobs()
        data = problem.build_data_generator().int_mean()
        objective = evaluator(solution, problem, data)

        assert objective == 12

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

        solution = Solution(tasks=[task1, task2])
        problem = TwoMachinesTwoJobs()
        data = problem.build_data_generator().int_mean()
        objective = evaluator(solution, problem, data)

        assert objective == 1
