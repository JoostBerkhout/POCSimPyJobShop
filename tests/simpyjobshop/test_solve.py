from tests.simpyjobshop.problems.OneMachineTwoJobs import OneMachineTwoJobs
from tests.simpyjobshop.problems.TwoMachineTwoJobs import TwoMachinesTwoJobs


def test_one_machine():
    """
    Tests whether solving a one-machine problem with two jobs works.
    """

    problem = OneMachineTwoJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)
    result = model.solve(display=False)

    assert result.objective == 2
    assert result.best.tasks[0].start == 0
    assert result.best.tasks[1].start == 1

    # Now same problem with total tardiness as objective
    model.set_objective(weight_total_tardiness=1)
    result = model.solve(display=False)

    assert result.objective == 0
    assert result.best.tasks[0].start == 11
    assert result.best.tasks[1].start == 0


def test_two_machines():
    """
    Tests whether solving a two-machines problem with two jobs works.
    """

    problem = TwoMachinesTwoJobs()
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)
    result = model.solve(display=False)

    assert result.objective == 1
    assert result.best.tasks[0].start == 0
    assert result.best.tasks[1].start == 0

    # Now same problem with total tardiness as objective
    model.set_objective(weight_total_tardiness=1)
    result = model.solve(display=False)

    assert result.objective == 0
    assert result.best.tasks[0].start == 0
    assert result.best.tasks[1].start == 0
    resources_used = [result.best.tasks[_].resources[0] for _ in range(2)]
    assert resources_used[0] != resources_used[1]
