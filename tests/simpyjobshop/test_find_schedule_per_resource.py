from pyjobshop import Solution, TaskData
from simpyjobshop.utils import find_schedule_per_resource


def test_find_schedule_per_resource():
    """
    Tests the find_schedule_per_resource function.
    """

    sol = Solution([TaskData(0, [0], 0, 1), TaskData(1, [0], 1, 3)])
    exp_schedule = {0: [0, 1]}
    assert find_schedule_per_resource(sol) == exp_schedule

    # Switch order
    sol = Solution([TaskData(0, [0], 1, 3), TaskData(1, [0], 0, 1)])
    exp_schedule = {0: [1, 0]}
    assert find_schedule_per_resource(sol) == exp_schedule

    # On different resources
    sol = Solution([TaskData(0, [0], 0, 1), TaskData(1, [1], 1, 3)])
    exp_schedule = {0: [0], 1: [1]}
    assert find_schedule_per_resource(sol) == exp_schedule

    # Task using multiple resources
    sol = Solution([TaskData(0, [0], 0, 1), TaskData(1, [0, 1], 1, 3)])
    exp_schedule = {0: [0, 1], 1: [1]}
    assert find_schedule_per_resource(sol) == exp_schedule

    # Task using multiple resources, different order
    sol = Solution([TaskData(0, [0], 1, 3), TaskData(1, [0, 1], 0, 1)])
    exp_schedule = {0: [1, 0], 1: [1]}
    assert find_schedule_per_resource(sol) == exp_schedule

    # Task with 0 duration should be correctly placed
    sol = Solution(
        [
            TaskData(0, [0], 1, 3),
            TaskData(1, [0, 1], 0, 1),
            TaskData(2, [0], 1, 1),
        ]
    )
    exp_schedule = {0: [1, 2, 0], 1: [1]}
    assert find_schedule_per_resource(sol) == exp_schedule

    # Also when using multiple resources
    sol = Solution(
        [
            TaskData(0, [0], 1, 3),
            TaskData(1, [0, 1], 0, 1),
            TaskData(2, [0, 1], 1, 1),
        ]
    )
    exp_schedule = {0: [1, 2, 0], 1: [1, 2]}
    assert find_schedule_per_resource(sol) == exp_schedule
