import json

import pytest
from numpy.testing import assert_, assert_equal

from pyjobshop.Solution import Solution, TaskData


def test_task_eq():
    """
    Tests the equality comparison of tasks.
    """
    task1 = TaskData(0, [0], 1, 2)

    assert_equal(task1, TaskData(0, [0], 1, 2))
    assert_(task1 != TaskData(0, [0], 1, 3))


def test_solution_eq():
    """
    Tests the equality comparison of solutions.
    """
    tasks = [TaskData(0, [0], 0, 0), TaskData(0, [1], 0, 1)]
    sol1 = Solution(tasks)

    assert_equal(sol1, Solution(tasks))
    other = [TaskData(0, [0], 0, 0), TaskData(1, [0], 0, 3)]
    assert_(sol1 != Solution(other))


def test_solution_makespan():
    """
    Tests the makespan calculation of a solution.
    """
    tasks = [TaskData(0, [0], 0, 0), TaskData(0, [1], 0, 100)]
    sol = Solution(tasks)

    assert_equal(sol.makespan, 100)


@pytest.fixture
def sample_solution():
    """
    Fixture to create a sample solution for testing.
    """
    tasks = [
        TaskData(mode=1, resources=[1, 2], start=0, end=10),
        TaskData(mode=2, resources=[2, 3], start=10, end=20),
    ]
    return Solution(tasks)


def test_to_dict(sample_solution):
    """
    Tests the conversion of a solution to a dictionary representation.
    """
    expected = {
        "tasks": [
            {"mode": 1, "resources": [1, 2], "start": 0, "end": 10},
            {"mode": 2, "resources": [2, 3], "start": 10, "end": 20},
        ],
    }
    assert sample_solution.to_dict() == expected


def test_to_json_str(sample_solution):
    """
    Tests the conversion of a solution to a JSON string representation.
    """
    json_str = sample_solution.to_json_str()
    loaded = json.loads(json_str)
    expected = {
        "tasks": [
            {"mode": 1, "resources": [1, 2], "start": 0, "end": 10},
            {"mode": 2, "resources": [2, 3], "start": 10, "end": 20},
        ],
    }
    assert loaded == expected


def test_from_json_str_roundtrip(sample_solution):
    """
    Tests the roundtrip conversion of a solution to and from JSON string repr.
    """
    json_str = sample_solution.to_json_str()
    restored_solution = Solution.from_json_str(json_str)

    assert isinstance(restored_solution, Solution)
    assert restored_solution == sample_solution
