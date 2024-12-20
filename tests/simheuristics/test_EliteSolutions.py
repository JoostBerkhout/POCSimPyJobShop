from unittest.mock import Mock

import pytest

from pyjobshop import Solution
from pyjobshop.simheuristic.EliteSolutions import EliteSolutions
from pyjobshop.simheuristic.Simulator import Simulator


@pytest.fixture
def mock_solution():
    """Fixture for creating a mock Solution object."""
    return Mock(spec=Solution)


@pytest.fixture
def mock_simulator():
    """Fixture for creating a mock Simulator object."""
    simulator = Mock(spec=Simulator)
    simulator.mean = 50.0
    simulator.variance = 10.0
    return simulator


@pytest.fixture
def elite_solutions():
    """Fixture for initializing EliteSolutions."""
    return EliteSolutions()


def test_add_solution(elite_solutions, mock_solution, mock_simulator):
    """Test adding a solution."""
    schedule = {0: [1, 2], 1: [3]}
    elite_solutions.add(
        mock_solution, mock_simulator, objective=25.0, schedule=schedule
    )

    assert len(elite_solutions.elite_solutions) == 1
    assert id(mock_solution) in elite_solutions.elite_solutions
    elite_solution = elite_solutions.elite_solutions[id(mock_solution)]
    assert elite_solution.schedule == schedule
    assert elite_solution.objective == 25.0
    assert elite_solution.simulator == mock_simulator


def test_keep_top_n(elite_solutions, mock_solution, mock_simulator):
    """Test keeping the top N solutions."""
    # Create multiple solutions with different objectives
    for i in range(5):
        solution = Mock(spec=Solution)
        simulator = Mock(spec=Simulator)
        simulator.mean = 50.0 - i  # Lower mean is better
        elite_solutions.add(solution, simulator, 10, {0: [0, 1]})

    assert len(elite_solutions.elite_solutions) == 5

    # Keep top 3 solutions
    elite_solutions.keep_top_n(3)

    assert len(elite_solutions.elite_solutions) == 3

    # Check that the solutions with the lowest mean remain
    remaining_means = [
        elite_solution.simulator.mean
        for elite_solution in elite_solutions.elite_solutions.values()
    ]
    assert sorted(remaining_means) == [46.0, 47.0, 48.0]  # Lowest means


def test_get_best_solution(elite_solutions, mock_solution, mock_simulator):
    """Test retrieving the best solution."""
    best_solution = Mock(spec=Solution)
    best_simulator = Mock(spec=Simulator)
    best_simulator.mean = 10.0  # Best mean

    # Add non-optimal solutions
    for i in range(3):
        solution = Mock(spec=Solution)
        simulator = Mock(spec=Simulator)
        simulator.mean = 11.0 + i  # Higher mean is worse
        elite_solutions.add(solution, simulator, 10.0, {0: [0, 1]})

    # Add the best solution
    elite_solutions.add(best_solution, best_simulator, 5.0, {0: [0, 1]})

    assert elite_solutions.get_best_elite_solution().solution == best_solution


def test_get_worst_solution(elite_solutions, mock_solution, mock_simulator):
    """Test retrieving the worst solution."""
    for i in range(3):
        solution = Mock(spec=Solution)
        simulator = Mock(spec=Simulator)
        simulator.mean = 11.0 + i  # Higher mean is worse
        elite_solutions.add(solution, simulator, 10.0, {0: [0, 1]})

    worst_solution = elite_solutions.get_worst_elite_solution().solution

    assert worst_solution == solution


def test_is_new_schedule(elite_solutions, mock_solution, mock_simulator):
    """Test checking if a schedule is new."""
    schedule = {0: [1, 2], 1: [3]}
    elite_solutions.add(mock_solution, mock_simulator, 25.0, schedule)

    new_schedule = {0: [2, 3], 1: [4]}
    assert elite_solutions.is_new_schedule(new_schedule) is True

    duplicate_schedule = {0: [1, 2], 1: [3]}
    assert elite_solutions.is_new_schedule(duplicate_schedule) is False


def test_print_summary(capsys, elite_solutions, mock_solution, mock_simulator):
    """Test the print_summary method."""
    schedule = {0: [1, 2], 1: [3]}
    elite_solutions.add(
        mock_solution,
        mock_simulator,
        25.0,
        schedule,
        metadata={"key": "value"},
    )

    elite_solutions.print_summary()
    captured = capsys.readouterr()

    assert "Elite Solutions:" in captured.out
    assert "Objective: 25.00" in captured.out
    assert "'key': 'value'" in captured.out
    assert "Mean: 50.00" in captured.out
    assert "Var: 10.00" in captured.out
