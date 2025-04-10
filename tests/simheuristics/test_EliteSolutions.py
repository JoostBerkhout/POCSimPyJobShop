import time
from unittest.mock import ANY, Mock

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
    simulator.num_sims = 3.0
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

    assert elite_solutions.get_best_mean_solution().solution == best_solution


def test_get_worst_solution(elite_solutions, mock_solution, mock_simulator):
    """Test retrieving the worst solution."""
    for i in range(3):
        solution = Mock(spec=Solution)
        simulator = Mock(spec=Simulator)
        simulator.mean = 11.0 + i  # Higher mean is worse
        elite_solutions.add(solution, simulator, 10.0, {0: [0, 1]})

    worst_solution = elite_solutions.get_worst_mean_solution().solution

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


def test_simulate_to_num_sims(elite_solutions, mock_solution, mock_simulator):
    """Test the simulate_to_num_sims method."""

    # Set up the elite solutions with a mock simulator
    schedule = {0: [1, 2], 1: [3]}
    elite_solutions.add(mock_solution, mock_simulator, 25.0, schedule)

    # Run the simulation with no time limit
    num_sims = 5
    elite_solutions.simulate_to_num_sims(num_sims)

    # Verify the simulator's simulate method is called with correct arguments
    assert mock_simulator.simulate.call_count == 1
    extra_sims = num_sims - mock_simulator.num_sims
    assert mock_simulator.simulate.call_args[0][0] == extra_sims
    assert mock_simulator.simulate.call_args[0][1] is None  # No time limit

    # Test that simulation stops if time limit is reached
    time_limit = 0.0  # no time for simulation
    elite_solutions.simulate_to_num_sims(num_sims, time_limit)
    assert mock_simulator.simulate.call_count == 1

    # Test that nothing is simulated for small num_sims
    num_sims = 1
    elite_solutions.simulate_to_num_sims(num_sims)
    assert mock_simulator.simulate.call_count == 1

    # Let's add another solution
    elite_solutions.add(mock_solution, mock_simulator, 25.0, schedule)
    second_solution = Mock(spec=Solution)
    second_simulator = Mock(spec=Simulator)
    second_simulator.num_sims = 10.0
    second_simulator.mean = 40.0
    second_simulator.variance = 5.0
    schedule2 = {0: [4, 5], 1: [6]}
    elite_solutions.add(
        second_solution, second_simulator, objective=30.0, schedule=schedule2
    )

    # Simulate till 10 so that only the first solution is simulated
    num_sims = 10
    elite_solutions.simulate_to_num_sims(num_sims)
    assert mock_simulator.simulate.call_count == 2
    assert second_simulator.simulate.call_count == 0

    # Simulate till 15 so that both are simulated
    num_sims = 15
    elite_solutions.simulate_to_num_sims(num_sims)
    assert mock_simulator.simulate.call_count == 3
    assert second_simulator.simulate.call_count == 1

    # Simulate till 2 so that both are not simulated
    num_sims = 2
    elite_solutions.simulate_to_num_sims(num_sims)
    assert mock_simulator.simulate.call_count == 3
    assert second_simulator.simulate.call_count == 1


def test_sim_to_time_limit(elite_solutions):
    """Test the simulate_to_time_limit method."""

    # Create two solutions with different initial simulation counts
    solution1 = Mock(spec=Solution)
    simulator1 = Mock(spec=Simulator)
    simulator1.num_sims = 10
    simulator1.mean = 45.0
    simulator1.variance = 5.0

    solution2 = Mock(spec=Solution)
    simulator2 = Mock(spec=Simulator)
    simulator2.num_sims = 7
    simulator2.mean = 40.0
    simulator2.variance = 4.0

    # Use simulate mock that sleeps to simulate time passing
    def simulate_mock(num, time_limit=None):
        time.sleep(num * 0.05)
        return None

    simulator1.simulate.side_effect = simulate_mock
    simulator2.simulate.side_effect = simulate_mock

    elite_solutions.add(solution1, simulator1, 25.0, {0: [1, 2]})
    elite_solutions.add(solution2, simulator2, 30.0, {0: [3, 4]})

    # Test 1: Only time to simulate the second solution 3 times
    start_time = time.time()
    elite_solutions.simulate_to_time_limit(0.15)
    duration = time.time() - start_time

    assert simulator1.simulate.call_count == 0
    simulator2.simulate.assert_called_once_with(3, ANY)

    # Test 2: Check time limit
    assert 0.15 <= duration <= 0.2

    # Test 3: Check whether both solutions get simulated
    simulator2.num_sims = 10
    start_time = time.time()
    elite_solutions.simulate_to_time_limit(0.1)
    duration = time.time() - start_time

    simulator1.simulate.assert_called_once_with(1, ANY)
    second_call_args = simulator2.simulate.call_args_list[1][0]
    assert second_call_args == (1, ANY)

    # Test 4: Check time limit
    assert 0.1 <= duration <= 0.15
