from pyjobshop.simheuristic.problems.SingleMachineProblem import (
    SingleMachineProblem,
)


def test_build_data_generator():
    """
    Tests that Problem.build_data_generator() returns an independent
    DataGenerator object that starts from the same random state.
    """

    problem = SingleMachineProblem()
    data_generator_1 = problem.build_data_generator()
    data_generator_2 = problem.build_data_generator()

    # Check that the two DataGenerator objects start from the same random state
    assert data_generator_1.random() == data_generator_2.random()
    assert data_generator_1.random() == data_generator_2.random()

    # Check that the two DataGenerator objects are independent
    data_generator_1.random()
    assert data_generator_1.random() != data_generator_2.random()
    data_generator_2.random()
    assert data_generator_1.random() == data_generator_2.random()

    # Check that the two DataGenerator objects generate the same data
    assert data_generator_1.int_mean() == data_generator_2.int_mean()
    data_generator_1.random()  # Advance the random state
    assert data_generator_1.int_mean() == data_generator_2.int_mean()
    assert data_generator_1.mean() == data_generator_2.mean()
    for q in [0.1, 0.5, 0.9]:
        assert data_generator_1.quantile(q) == data_generator_2.quantile(q)
