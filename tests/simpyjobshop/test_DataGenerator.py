from unittest.mock import MagicMock

import pytest

from simpyjobshop.DataGenerator import DataGenerator
from simpyjobshop.DiscreteRV import DiscreteRV


@pytest.fixture
def mock_distributions():
    # Mock DiscreteRV objects
    rv1 = MagicMock(spec=DiscreteRV)
    rv1.rvs.return_value = [10]
    rv1.mean.return_value = 12.5
    rv1.ppf.return_value = 11

    rv2 = MagicMock(spec=DiscreteRV)
    rv2.rvs.return_value = [20]
    rv2.mean.return_value = 22.5
    rv2.ppf.return_value = 21

    return {"param1": rv1, "param2": rv2}


@pytest.fixture
def constants():
    return {"constant1": 5, "constant2": 15}


@pytest.fixture
def data_generator(mock_distributions, constants):
    return DataGenerator(mock_distributions, constants)


def test_random(data_generator):
    result = data_generator.random()
    expected = {"param1": 10, "param2": 20, "constant1": 5, "constant2": 15}
    assert result == expected, f"Expected {expected}, but got {result}"


def test_mean(data_generator):
    result = data_generator.mean()
    expected = {
        "param1": 12.5,
        "param2": 22.5,
        "constant1": 5,
        "constant2": 15,
    }
    assert result == expected, f"Expected {expected}, but got {result}"


def test_int_mean(data_generator):
    result = data_generator.int_mean()
    expected = {"param1": 12, "param2": 22, "constant1": 5, "constant2": 15}
    assert result == expected, f"Expected {expected}, but got {result}"


def test_quantile(data_generator):
    result = data_generator.quantile(0.5)  # Test with p=0.5
    expected = {"param1": 11, "param2": 21, "constant1": 5, "constant2": 15}
    assert result == expected, f"Expected {expected}, but got {result}"


def test_by_key_mean(data_generator):
    result = data_generator.by_key("mean")
    expected = {"param1": 12, "param2": 22, "constant1": 5, "constant2": 15}
    assert result == expected, f"Expected {expected}, but got {result}"


def test_by_key_quantile(data_generator):
    result = data_generator.by_key(0.5)  # Test with p=0.5
    expected = {"param1": 11, "param2": 21, "constant1": 5, "constant2": 15}
    assert result == expected, f"Expected {expected}, but got {result}"


def test_by_key_value_error(data_generator):
    with pytest.raises(
        ValueError,
        match="Invalid key 'invalid_key'. Use a "
        "float between 0 and 1 for quantiles"
        " or 'mean' for integer means.",
    ):
        data_generator.by_key("invalid_key")

    with pytest.raises(
        ValueError,
        match="Invalid key '1.2'. Use a float "
        "between 0 and 1 for quantiles or "
        "'mean' for integer means.",
    ):
        data_generator.by_key(1.2)

    with pytest.raises(
        ValueError,
        match="Invalid key '-0.5'. Use a float "
        "between 0 and 1 for quantiles or "
        "'mean' for integer means.",
    ):
        data_generator.by_key(-0.5)
