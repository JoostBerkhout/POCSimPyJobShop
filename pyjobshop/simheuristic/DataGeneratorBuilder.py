from typing import Callable, Dict, Tuple

from pyjobshop.simheuristic.DataGenerator import DataGenerator
from pyjobshop.simheuristic.DiscreteRV import DiscreteRV


class DataGeneratorBuilder:
    """
    Builder class for constructing DataGenerator instances.

    Attributes
    ----------
    distributions : Dict[str, DiscreteRV]
        Stores distributions for the parameters being configured.
    constants : Dict[str, Any]
        Stores constants (without uncertainty).
    """

    def __init__(
        self,
        problem_data: Callable[
            [], Tuple[Dict[str, DiscreteRV], Dict[str, int]]
        ],
    ) -> None:
        self.distributions, self.constants = problem_data()

    def build(self) -> DataGenerator:
        """
        Constructs the DataGenerator instance.

        Returns
        -------
        DataGenerator
            A DataGenerator with the configured distributions and seed.
        """

        # make (deep) copy of distributions to avoid shared state
        distributions = {k: v.copy() for k, v in self.distributions.items()}

        return DataGenerator(distributions, self.constants)
