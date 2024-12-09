from typing import Any, Dict, Tuple

from pyjobshop import Model
from pyjobshop.simheuristic.DataGenerator import DataGenerator
from pyjobshop.simheuristic.DiscreteRV import DiscreteRV


class Problem:
    """
    Base class for problem models with distribution data and model creation.

    Attributes
    ----------
    distributions : Dict[str, DiscreteRV]
        Stores distributions for the parameters being configured.
    constants : Dict[str, Any]
        Stores constants (without uncertainty).
    """

    def __init__(self) -> None:
        self.distributions, self.constants = self.distribution_data()

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        """
        Abstract placeholder for creating and returning a specific problem
        model. Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def distribution_data(
        self,
    ) -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
        """
        Abstract placeholder for generating and returning distribution data.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def build_data_generator(self) -> DataGenerator:
        """
        Returns an independent DataGenerator based on the given distribution
        data that starts from a common random state.
        """

        # make (deep)copy of distributions to avoid shared state
        distributions = {k: v.copy() for k, v in self.distributions.items()}

        return DataGenerator(distributions, self.constants)
