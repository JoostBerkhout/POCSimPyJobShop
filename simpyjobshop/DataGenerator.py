from typing import Dict

from simpyjobshop.DiscreteRV import DiscreteRV


class DataGenerator:
    """
    Generates data based on data distributions and constant data. It can return
    random, mean, or quantile data regarding the data with distributions.

    Attributes
    ----------
    distributions : Dict[str, DiscreteRV]
        A dictionary mapping parameter names to their distributions.
    constants : Dict[str, int]
        A dictionary mapping parameter names to their constant values.
    """

    def __init__(
        self, distributions: Dict[str, DiscreteRV], constants: Dict[str, int]
    ) -> None:
        self.distributions = distributions
        self.constants = constants

    def random(self) -> Dict[str, int]:
        """Generates random data based on the defined distributions."""
        rand_data = {k: int(v.rvs()[0]) for k, v in self.distributions.items()}
        return rand_data | self.constants

    def mean(self) -> Dict[str, float]:
        """Returns the mean of each parameter's distribution."""
        mean_data = {k: v.mean() for k, v in self.distributions.items()}
        return mean_data | self.constants

    def int_mean(self) -> Dict[str, int]:
        """Returns the int(mean) of each parameter's distribution."""
        int_means = {k: int(v.mean()) for k, v in self.distributions.items()}
        return int_means | self.constants

    def quantile(self, p: float) -> Dict[str, int]:
        """Returns the p-quantile of each parameter's distribution."""
        quant_data = {k: v.ppf(p) for k, v in self.distributions.items()}
        return quant_data | self.constants
