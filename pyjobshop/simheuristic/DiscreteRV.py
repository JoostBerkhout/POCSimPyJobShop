import numpy as np
from scipy.stats import poisson, rv_discrete


class DiscreteRV:
    """
    Base class for discrete random variables with seeded RNG support.

    Attributes
    ----------
    distribution : rv_discrete
        A discrete random variable distribution from scipy.stats.
    rng : np.random.Generator
        A dedicated random number generator for reproducibility.
    """

    def __init__(self, distribution: rv_discrete, seed: int = 0) -> None:
        """
        Initializes the discrete random variable.

        Parameters
        ----------
        distribution : rv_discrete
            The discrete distribution (e.g., poisson, binomial).
        seed : int, optional
            The seed for the random number generator. Defaults to 0.
        """
        self.distribution = distribution
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def copy(self):
        """
        Creates a (deep)copy of the object.

        Returns
        -------
        DiscreteRV
            A copy of the random variable.
        """
        return DiscreteRV(self.distribution, self.seed)

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Generates random samples from the distribution.

        Parameters
        ----------
        size : int, optional
            Number of random samples to generate (default is 1).

        Returns
        -------
        np.ndarray
            Array of random samples.
        """
        return self.distribution.rvs(size=size, random_state=self.rng)

    def mean(self) -> float:
        """
        Returns the mean of the distribution.

        Returns
        -------
        float
            The mean of the distribution.
        """
        return self.distribution.mean()

    def var(self) -> float:
        """
        Returns the variance of the distribution.

        Returns
        -------
        float
            The variance of the distribution.
        """
        return self.distribution.var()

    def ppf(self, p: float) -> int:
        """
        Returns the p-quantile of the distribution (percent-point function).

        Parameters
        ----------
        p : float
            Probability for which the quantile is computed (0 <= p <= 1).

        Returns
        -------
        int
            The smallest integer k such that P(X <= k) >= p.
        """
        return int(self.distribution.ppf(p))


class SeededPoisson(DiscreteRV):
    """
    A seeded Poisson random variable.

    Attributes
    ----------
    lam : float
        The lambda (rate parameter) of the Poisson distribution.
    """

    def __init__(self, lam: float, seed: int = 0) -> None:
        """
        Initializes the Poisson random variable.

        Parameters
        ----------
        lam : float
            The lambda (rate parameter) of the Poisson distribution.
        seed : int, optional
            The seed for the random number generator. Defaults to 0.
        """
        super().__init__(poisson(mu=lam), seed)


class Constant(DiscreteRV):
    """
    A constant random variable that always returns the same value.

    Attributes
    ----------
    value : int
        The constant value returned by the random variable.
    """

    def __init__(self, value: int, seed: int = 0) -> None:
        """
        Initializes the Constant random variable.

        Parameters
        ----------
        value : int
            The constant value to be returned by the random variable.
        seed : int, optional
            The seed for the random number generator. Not used in this case.
        """
        # Create a custom discrete distribution with the constant value
        xk = [value]  # The value the generator will always return
        pk = [1]  # Probability of returning the constant value (100%)

        # Call the parent class constructor with this custom distribution
        super().__init__(rv_discrete(name="constant", values=(xk, pk)), seed)
