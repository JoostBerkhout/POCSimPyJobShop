from abc import ABC, abstractmethod
from typing import Tuple

from numpy.random import RandomState

from pyjobshop.simheuristic.Outcome import Outcome


class StrategySelectionScheme(ABC):
    """
    Base class describing a strategy selection scheme.

    Parameters
    ----------
    num_strategies
        Number of strategeies
    """

    def __init__(
        self,
        num_strategies: int,
    ):

        self._num_strategies = num_strategies

    @property
    def num_strategies(self) -> int:
        return self._num_strategies

    @abstractmethod
    def __call__(
        self, rnd: RandomState
    ) -> Tuple[int, int]:
        """
        Determine which destroy and repair operator pair to apply in this
        iteration.

        Parameters
        ----------
        rnd_state
            Random state object, to be used for random number generation.
        best
            The best solution state observed so far.
        current
            The current solution state.

        Returns
        -------
        A tuple of (d_idx, r_idx), which are indices into the destroy and
        repair operator lists, respectively.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self, s_idx: int, outcome: Outcome
    ):
        """
        Updates the selection schame based on the outcome of the applied
        destroy (d_idx) and repair (r_idx) operators.

        Parameters
        ----------
        candidate
            The candidate solution state.
        s_idx
            Strategy index.
        outcome
            Score enum value used for the various iteration outcomes.
        """
        return NotImplemented

    @staticmethod
    def _validate_arguments(
        num_strategies: int
    ):
        if num_strategies <= 0:
            raise ValueError("Missing strategies.")
