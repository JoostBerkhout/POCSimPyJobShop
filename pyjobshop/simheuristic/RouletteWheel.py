from typing import List, Tuple

import numpy as np
from numpy.random import RandomState
from pyjobshop.simheuristic.SelectionScheme import StrategySelectionScheme


class RouletteWheel(StrategySelectionScheme):
    R"""
    The ``RouletteWheel`` scheme updates operator weights as a convex
    combination of the current weight, and the new score.

    When the algorithm starts, all operators :math:`i` are assigned weight
    :math:`\omega_i = 1`. In each iteration, a destroy and a repair operator
    are selected by the ALNS algorithm, based on the normalised current weights
    :math:`\omega_i`. The selected operators are applied to the current
    solution, resulting in a new candidate solution. This candidate is
    evaluated by the ALNS algorithm, which leads to one of four outcomes:

   A list of three non-negative elements, representing the rewards when
   - the candidate solution results in a new global best SCOP (idx 0)
   - a new best DCOP (idx 1)
   - the solution is reject (idx 2).

    Each of these outcomes is assigned a score :math:`s_j` (with
    :math:`j = 1,...,3`). After observing outcome :math:`j`, the weights of
    the selected strategies

    .. math::

        \begin{align}
            \omega_s &= \theta \omega_s + (1 - \theta) s_j, \\
        \end{align}

    where :math:`0 \le \theta \le 1` (known as the *operator decay rate*)
    is a parameter.

    Parameters
    ----------
    scores
        A list of three three-negative elements, representing the weight
        updates when the candidate solution results in a new global best
        SCOP (idx 0), a new best DCOP (idx 1), the solution is reject (idx 2).
    decay
        Operator decay parameter :math:`\theta \in [0, 1]`. This parameter is
        used to weigh the running performance of each operator.
    num_strategies
        Number of strategies.
    """

    def __init__(
        self,
        scores: List[float],
        decay: float,
        num_strategies: int
    ):
        super().__init__(num_strategies)

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 3:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected three scores, found {len(scores)}")

        if not (0 <= decay <= 1):
            raise ValueError("decay outside [0, 1] not understood.")

        self._scores = scores
        self._weights = np.ones(num_strategies, dtype=float)
        self._decay = decay

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def decay(self) -> float:
        return self._decay

    def __call__(
        self, rnd_state: RandomState
    ) -> Tuple[int, int]:
        """
        Selects a strategy operator to apply in this iteration.
        The probability of an operator being selected is based on the operator
        weights: operators that frequently improve the current solution - and
        thus have higher weights - are selected with a higher probability.

        Parameters
        ----------
        rnd_state
            Random state object, to be used for random number generation.

        Returns
        -------
        A strategy index.
        """

        def select(op_weights):
            probs = op_weights / np.sum(op_weights)
            return rnd_state.choice(range(len(op_weights)), p=probs)

        s_idx = select(self._weights)

        return s_idx

    def update(self, s_idx, outcome):
        self._weights[s_idx] *= self._decay
        self._weights[s_idx] += (1 - self._decay) * self._scores[outcome]


