from enum import IntEnum


class Outcome(IntEnum):
    """
    Enum of evaluation outcomes. A candidate solution can be a new global best SCOP, a new best DCOP, or no improvement.
    """

    BEST = 0     #: Following this strategy led to a new best SCOP
    BETTER = 1   #: Following this strategy led to a new best DCOP, but the solution was rejected as SCOP.
    REJECT = 2   #: No new best DCOP found was
    #ACCEPT = 3  #: The solution was added to the elite set, but no global best SCOP