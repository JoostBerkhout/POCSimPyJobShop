from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SimheuristicSpec:
    name: str
    fun: Callable
    config: Any
