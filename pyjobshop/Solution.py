import json
from dataclasses import dataclass


@dataclass
class TaskData:
    """
    Stores scheduling data related to a task.

    Parameters
    ----------
    mode
        The selected mode.
    resources
        The selected resources.
    start
        The start time.
    end
        The end time.
    """

    mode: int
    resources: list[int]
    start: int
    end: int


class Solution:
    """
    Solution to the problem.

    Parameters
    ----------
    tasks
        The list of scheduled tasks.
    """

    def __init__(self, tasks: list[TaskData]):
        self._tasks = tasks

    @property
    def tasks(self) -> list[TaskData]:
        """
        Returns the list of tasks and its scheduling data.
        """
        return self._tasks

    def __eq__(self, other) -> bool:
        return self.tasks == other.tasks

    @property
    def makespan(self) -> int:
        """
        Returns the makespan of the solution.
        """
        return max(task.end for task in self.tasks)

    def to_dict(self) -> dict:
        """
        Converts the solution to a dictionary representation.
        """
        return {
            "tasks": [
                {
                    "mode": task.mode,
                    "resources": task.resources,
                    "start": task.start,
                    "end": task.end,
                }
                for task in self.tasks
            ],
        }

    def to_json_str(self) -> str:
        """
        Converts the solution to a JSON string representation.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json_str(cls, json_str: str) -> "Solution":
        """
        Creates a Solution instance from a JSON string representation.
        """
        data = json.loads(json_str)
        tasks = [TaskData(**t) for t in data["tasks"]]
        return Solution(tasks)
