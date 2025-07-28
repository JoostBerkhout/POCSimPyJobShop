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


Schedule = dict[int, list[int]]  # resource idx -> task schedule


def find_schedule_per_resource(solution: Solution) -> Schedule:
    """
    Returns a Schedule mapping resource indices to schedule of task indices.

    Warning: In case resources can work on multiple tasks at the same time,
    this function still sets the order based on the midpoints of the tasks
    despite that it induces no ordering per se.
    """

    schedule_per_resource: dict[int, list[int]] = {}

    # Group tasks by resource
    for task_idx, task in enumerate(solution.tasks):
        for resource_idx in task.resources:
            if resource_idx not in schedule_per_resource:
                schedule_per_resource[resource_idx] = []
            schedule_per_resource[resource_idx].append(task_idx)

    # Sort tasks per resource by their midpoints (start + end) / 2
    # (to ensure zero duration tasks are sorted correctly)
    midpoints = [(t.start + t.end) / 2 for t in solution.tasks]
    for resource_idx, task_indices in schedule_per_resource.items():
        schedule_per_resource[resource_idx] = sorted(
            task_indices, key=lambda idx: midpoints[idx]
        )

    return schedule_per_resource
