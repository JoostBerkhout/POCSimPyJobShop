from typing import Dict

from pyjobshop import Solution


def find_schedule_per_resource(solution: Solution) -> Dict[int, list[int]]:
    """
    Returns a dictionary mapping resource indices to schedule of task indices.
    """

    schedule_per_resource: dict[int, list[int]] = {}

    # Group tasks by resource
    for task_idx, task in enumerate(solution.tasks):
        for resource_idx in task.resources:
            if resource_idx not in schedule_per_resource:
                schedule_per_resource[resource_idx] = []
            schedule_per_resource[resource_idx].append(task_idx)

    # Sort tasks per resource by start time
    for resource_idx, task_indices in schedule_per_resource.items():
        schedule_per_resource[resource_idx] = sorted(
            task_indices, key=lambda idx: solution.tasks[idx].start
        )

    return schedule_per_resource


def print_solution_schedule(solution: Solution):
    """
    Prints the schedule of the given solution.
    """
    schedule_per_resource = find_schedule_per_resource(solution)
    for resource_idx, task_indices in schedule_per_resource.items():
        print(f"Resource {resource_idx}: {task_indices}")
