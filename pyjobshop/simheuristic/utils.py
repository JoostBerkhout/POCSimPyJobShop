import csv
from typing import Dict

from pyjobshop import Solution


def find_schedule_per_resource(solution: Solution) -> Dict[int, list[int]]:
    """
    Returns a dictionary mapping resource indices to schedule of task indices.

    Warning: if tasks start times are equal, the order is not guaranteed.
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


def save_elite_solutions_to_csv(elite_solutions, filename: str):
    """
    Saves the elite solutions to a CSV file.
    """

    # Define the CSV file path
    csv_file_path = filename + ".csv"

    # Open the file in write mode
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        col_headers = ["Solution ID", "Time", "Objective", "Mean objective"]
        writer.writerow(col_headers)

        # Iterate over elite solutions and write the data rows
        for elite_solution in elite_solutions.elite_solutions.values():
            writer.writerow(
                [
                    id(elite_solution.solution),
                    elite_solution.metadata["metadata"]["current_time"],
                    round(elite_solution.objective, 2),
                    round(elite_solution.simulator.mean, 2),
                ]
            )

    print(f"Data saved to {csv_file_path}")
