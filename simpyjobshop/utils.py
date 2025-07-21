import csv

import wandb
from matplotlib import pyplot as plt

from pyjobshop.plot import plot_machine_gantt
from pyjobshop.Solution import Solution
from simpyjobshop.problems.Problem import Problem

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
        for elite_solution in elite_solutions:
            writer.writerow(
                [
                    id(elite_solution.solution),
                    elite_solution.metadata["metadata"]["current_time"],
                    round(elite_solution.objective, 2),
                    round(elite_solution.simulator.mean, 2),
                ]
            )

    print(f"Data saved to {csv_file_path}")


def init_wandb(
    problem_name: str, config: dict, wandb_config: dict[str, str] | None = None
) -> None:
    """
    Initializes a Weights & Biases run with a structured naming scheme.

    Parameters
    ----------
    problem_name : str
        The name of the problem class.
    config : dict
        Configuration to be logged (usually simheuristic config).
    wandb_config : dict[str, str], optional
        Configuration for Weights & Biases logging, by default None.
    """

    if wandb_config is None:
        wandb_config = {}

    project_name = wandb_config.get("project_name")
    run_name = wandb_config.get("run_name")
    job_name = wandb_config.get("job_name")

    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        group=problem_name,
        reinit=True,
        job_type=job_name,
    )


def plot_gantt_chart(solution: Solution, problem: Problem, title: str = ""):
    """
    Plots a Gantt chart of the solution.
    """
    data_generator = problem.build_data_generator()
    data = data_generator.int_mean()
    model = problem.concrete_model(data)  # data used is irrelevant
    plot_machine_gantt(solution, model.data(), title=title)
    plt.show()
