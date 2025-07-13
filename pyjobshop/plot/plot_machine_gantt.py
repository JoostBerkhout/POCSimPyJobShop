from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from pyjobshop.ProblemData import ProblemData
from pyjobshop.Solution import Solution

from .utils import get_colors as _get_colors


def plot_machine_gantt(
    solution: Solution,
    data: ProblemData,
    resources: Optional[list[int]] = None,
    plot_labels: bool = False,
    title: str = "Solution",
    ax: Optional[Axes] = None,
):
    """
    Plots a Gantt chart of the solution, where each row represents a machine
    and each bar represents a task processed on that machine.

    Parameters
    ----------
    solution
        A solution to the problem.
    data
        The problem data instance.
    resources
        The resources (by index) to plot and in which order they should appear
        (from top to bottom). Defaults to all resources in the data instance.
    plot_labels
        Whether to plot the task names as labels.
    title
        The title of the plot.
    ax
        Axes object to draw the plot on. One will be created if not provided.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        assert ax is not None  # for linting

    if resources is None:
        resources = list(range(data.num_resources))

    # Tasks belonging to the same job get the same color. Task that do not
    # belong to a job are colored grey.
    task2color = defaultdict(lambda: "grey")
    colors = _get_colors()
    for idx, task in enumerate(data.tasks):
        if task.job is not None:
            task2color[idx] = colors[task.job % len(colors)]

    for idx, task_data in enumerate(solution.tasks):
        kwargs = {
            "color": task2color[idx],
            "linewidth": 1,
            "edgecolor": "black",
            "alpha": 0.75,
        }
        duration = task_data.end - task_data.start
        for resource in task_data.resources:
            if resource not in resources:
                continue  # skip resources not in the order

            ax.barh(
                resources.index(resource),
                duration,
                left=task_data.start,
                **kwargs,
            )

            if plot_labels:
                ax.text(
                    task_data.start + duration / 2,
                    resources.index(resource),
                    data.tasks[idx].name or f"{idx}",
                    ha="center",
                    va="center",
                )

    labels = [
        data.resources[idx].name or f"Machine {idx}" for idx in resources
    ]

    ax.set_yticks(ticks=range(len(labels)), labels=labels)
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xlim(0, ax.get_xlim()[1])  # start time at zero
    ax.set_xlabel("Time")
    ax.set_title(title)


def plot_machine_gantt_for_OS(
    solution: Solution,
    data: ProblemData,
    resources: Optional[list[int]] = None,
    plot_labels: bool = False,
    title: str = "Solution",
    ax: Optional[Axes] = None,
):
    """
    Plots a Gantt chart of the solution, where each row represents a machine
    and each bar represents a task processed on that machine.

    Modified version of plot_machine_gantt for open shop problems (OS).

    Parameters
    ----------
    solution
        A solution to the problem.
    data
        The problem data instance.
    resources
        The resources (by index) to plot and in which order they should appear
        (from top to bottom). Defaults to all resources in the data instance.
    plot_labels
        Whether to plot the task names as labels.
    title
        The title of the plot.
    ax
        Axes object to draw the plot on. One will be created if not provided.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        assert ax is not None  # for linting

    if resources is None:
        resources = list(range(data.num_resources))

    # Tasks belonging to the same job get the same color. Task that do not
    # belong to a job are colored grey.
    task2color = defaultdict(lambda: "grey")
    colors = _get_colors()
    task2job = {}
    job2tasks: dict[Any, list[Any]] = {job: [] for job in range(data.num_jobs)}
    for idx, task in enumerate(data.tasks):
        task2job[idx] = task.job
        job2tasks[task.job].append(idx)
        if task.job is not None:
            task2color[idx] = colors[task.job % len(colors)]

    num_machines = data.num_resources - data.num_jobs  # OS specific!
    for idx, task_data in enumerate(solution.tasks):
        kwargs = {
            "color": task2color[idx],
            "linewidth": 1,
            "edgecolor": "black",
            "alpha": 0.75,
        }
        duration = task_data.end - task_data.start
        for resource in task_data.resources:
            if resource not in resources:
                continue  # skip resources not in the order

            ax.barh(
                resources.index(resource),
                duration,
                left=task_data.start,
                **kwargs,
            )

            if plot_labels:
                ax.text(
                    task_data.start + duration / 2,
                    resources.index(resource),
                    data.tasks[idx].name or f"{idx}",
                    ha="center",
                    va="center",
                )

    # Draw deadlines for OS
    job_end_times = [
        max(solution.tasks[task].end for task in job2tasks[job])
        for job in range(data.num_jobs)
    ]
    for job, end_time in enumerate(job_end_times):
        x_loc = data.jobs[job].due_date
        y_loc_start = num_machines + job - 0.5
        y_loc_end = y_loc_start + 1
        b_late = end_time > data.jobs[job].due_date
        ax.plot(
            [x_loc, x_loc],
            [y_loc_start, y_loc_end],
            # marker='o',
            color="red" if b_late else "green",
            linestyle="-",
            linewidth=2,
        )
        ax.plot(
            [end_time, x_loc],
            [y_loc_start + 0.5, y_loc_start + 0.5],
            # marker='o',
            color="red" if b_late else "green",
            linestyle=":",
            linewidth=1,
        )

    labels = [
        data.resources[idx].name or f"Machine {idx}" for idx in resources
    ]

    ax.set_yticks(ticks=range(len(labels)), labels=labels)
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xlim(0, ax.get_xlim()[1])  # start time at zero
    ax.set_xlabel("Time")
    ax.set_title(title)


def plot_machine_gantt_with_tardiness(
    solution: Solution,
    data: ProblemData,
    resources: Optional[list[int]] = None,
    plot_labels: bool = False,
    title: str = "Solution",
    ax: Optional[Axes] = None,
):
    """
    Plots a Gantt chart of the solution, where each row represents a machine
    and each bar represents a task processed on that machine.

    Modified version of plot_machine_gantt that indicates tardiness.

    Parameters
    ----------
    solution
        A solution to the problem.
    data
        The problem data instance.
    resources
        The resources (by index) to plot and in which order they should appear
        (from top to bottom). Defaults to all resources in the data instance.
    plot_labels
        Whether to plot the task names as labels.
    title
        The title of the plot.
    ax
        Axes object to draw the plot on. One will be created if not provided.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        assert ax is not None  # for linting

    if resources is None:
        resources = list(range(data.num_resources))

    # Tasks belonging to the same job get the same color. Task that do not
    # belong to a job are colored grey.
    fontsize = 16
    task2color = defaultdict(lambda: "grey")
    colors = _get_colors()
    task2job = {}
    job2tasks: dict[Any, list[Any]] = {job: [] for job in range(data.num_jobs)}
    for idx, task in enumerate(data.tasks):
        task2job[idx] = task.job
        job2tasks[task.job].append(idx)
        if task.job is not None:
            task2color[idx] = colors[task.job % len(colors)]

    for idx, task_data in enumerate(solution.tasks):
        kwargs = {
            "color": task2color[idx],
            "linewidth": 1,
            "edgecolor": "black",
            "alpha": 0.75,
        }
        duration = task_data.end - task_data.start
        for resource in task_data.resources:
            if resource not in resources:
                continue  # skip resources not in the order

            ax.barh(
                resources.index(resource),
                duration,
                left=task_data.start,
                **kwargs,
            )

            if plot_labels:
                ax.text(
                    task_data.start + duration / 2,
                    resources.index(resource),
                    data.tasks[idx].name or f"{idx + 1}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )

    # Draw deadlines lines
    num_resources = len(resources)
    last_taks_per_job = [
        job2tasks[job][
            np.argmax([solution.tasks[task].end for task in job2tasks[job]])
        ]
        for job in range(data.num_jobs)
    ]
    job_end_times = [solution.tasks[t].end for t in last_taks_per_job]
    job_last_machines = [
        solution.tasks[t].resources[0] for t in last_taks_per_job
    ]
    for job, end_time in enumerate(job_end_times):
        x_loc_start = end_time
        x_loc_end = data.jobs[job].due_date
        y_loc_start = job_last_machines[job]
        y_loc_end = num_resources - 1 + 0.6
        b_late = end_time > data.jobs[job].due_date
        ax.plot(
            [x_loc_start, x_loc_end],
            [y_loc_start, y_loc_end],
            # marker='o',
            color="red" if b_late else "green",
            linestyle=":",
            linewidth=2,
        )

    labels = [
        data.resources[idx].name or f"Machine {idx + 1}" for idx in resources
    ]

    ticks = [float(x) for x in range(len(labels))]
    ticks += [float(y_loc_end)]
    labels += ["Due date  "]
    ax.set_yticks(ticks=ticks, labels=labels, fontsize=fontsize)
    ax.set_ylim(-0.5, y_loc_end)
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xlim(0, ax.get_xlim()[1])  # start time at zero
    xticks = list(range(0, int(ax.get_xlim()[1]) + 1))
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.set_xlabel("Time", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
