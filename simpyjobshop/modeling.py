from typing import Dict

from pyjobshop import Model, Solution
from simpyjobshop.problems.Problem import Problem
from simpyjobshop.utils import find_schedule_per_resource


def fix_solution(solution: Solution, model: Model) -> Model:
    """
    Fixes the order and mode-assignments of tasks in the given model based on
    the given solution.

    Warning: This function modifies the given model in-place. Furthermore, it
    changes the model's internal state by adding new modes and removing the
    original modes. See the "hacky" comments in the code.

    Parameters
    ----------
    solution : Solution
        The solution to apply to the model.
    model : Model
        The model to fix the solution in.

    Returns
    -------
    Model
        The model with fixed order and mode-assignments.
    """

    # Fix mode-assignments by adding the one mode per task
    num_modes = len(model.modes)
    map_to_old_mode = {}
    for task_idx, task in enumerate(solution.tasks):
        mode = model.modes[task.mode]
        map_to_old_mode[task_idx] = task.mode
        resources = [model.resources[idx] for idx in mode.resources]
        model.add_mode(
            model.tasks[task_idx],
            resources,
            duration=mode.duration,
            demands=mode.demands,
        )
    model._map_to_old_mode = map_to_old_mode  # hacky: store mapping for test
    model._modes = model._modes[num_modes:]  # hacky: remove original modes

    # Fix order
    schedule_per_resource = find_schedule_per_resource(solution)
    for _schedule in schedule_per_resource.values():
        for idx1, idx2 in zip(_schedule[:-1], _schedule[1:], strict=True):
            task1 = model.tasks[idx1]
            task2 = model.tasks[idx2]
            model.add_end_before_start(task1, task2)

    return model


def find_solution_for_other_data(
    solution: Solution,
    problem: Problem,
    data: Dict[str, int],
    num_workers: int | None = None,
) -> tuple[Solution, float]:
    """
    Transforms the given solution to a new solution that suits that data given
    for the problem at hand. In particular, the new start and end times of
    all the tasks are adjusted to the new data.

    By default, num_workers = 1 since that turned out to be faster than more
    workers in a preliminary experiment, for more details:
    # d-krupke.github.io/cpsat-primer/05_parameters.html#parallelization
    """

    model = problem.concrete_model(data)
    model = fix_solution(solution, model)
    num_workers = 1 if num_workers is None else num_workers
    results = model.solve(display=False, num_workers=num_workers)
    new_solution = results.best
    new_obj_val = results.objective

    # Reset mode choices to original mode indices
    for task in new_solution.tasks:
        task.mode = model._map_to_old_mode[task.mode]

    return new_solution, new_obj_val
