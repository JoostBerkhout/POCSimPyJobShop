"""Contains the running example for EURO 2025 presentation. It is a simple
2-parallel machines scheduling problem with three jobs."""

import itertools

import numpy as np
from matplotlib import pyplot as plt

from experiments.utils.SimheuristicSpec import SimheuristicSpec
from pyjobshop import Solution, TaskData
from pyjobshop.plot.plot_machine_gantt import plot_machine_gantt_with_tardiness
from simpyjobshop.EliteSet import EliteSet
from simpyjobshop.modeling import find_result_for_expected_data
from simpyjobshop.problems import ParallelMachinesEURO2025
from simpyjobshop.simheuristics import (
    StandardSimheuristicConfig,
    standard_simheuristic,
)
from simpyjobshop.Simulator import Simulator
from simpyjobshop.utils import find_schedule_per_resource

problem_seed = 15
problem = ParallelMachinesEURO2025(seed=problem_seed)

# Find all possible schedules (where 3 indicates the machine divider)
all_schedules = [list(x) for x in list(itertools.permutations([0, 1, 2, 3]))]


def schedule_to_solution(schedule: list[int]) -> Solution:
    """
    Converts a schedule to a Solution object. Hacky PM version.
    """
    machine_idx = schedule.index(3)
    schedule_per_machine = [
        schedule[:machine_idx],
        schedule[machine_idx + 1 :],
    ]
    machine_task_to_mode = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 3,
        (1, 1): 4,
        (1, 2): 5,
    }
    tasks = {}
    for mach_idx, mach_schedule in enumerate(schedule_per_machine):
        start_time = 0
        for task_idx in mach_schedule:
            # Create a TaskData object for each job with its assigned machine
            task = TaskData(
                mode=machine_task_to_mode[mach_idx, task_idx],
                resources=[mach_idx],
                start=start_time,  # dummy start time just to get order right
                end=start_time + 1,  # dummy end time just to get order right
            )
            start_time += 1
            tasks[task_idx] = task

    tasks2 = [tasks[i] for i in range(len(tasks))]

    return Solution(tasks2)


# Convert all schedules to solutions
all_sols_dummy_times = [schedule_to_solution(s) for s in all_schedules]
all_schedule_again = [
    find_schedule_per_resource(sol).get(0, [])
    + [3]
    + find_schedule_per_resource(sol).get(1, [])
    for sol in all_sols_dummy_times
]
assert all_schedule_again == all_schedules, "Schedules do not match!"

# Create an elite set of all solutions
elite_solutions = EliteSet()
num_sims = 5000
all_solutions = []
count = 0
for sol in all_sols_dummy_times:
    exp_data_result = find_result_for_expected_data(sol, problem)
    all_solutions.append(exp_data_result.best)  # Now with correct times
    simulator = Simulator(problem, sol)
    simulator.simulate(num_sims)
    elite_solutions.add(
        solution=exp_data_result.best,
        simulator=simulator,
        objective=exp_data_result.objective,
        schedule=find_schedule_per_resource(sol),
    )
    count += 1
    print(f"{count}/{len(all_sols_dummy_times)}")

# Plot objective value vs. mean objective value
plt.figure()
mean_obj_vals = [es.simulator.mean for es in elite_solutions]
obj_vals = [es.objective for es in elite_solutions]
plt.plot(obj_vals, mean_obj_vals, "o", markersize=18)
plt.xlabel("DSP objective value", fontsize=18)
plt.ylabel("SSP objective value", fontsize=18)
plt.title("Objective values of all solutions for DSP and SSP", fontsize=18)
plt.grid()
plt.show()

# Plot best solutions
best_DCOP_solution = elite_solutions[np.argmin(obj_vals)]
best_SCOP_solution = elite_solutions[np.argmin(mean_obj_vals)]
data_generator = problem.build_data_generator()
data = data_generator.int_mean()
model = problem.concrete_model(data)
for es in [best_DCOP_solution, best_SCOP_solution]:
    plot_machine_gantt_with_tardiness(
        es.solution,
        model.data(),
        plot_labels=True,
        title=f"Schedule with DSP objective value {es.objective} "
        f"and SSP objective value {round(es.simulator.mean, 2)}",
    )
    plt.show()

# Apply simheuristic
exp_config = {
    "time_limit": 20,
    "num_workers": 2,  # (per instance)
    "use_wandb": False,
    # only relevant for cli_run_experiments.py and submit_slurm_job.py:
    "num_rand_experiments": None,
    "num_parallel_instances": None,
    "num_sims_for_true_expec_objective": None,
}
std_simh_config: StandardSimheuristicConfig = {
    "det_repr": "mean",
    "num_sims": 100,
    "max_size_elite_set": 10**10,
    "frac_budget_before_sims": 0,
    "frac_budget_final_elites_sim": 0,
}
simheuristics: list[SimheuristicSpec] = [
    SimheuristicSpec("std_simh", standard_simheuristic, std_simh_config),
]
simheuristic = simheuristics[0]

callback, last_CP_results, durations = simheuristic.fun(
    problem,
    simheuristic.config,
    exp_config,
)
callback.solutions.print_summary()
