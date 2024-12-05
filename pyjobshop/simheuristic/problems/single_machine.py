import itertools
from typing import Dict, Tuple

import numpy as np

from pyjobshop import Model
from pyjobshop.simheuristic.DiscreteRV import DiscreteRV, SeededPoisson


def single_machine_model(data: Dict[str, int]) -> Model:
    # Unpack data
    num_jobs = data["num_jobs"]
    durations = [data[f"duration_{j}"] for j in range(num_jobs)]
    due_dates = [data[f"due_date_{j}"] for j in range(num_jobs)]
    setup_times = {
        (i, j): data[f"setup_time_{i}_{j}"]
        for i, j in itertools.permutations(range(num_jobs), 2)
    }
    objective_weights = {
        "weight_makespan": data["weight_makespan"],
        "weight_tardy_jobs": data["weight_tardy_jobs"],
        "weight_total_flow_time": data["weight_total_flow_time"],
        "weight_total_tardiness": data["weight_total_tardiness"],
        "weight_total_earliness": data["weight_total_earliness"],
        "weight_max_tardiness": data["weight_max_tardiness"],
        "weight_max_lateness": data["weight_max_lateness"],
    }

    # Create model
    model = Model()
    jobs = [model.add_job(due_date=due_dates[j]) for j in range(num_jobs)]
    tasks = [model.add_task(job) for job in jobs]  # one task per job
    machine = model.add_machine()

    # Add modes to the model
    for task, duration in zip(tasks, durations):
        model.add_mode(task, machine, duration=duration)

    # Add sequence-dependent setup times
    for idx1, task1 in enumerate(tasks):
        for idx2, task2 in enumerate(tasks):
            if idx1 == idx2:
                continue
            model.add_setup_time(
                machine, task1, task2, duration=setup_times[idx1, idx2]
            )

    # Set objective
    model.set_objective(**objective_weights)

    return model


def single_machine_data() -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
    distributions: Dict[str, DiscreteRV] = {}
    constants: Dict[str, int] = {}

    num_jobs = 10
    max_rand_mean = 10
    seed = 0

    # job durations
    np.random.seed(seed)  # for reproducibility
    mean_job_durations = []
    for i in range(num_jobs):
        mean_job_duration = np.random.randint(max_rand_mean)
        mean_job_durations.append(mean_job_duration)
        gen = SeededPoisson(lam=mean_job_duration, seed=i)
        distributions[f"duration_{i}"] = gen

    # setup times
    setup_times = {
        (i, j): np.random.randint(1, num_jobs)
        for i, j in itertools.permutations(range(num_jobs), 2)
    }

    # calculate meaningful due dates
    durations = mean_job_durations
    due_dates = [0]  # dummy, will be removed later
    for j in range(num_jobs - 1):
        due_date = due_dates[-1] + durations[j] + setup_times[j, j + 1]
        due_dates.append(due_date)
    due_dates.append(due_dates[-1] + durations[-1])
    due_dates.pop(0)
    assert len(due_dates) == num_jobs

    # store constants
    constants["num_jobs"] = num_jobs
    constants["max_rand_mean"] = max_rand_mean
    constants["seed"] = seed
    for (i, j), setup_time in setup_times.items():
        constants[f"setup_time_{i}_{j}"] = setup_time
    for i, due_date in enumerate(due_dates):
        constants[f"due_date_{i}"] = due_date
    constants["weight_makespan"] = 1
    constants["weight_tardy_jobs"] = 0
    constants["weight_total_flow_time"] = 0
    constants["weight_total_tardiness"] = 100
    constants["weight_total_earliness"] = 0
    constants["weight_max_tardiness"] = 0
    constants["weight_max_lateness"] = 0

    return distributions, constants
