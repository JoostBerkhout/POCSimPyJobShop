import itertools
from typing import Any, Dict, Tuple

import numpy as np

from pyjobshop import Model
from simpyjobshop.DiscreteRV import DiscreteRV, SeededPoisson
from simpyjobshop.problems.Problem import Problem


class ParallelMachines(Problem):
    """
    Specific implementation of a parallel machine scheduling problem.
    """

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        # Unpack data
        num_jobs = data["num_jobs"]
        num_machines = data["num_machines"]
        durations = [
            [data[f"duration_{j}_on_{m}"] for j in range(num_jobs)]
            for m in range(num_machines)
        ]
        due_dates = [data[f"due_date_{j}"] for j in range(num_jobs)]
        setup_times = {
            (m, i, j): data[f"setup_time_{i}_{j}_on_{m}"]
            for i, j in itertools.permutations(range(num_jobs), 2)
            for m in range(num_machines)
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
        machines = [model.add_machine() for _ in range(num_machines)]

        # Add modes to the model
        for m, machine in enumerate(machines):
            for task, duration in zip(tasks, durations[m], strict=True):
                model.add_mode(task, machine, duration=duration)

        # Add sequence-dependent setup times
        for m, machine in enumerate(machines):
            for idx1, task1 in enumerate(tasks):
                for idx2, task2 in enumerate(tasks):
                    if idx1 == idx2:
                        continue
                    model.add_setup_time(
                        machine,
                        task1,
                        task2,
                        duration=setup_times[m, idx1, idx2],
                    )

        # Set objective
        model.set_objective(**objective_weights)

        return model

    def distribution_data(
        self, seed: int = 0
    ) -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
        distributions: Dict[str, DiscreteRV] = {}
        constants: Dict[str, int] = {}

        num_jobs = 40
        num_machines = 4
        loc = 1
        max_rand_mean = 10
        max_setup_time = 10

        # job durations
        np.random.seed(seed)  # for reproducibility
        mean_job_durations: list[list[int]] = []
        for m in range(num_machines):
            mean_job_durations.append([])
            for i in range(num_jobs):
                mean_job_duration = np.random.randint(max_rand_mean)
                mean_job_durations[-1].append(mean_job_duration)
                seed = i + m * num_jobs
                gen = SeededPoisson(lam=mean_job_duration, loc=loc, seed=seed)
                distributions[f"duration_{i}_on_{m}"] = gen

        # setup times
        setup_times = {
            (m, i, j): np.random.randint(1, max_setup_time)
            for i, j in itertools.permutations(range(num_jobs), 2)
            for m in range(num_machines)
        }

        # calculate meaningful due dates
        machine_schedules = np.array_split(np.arange(num_jobs), num_machines)
        finish_times = -np.inf * np.ones(num_jobs)
        for idx, schedule in enumerate(machine_schedules):
            durations = [x + loc for x in mean_job_durations[idx]]
            first = schedule[0]
            finish_times[first] = durations[first]
            for i, j in zip(schedule[:-1], schedule[1:], strict=True):
                start_time = finish_times[i] + setup_times[idx, i, j]
                finish_times[j] = start_time + durations[j]
        due_dates = [int(x) for x in finish_times]

        # store constants
        constants["num_jobs"] = num_jobs
        constants["num_machines"] = num_machines
        constants["max_rand_mean"] = max_rand_mean
        constants["seed"] = seed
        for (m, i, j), setup_time in setup_times.items():
            constants[f"setup_time_{i}_{j}_on_{m}"] = setup_time
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
