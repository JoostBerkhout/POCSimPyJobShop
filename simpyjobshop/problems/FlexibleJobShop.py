from typing import Any, Dict, Tuple

import numpy as np

from pyjobshop import Model
from simpyjobshop.DiscreteRV import (
    Constant,
    DiscreteRV,
    SeededPoisson,
)
from simpyjobshop.problems.Problem import Problem


class FlexibleJobShop(Problem):
    """
    Specific implementation of a flexible flow shop. Inspired by the notebook
    flexible_job_shop.ipynb.
    """

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        # Unpack data
        num_jobs = data["num_jobs"]
        num_tasks = data["num_tasks"]
        num_machines = data["num_machines"]
        durations = []
        for job_idx in range(num_jobs):
            job_tasks = []
            for task_idx in range(num_tasks):
                task_durations = [
                    (data[f"dur_{job_idx}_{task_idx}_on_{mach_idx}"], mach_idx)
                    for mach_idx in range(num_machines)
                ]
                job_tasks.append(task_durations)
            durations.append(job_tasks)

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
        machines = [
            model.add_machine(name=f"Machine {idx}")
            for idx in range(num_machines)
        ]

        jobs = {}
        tasks = {}

        for job_idx, job_data in enumerate(durations):
            job = model.add_job(name=f"Job {job_idx}")
            jobs[job_idx] = job

            for idx in range(len(job_data)):
                task_tup = (job_idx, idx)
                tasks[task_tup] = model.add_task(job, name=f"Task {task_tup}")

        for job_idx, job_data in enumerate(durations):
            for idx, task_data in enumerate(job_data):
                task = tasks[(job_idx, idx)]

                for duration, machine_idx in task_data:
                    machine = machines[machine_idx]
                    model.add_mode(task, machine, duration)

            for idx in range(len(job_data) - 1):
                first = tasks[(job_idx, idx)]
                second = tasks[(job_idx, idx + 1)]
                model.add_end_before_start(first, second)

        # Set objective
        model.set_objective(**objective_weights)

        return model

    def distribution_data(
        self, seed: int = 0
    ) -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
        distributions: Dict[str, DiscreteRV] = {}
        constants: Dict[str, int] = {}

        num_jobs = 40
        num_tasks = 5
        num_machines = 5
        loc = 1
        max_rand_mean = 15
        prob_random_duration = 0.3

        # Set job distributions
        np.random.seed(seed)  # for reproducibility
        gen: DiscreteRV
        for job in range(num_jobs):
            for task in range(num_tasks):
                for machine in range(num_machines):
                    mean_job_duration = np.random.randint(max_rand_mean)
                    if np.random.rand() < prob_random_duration:
                        gen = SeededPoisson(
                            lam=mean_job_duration,
                            loc=loc,
                            seed=task + job * num_tasks,
                        )
                    else:
                        gen = Constant(loc + mean_job_duration)
                    distributions[f"dur_{job}_{task}_on_{machine}"] = gen

        # store constants
        constants["num_jobs"] = num_jobs
        constants["num_tasks"] = num_tasks
        constants["num_machines"] = num_machines
        constants["max_rand_mean"] = max_rand_mean
        constants["seed"] = seed
        constants["weight_makespan"] = 1
        constants["weight_tardy_jobs"] = 0
        constants["weight_total_flow_time"] = 0
        constants["weight_total_tardiness"] = 0
        constants["weight_total_earliness"] = 0
        constants["weight_max_tardiness"] = 0
        constants["weight_max_lateness"] = 0

        return distributions, constants
