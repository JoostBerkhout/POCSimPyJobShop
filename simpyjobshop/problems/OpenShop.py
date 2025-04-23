from typing import Any, Dict, Tuple

import numpy as np

from pyjobshop import Model
from simpyjobshop.DiscreteRV import (
    Constant,
    DiscreteRV,
    SeededPoisson,
)
from simpyjobshop.problems.MachineSchedule import MachineSchedule
from simpyjobshop.problems.Problem import Problem


class OpenShop(Problem):
    """
    Specific implementation of an open shop scheduling problem.
    It takes FlexibleJobShop as a basis.
    """

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        # Unpack data
        num_jobs = data["num_jobs"]
        num_machines = data["num_machines"]
        due_dates = [data[f"due_date_{i}"] for i in range(num_jobs)]
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
        dummy_machines = [
            model.add_machine(name=f"Dummy machine for job {idx}")
            for idx in range(num_jobs)
        ]  # to prevent working on same job on different machines

        for job_idx in range(num_jobs):
            job = model.add_job(
                name=f"Job {job_idx}", due_date=due_dates[job_idx]
            )
            for machine_idx in range(num_machines):
                # For each machine create a task that also need dummy machine
                task_tuple = (job_idx, machine_idx)
                task = model.add_task(job, name=f"Task {task_tuple}")
                resources = [machines[machine_idx], dummy_machines[job_idx]]
                duration = data[f"dur_{job_idx}_on_{machine_idx}"]
                model.add_mode(task, resources, duration)

        # Set objective
        model.set_objective(**objective_weights)

        return model

    def distribution_data(
        self, seed: int = 0
    ) -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
        distributions: Dict[str, DiscreteRV] = {}
        constants: Dict[str, int] = {}

        num_jobs = 120
        num_machines = 5
        loc = 1
        max_rand_mean = 15
        prob_random_duration = 0.3

        # Set job distributions
        np.random.seed(seed)  # for reproducibility
        gen: DiscreteRV
        for job in range(num_jobs):
            for machine in range(num_machines):
                mean_job_duration = np.random.randint(max_rand_mean)
                if np.random.rand() < prob_random_duration:
                    gen = SeededPoisson(
                        lam=mean_job_duration,
                        loc=loc,
                        seed=machine + job * num_machines,
                    )
                else:
                    gen = Constant(loc + mean_job_duration)
                distributions[f"dur_{job}_on_{machine}"] = gen

        # Create a schedule to determine due dates
        machine_schedules = [MachineSchedule() for _ in range(num_machines)]
        due_dates = []
        for job in range(num_jobs):
            prev_task_end = 0
            machine_order = np.random.permutation(num_machines)
            for machine in machine_order:
                duration = int(distributions[f"dur_{job}_on_{machine}"].mean())
                start_time = machine_schedules[machine].find_earliest_slot(
                    prev_task_end, duration
                )
                end_time = start_time + duration
                machine_schedules[machine].add_task(start_time, end_time)
                prev_task_end = end_time
            due_dates.append(prev_task_end)

        # store constants
        constants["num_jobs"] = num_jobs
        constants["num_machines"] = num_machines
        constants["max_rand_mean"] = max_rand_mean
        constants["seed"] = seed
        for job in range(num_jobs):
            constants[f"due_date_{job}"] = due_dates[job]
        constants["weight_makespan"] = 0
        constants["weight_tardy_jobs"] = 0
        constants["weight_total_flow_time"] = 1
        constants["weight_total_tardiness"] = 100
        constants["weight_total_earliness"] = 0
        constants["weight_max_tardiness"] = 0
        constants["weight_max_lateness"] = 0

        return distributions, constants
