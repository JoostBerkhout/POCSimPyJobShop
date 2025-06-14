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


class JobShopBase(Problem):
    """
    Base implementation of a job shop scheduling problem.
    It can be inherited by specific job shop  instances that specify
    prob_random_duration and weight_total_tardiness.
    """

    prob_random_duration: float
    weight_total_tardiness: int

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        # Unpack data
        num_jobs = data["num_jobs"]
        num_tasks = data["num_tasks"]
        num_machines = data["num_machines"]
        due_dates = [
            data[f"due_date_{job_idx}"] for job_idx in range(num_jobs)
        ]
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

        # Create all jobs
        tasks = {}
        for job_idx in range(num_jobs):
            job = model.add_job(
                name=f"Job {job_idx}", due_date=due_dates[job_idx]
            )

            # Create all job's tasks
            for task_idx in range(num_tasks):
                task_tup = (job_idx, task_idx)
                tasks[task_tup] = model.add_task(job, name=f"Task {task_tup}")
                machine_found = False
                for machine_idx in range(num_machines):
                    dur_key = f"dur_{job_idx}_{task_idx}_on_{machine_idx}"
                    if dur_key in data:
                        machine_found = True
                        duration = data[dur_key]
                        machine = machines[machine_idx]
                        break
                assert machine_found
                model.add_mode(tasks[task_tup], machine, duration)

            # Add precedence constraints between job's tasks
            for task_idx in range(num_tasks - 1):
                first = tasks[(job_idx, task_idx)]
                second = tasks[(job_idx, task_idx + 1)]
                model.add_end_before_start(first, second)

        # Set objective
        model.set_objective(**objective_weights)

        return model

    def distribution_data(
        self, seed: int = 0
    ) -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
        distributions: Dict[str, DiscreteRV] = {}
        constants: Dict[str, int] = {}

        num_jobs = 50
        num_tasks = 4
        num_machines = 4
        loc = 1
        max_rand_mean = 15
        prob_random_duration = self.prob_random_duration

        # Set job distributions
        np.random.seed(seed)  # for reproducibility
        gen: DiscreteRV
        machine_schedules = [MachineSchedule() for _ in range(num_machines)]
        for job in range(num_jobs):
            prev_task_end = 0
            for task in range(num_tasks):
                machine = np.random.choice(num_machines)
                mean_job_duration = np.random.randint(max_rand_mean)
                if np.random.rand() < prob_random_duration:
                    gen = SeededPoisson(
                        lam=mean_job_duration,
                        loc=loc,
                        seed=task + job * num_tasks,
                    )
                else:
                    gen = Constant(loc + mean_job_duration)
                task_dur_key = f"dur_{job}_{task}_on_{machine}"
                distributions[task_dur_key] = gen

                # Schedule task on earliest available machine for due date
                mean_task_dur = int(gen.mean())
                start_time = machine_schedules[machine].find_earliest_slot(
                    prev_task_end,
                    mean_task_dur,
                )
                end_time = start_time + mean_task_dur
                machine_schedules[machine].add_task(start_time, end_time)
                prev_task_end = end_time
            constants[f"due_date_{job}"] = prev_task_end

        # store constants
        constants["num_jobs"] = num_jobs
        constants["num_tasks"] = num_tasks
        constants["num_machines"] = num_machines
        constants["max_rand_mean"] = max_rand_mean
        constants["seed"] = seed
        constants["weight_makespan"] = 0
        constants["weight_tardy_jobs"] = 0
        constants["weight_total_flow_time"] = 1
        constants["weight_total_tardiness"] = self.weight_total_tardiness
        constants["weight_total_earliness"] = 0
        constants["weight_max_tardiness"] = 0
        constants["weight_max_lateness"] = 0

        return distributions, constants


class JobShop(JobShopBase):
    prob_random_duration = 0.3
    weight_total_tardiness = 100


class JobShopFullStoch(JobShopBase):
    prob_random_duration = 1.0  # full stochasticity (FS)
    weight_total_tardiness = 100


class JobShopNoTard(JobShopBase):
    prob_random_duration = 0.3
    weight_total_tardiness = 0  # no tardiness


class JobShopFullStochNoTard(JobShopBase):
    prob_random_duration = 1.0  # full stochasticity (FS)
    weight_total_tardiness = 0  # no tardiness
