from typing import Any, Dict, Tuple

import numpy as np

from pyjobshop import Model
from simpyjobshop.DiscreteRV import (
    Constant,
    DiscreteRV,
    SeededPoisson,
)
from simpyjobshop.problems.Problem import Problem


class HybridFlowShop(Problem):
    """
    Specific implementation of a hybrid flow shop. Taken from the notebook
    hybrid_flow_shop.ipynb.
    """

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        # Unpack data
        num_jobs = data["num_jobs"]
        num_stages = data["num_stages"]
        num_machines = [
            data[f"num_machines_on_{s}"] for s in range(num_stages)
        ]
        durations = {
            (job, stage): data[f"duration_{job}_on_{stage}"]
            for job in range(num_jobs)
            for stage in range(num_stages)
        }
        due_dates = [data[f"due_date_{j}"] for j in range(num_jobs)]
        objective_weights = {
            "weight_makespan": data["weight_makespan"],
            "weight_tardy_jobs": data["weight_tardy_jobs"],
            "weight_total_flow_time": data["weight_total_flow_time"],
            "weight_total_tardiness": data["weight_total_tardiness"],
            "weight_total_earliness": data["weight_total_earliness"],
            "weight_max_tardiness": data["weight_max_tardiness"],
            "weight_max_lateness": data["weight_max_lateness"],
        }

        def machine_name(machine, stage):
            return f"$M_{{{machine}{stage}}}$"

        def job_name(job: int, stage: int):
            return f"$t_{{{job}{stage}}}$"

        # Create model
        model = Model()
        stage2machines = {}
        for k in range(num_stages):
            stage2machines[k] = [
                model.add_machine(name=machine_name(m, k))
                for m in range(num_machines[k])
            ]

        # Add jobs, tasks, and modes to the model
        jobs = [
            model.add_job(due_date=due_dates[job]) for job in range(num_jobs)
        ]

        for j, job in enumerate(jobs):
            tasks = [
                model.add_task(job=job, name=job_name(j, k))
                for k in range(num_stages)
            ]

            for stage in range(num_stages):
                for machine in stage2machines[stage]:
                    duration = durations[j, stage]
                    model.add_mode(tasks[stage], machine, duration)

            for idx in range(num_stages - 1):
                first = tasks[idx]
                second = tasks[idx + 1]
                model.add_end_before_start(first, second)

        # Set objective
        model.set_objective(**objective_weights)

        return model

    def distribution_data(
        self, seed: int = 0
    ) -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
        distributions: Dict[str, DiscreteRV] = {}
        constants: Dict[str, int] = {}

        num_jobs = 60
        num_stages = 3
        num_machines = [4, 5, 4]
        loc = 1
        max_rand_mean = 15
        prob_random_duration = 1.0

        # Job durations
        np.random.seed(seed)  # for reproducibility
        mean_job_durations = {}
        gen: DiscreteRV
        for job in range(num_jobs):
            for stage in range(num_stages):
                mean_job_duration = np.random.randint(max_rand_mean)
                mean_job_durations[job, stage] = mean_job_duration
                if np.random.rand() < prob_random_duration:
                    gen = SeededPoisson(
                        lam=mean_job_duration,
                        loc=loc,
                        seed=stage + job * num_stages,
                    )
                else:
                    gen = Constant(loc + mean_job_duration)
                distributions[f"duration_{job}_on_{stage}"] = gen

        # Calculate meaningful due dates
        finish_times: list[list[int]] = []
        for stage in range(num_stages):
            # Find finish time of previous stage
            if stage == 0:
                finish_times_prev_stage = [0 for _ in range(num_jobs)]
            else:
                finish_times_prev_stage = finish_times[stage - 1]

            # Find a stage schedule to work with
            machine_schedules = np.array_split(
                np.arange(num_jobs), num_machines[stage]
            )

            # Determine finish times of jobs in this stage
            stage_finish_times = [0 for _ in range(num_jobs)]
            durations = [
                mean_job_durations[job, stage] + loc for job in range(num_jobs)
            ]
            for _, schedule in enumerate(machine_schedules):
                first = schedule[0]
                stage_finish_times[first] = (
                    finish_times_prev_stage[first] + durations[first]
                )
                for i, j in zip(schedule[:-1], schedule[1:], strict=True):
                    start_time = max(
                        stage_finish_times[i], finish_times_prev_stage[j]
                    )
                    stage_finish_times[j] = start_time + durations[j]
            finish_times.append(stage_finish_times)
        due_dates = [int(x) for x in finish_times[-1]]

        # store constants
        constants["num_jobs"] = num_jobs
        constants["num_stages"] = num_stages
        for stage, num_machines_stage in enumerate(num_machines):
            constants[f"num_machines_on_{stage}"] = num_machines_stage
        constants["max_rand_mean"] = max_rand_mean
        for i, due_date in enumerate(due_dates):
            constants[f"due_date_{i}"] = due_date
        constants["seed"] = seed
        constants["weight_makespan"] = 1
        constants["weight_tardy_jobs"] = 0
        constants["weight_total_flow_time"] = 0
        constants["weight_total_tardiness"] = 100
        constants["weight_total_earliness"] = 0
        constants["weight_max_tardiness"] = 0
        constants["weight_max_lateness"] = 0

        return distributions, constants
