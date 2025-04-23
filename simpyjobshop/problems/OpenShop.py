import math
from typing import Any, Dict, Tuple

import numpy as np

from pyjobshop import Model
from simpyjobshop.DiscreteRV import (
    Constant,
    DiscreteRV,
    SeededPoisson,
)
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
        due_dates = [data.get(f"due_date_{i}", None) for i in range(num_jobs)]

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

        num_jobs = 60
        num_machines = 5
        loc = 1
        max_rand_mean = 15
        prob_random_duration = 0.3

        # Set job distributions
        np.random.seed(seed)  # for reproducibility
        gen: DiscreteRV
        exp_job_durs = []
        jobs_durs = []
        for job in range(num_jobs):
            exp_job_dur = 0
            job_durs = []
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
                exp_job_dur += loc + mean_job_duration
                job_durs.append(loc + mean_job_duration)
            exp_job_durs.append(exp_job_dur)
            jobs_durs.append(job_durs)

        # Set meaningful due dates
        due_dates = [exp_job_durs[0]]
        slack = 2
        for job_idx in range(1, num_jobs):
            new_due_date = math.ceil(
                due_dates[-1] + max(jobs_durs[job_idx]) + slack
            )
            constants[f"due_date_{job_idx}"] = new_due_date
            due_dates.append(new_due_date)
            print(constants[f"due_date_{job_idx}"])

        # store constants
        constants["num_jobs"] = num_jobs
        constants["num_machines"] = num_machines
        constants["max_rand_mean"] = max_rand_mean
        constants["seed"] = seed
        for job_idx in range(num_jobs):
            constants[f"due_date_{job_idx}"] = due_dates[job_idx]
        constants["weight_makespan"] = 1
        constants["weight_tardy_jobs"] = 0
        constants["weight_total_flow_time"] = 0
        constants["weight_total_tardiness"] = 100
        constants["weight_total_earliness"] = 0
        constants["weight_max_tardiness"] = 0
        constants["weight_max_lateness"] = 0

        # # create meaningful due dates
        # self.distributions = distributions
        # self.constants = constants
        #
        # # Generate problem data and build model
        # data_generator = self.build_data_generator()
        # data = data_generator.int_mean()
        # model = self.concrete_model(data)
        # result = model.solve(
        #     display=False,
        #     time_limit=10,
        #     num_workers=1,
        #     )
        # # Print the results
        # print("Results:")
        # print("========")
        # print(f"Status: {result.status}")
        # print(f"runtime: {result.runtime}")
        # print(f"runtime: {result.objective}")
        # # due_date_slack = 4
        # tasks = result.best.tasks
        # for job_idx in range(num_jobs):
        #     job_tasks = job_to_tasks[job_idx]
        #     due_date = max([tasks[i].end for i in job_tasks])
        #     constants[f"due_date_{job_idx}"] = due_date
        #     print(constants[f"due_date_{job_idx}"])
        #
        # constants["weight_makespan"] = 1
        # constants["weight_total_tardiness"] = 100
        # constants["weight_total_flow_time"] = 0

        return distributions, constants
