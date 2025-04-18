from typing import Any, Dict, Tuple

from pyjobshop import Model
from simpyjobshop.DiscreteRV import DiscreteRV, SeededPoisson
from simpyjobshop.problems.Problem import Problem


class TwoMachinesTwoJobs(Problem):
    """
    Specific implementation of a single-machine scheduling problem with
    two jobs. Based on objectives_examples.ipynb.
    """

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        # Unpack data
        num_jobs = data["num_jobs"]
        num_machines = data["num_machines"]
        durations = [data[f"duration_{j}"] for j in range(num_jobs)]
        due_dates = [data[f"due_date_{j}"] for j in range(num_jobs)]
        setup_t_1_to_0 = data["setup_time_1_to_0"]
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
        machines = [model.add_machine() for _ in range(num_machines)]
        jobs = [model.add_job(due_date=due_dates[i]) for i in range(num_jobs)]
        tasks = [model.add_task(job=job) for job in jobs]

        # Add modes to the model
        for task, duration in zip(tasks, durations, strict=True):
            for machine in machines:
                model.add_mode(task, machine, duration=duration)

        # Add sequence-dependent setup time
        for machine in machines:
            model.add_setup_time(machine, tasks[1], tasks[0], setup_t_1_to_0)

        # Set objective
        model.set_objective(**objective_weights)

        return model

    def distribution_data(
        self, seed: int = 0
    ) -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
        distributions: Dict[str, DiscreteRV] = {}
        constants: Dict[str, int] = {}

        num_jobs = 2
        num_machines = 2
        setup_time_1_to_0 = 10

        # job durations
        mean_job_durations = [1, 1]
        for i in range(num_jobs):
            gen = SeededPoisson(lam=mean_job_durations[i], seed=i)
            distributions[f"duration_{i}"] = gen

        due_dates = [1, 1]

        # store constants
        constants["num_jobs"] = num_jobs
        constants["num_machines"] = num_machines
        for i, due_date in enumerate(due_dates):
            constants[f"due_date_{i}"] = due_date
        constants["setup_time_1_to_0"] = setup_time_1_to_0
        constants["weight_makespan"] = 1
        constants["weight_tardy_jobs"] = 0
        constants["weight_total_flow_time"] = 0
        constants["weight_total_tardiness"] = 0
        constants["weight_total_earliness"] = 0
        constants["weight_max_tardiness"] = 0
        constants["weight_max_lateness"] = 0

        return distributions, constants
