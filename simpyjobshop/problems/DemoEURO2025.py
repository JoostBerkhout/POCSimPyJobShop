from typing import Any, Dict, Tuple

from pyjobshop import Model
from simpyjobshop.DiscreteRV import Constant, DiscreteRV, SeededPoisson
from simpyjobshop.problems.Problem import Problem


class ParallelMachinesEURO2025(Problem):
    """
    Specific implementation of a parallel machine scheduling problem.
    """

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        # Unpack data
        num_jobs = data["num_jobs"]
        num_machines = data["num_machines"]
        durations = [data[f"duration_{j}"] for j in range(num_jobs)]
        due_dates = [data[f"due_date_{j}"] for j in range(num_jobs)]
        objective_weights = {
            "weight_makespan": data["weight_makespan"],
            "weight_total_tardiness": data["weight_total_tardiness"],
        }

        # Create model
        model = Model()
        jobs = [model.add_job(due_date=due_dates[j]) for j in range(num_jobs)]
        tasks = [model.add_task(job) for job in jobs]  # one task per job
        machines = [model.add_machine() for _ in range(num_machines)]

        # Add modes to the model
        for machine in machines:
            for task, duration in zip(tasks, durations, strict=True):
                model.add_mode(task, machine, duration=duration)

        # Set objective
        model.set_objective(**objective_weights)

        return model

    def distribution_data(
        self, seed: int = 0
    ) -> Tuple[Dict[str, DiscreteRV], Dict[str, int]]:
        distributions: Dict[str, DiscreteRV] = {}
        constants: Dict[str, int] = {}

        # Job durations
        mean_job_durations = [1, 2, 3]
        job_distrs = [Constant, SeededPoisson, Constant]
        durations = [
            dist(m, seed=m)
            for dist, m in zip(job_distrs, mean_job_durations, strict=True)
        ]

        # Due dates
        due_dates = [4, 4, 4]

        # store constants
        constants["num_jobs"] = 3
        constants["num_machines"] = 2
        constants["seed"] = seed
        for i, dur in enumerate(durations):
            distributions[f"duration_{i}"] = dur
        for i, due_date in enumerate(due_dates):
            constants[f"due_date_{i}"] = due_date
        constants["weight_makespan"] = 1
        constants["weight_total_tardiness"] = 100

        return distributions, constants
