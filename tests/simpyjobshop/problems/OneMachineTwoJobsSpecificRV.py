from typing import Any, Dict, Tuple

from pyjobshop import Model
from simpyjobshop.DiscreteRV import Constant, CustomRV, DiscreteRV
from simpyjobshop.problems import Problem


class OneMachineTwoJobsSpecificRV(Problem):
    """
    Specific implementation of a single-machine scheduling problem with
    two jobs. Based on objectives_examples.ipynb.
    """

    @staticmethod
    def concrete_model(data: Dict[str, Any]) -> Model:
        # Unpack data
        num_jobs = data["num_jobs"]
        durations = [data[f"duration_{j}"] for j in range(num_jobs)]
        objective_weights = {
            "weight_total_flow_time": data["weight_total_flow_time"],
        }

        # Create model
        model = Model()
        machine = model.add_machine()
        jobs = [model.add_job() for _ in range(num_jobs)]
        tasks = [model.add_task(job=job) for job in jobs]

        # Add modes to the model
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

        num_jobs = 2

        # Set job durations
        distributions["duration_0"] = CustomRV([0, 3], [2 / 3, 1 / 3], seed)
        distributions["duration_1"] = Constant(2)

        # Store constants
        constants["num_jobs"] = num_jobs
        constants["weight_total_flow_time"] = 1

        return distributions, constants
