import itertools
from typing import Dict

import numpy as np

from pyjobshop.simheuristic.DataGenerator import DataGenerator
from pyjobshop.simheuristic.DiscreteRV import DiscreteRV, SeededPoisson


class DataGeneratorBuilder:
    """
    Builder class for constructing DataGenerator instances.

    Attributes
    ----------
    distributions : Dict[str, DiscreteRV]
        Stores distributions for the parameters being configured.
    constants : Dict[str, Any]
        Stores constants (without uncertainty).
    """

    def __init__(self, seed: int = 0) -> None:
        # the user specifies self.constants and self.distributions below
        self.distributions: Dict[str, DiscreteRV] = {}
        self.constants: Dict[str, int] = {}

        num_jobs = 10
        max_rand_mean = 10

        # job durations
        np.random.seed(seed)  # for reproducibility
        mean_job_durations = []
        for i in range(num_jobs):
            mean_job_duration = np.random.randint(max_rand_mean)
            mean_job_durations.append(mean_job_duration)
            gen = SeededPoisson(lam=mean_job_duration, seed=i)
            self.distributions[f"duration_{i}"] = gen

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
        self.constants["num_jobs"] = num_jobs
        self.constants["max_rand_mean"] = max_rand_mean
        self.constants["seed"] = seed
        for (i, j), setup_time in setup_times.items():
            self.constants[f"setup_time_{i}_{j}"] = setup_time
        for i, due_date in enumerate(due_dates):
            self.constants[f"due_date_{i}"] = due_date
        self.constants["weight_makespan"] = 1
        self.constants["weight_tardy_jobs"] = 0
        self.constants["weight_total_flow_time"] = 0
        self.constants["weight_total_tardiness"] = 100
        self.constants["weight_total_earliness"] = 0
        self.constants["weight_max_tardiness"] = 0
        self.constants["weight_max_lateness"] = 0

    def build(self) -> DataGenerator:
        """
        Constructs the DataGenerator instance.

        Returns
        -------
        DataGenerator
            A DataGenerator with the configured distributions and seed.
        """

        # make (deep) copy of distributions to avoid shared state
        distributions = {k: v.copy() for k, v in self.distributions.items()}

        return DataGenerator(distributions, self.constants)
