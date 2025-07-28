import time
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from pyjobshop import Solution
from simpyjobshop.evaluator import evaluator
from simpyjobshop.problems import Problem


class Simulator:
    """
    Simulates a solution's performance for a problem.

    Attributes
    ----------
    problem : Problem
        Problem representation.
    solution : Solution
        Representing the solution to be evaluated.
    data_generator : DataGenerator
        Instance of DataGenerator to generate random data.
    results : List[float]
        List of results from simulation runs.
    """

    def __init__(self, problem: Problem, solution: Solution):
        self.problem = problem
        self.solution = solution
        self.data_generator = problem.build_data_generator()
        self.results: List[float] = []
        self.rand_data_list: List[dict[str, int]] = []

    def parallel_simulate(
        self,
        num_sims: int,
        num_workers: int | None = None,
    ) -> None:
        """
        Performs `num_sims` simulation runs in parallel using process_map.

        Parameters
        ----------
        num_sims : int
            The number of simulations to run.
        num_workers : int, optional
            Number of parallel workers to use, by default None.

        Notes
        -----
        - Results are appended to self.results in order of input.
        """

        data_list = [self.data_generator.random() for _ in range(num_sims)]
        args_list = [
            (self.solution, self.problem, data, 1) for data in data_list
        ]

        results = process_map(
            Simulator._single_sim,
            args_list,
            max_workers=num_workers,
        )

        self.results.extend(results)

    @staticmethod
    def _single_sim(args: tuple) -> float:
        solution, problem, data, num_workers = args
        return evaluator(solution, problem, data, None)

    def simulate(
        self,
        num_sims: int,
        time_limit: float | None = None,
        num_workers: int | None = None,
    ) -> None:
        """
        Performs num_sims simulation runs and stores the results.

        Parameters
        ----------
        num_sims : int
            The number of simulations to run.
        time_limit : float, optional
            Time limit (in seconds) for simulate(), by default None.
        num_workers : int, optional
            Number of workers in evaluator, by default None.
        """

        start_time = None if time_limit is None else time.time()

        for _ in range(num_sims):
            if time_limit is not None:
                assert start_time is not None
                if time.time() - start_time >= time_limit:
                    break

            data = self.data_generator.random()
            result = evaluator(self.solution, self.problem, data, num_workers)
            self.results.append(result)

    def simulate_and_store_data(
        self,
        num_sims: int,
        time_limit: float | None = None,
        num_workers: int | None = None,
    ) -> None:
        """
        Performs num_sims simulation runs and stores results and sampled data.

        Parameters
        ----------
        num_sims : int
            The number of simulations to run.
        time_limit : float, optional
            Time limit (in seconds) for simulate(), by default None.
        num_workers : int, optional
            Number of workers in evaluator, by default None.
        """

        start_time = None if time_limit is None else time.time()

        for _ in range(num_sims):
            if time_limit is not None:
                assert start_time is not None
                if time.time() - start_time >= time_limit:
                    break

            rand_data = self.data_generator.random_only()
            data = rand_data | self.data_generator.constants
            result = evaluator(self.solution, self.problem, data, num_workers)
            self.results.append(result)
            self.rand_data_list.append(rand_data)

    def get_rel_data_diffs(
        self, ref_data: Optional[dict[str, int]] = None
    ) -> list[dict[str, float]]:
        """
        Computes relative differences of random data from reference data.

        Parameters
        ----------
        ref_data : dict[str, int], optional
            Reference data to compute relative differences from. If None,
            the mean of the random data is used.

        Returns
        -------
        list[dict[str, float]]
            List of dictionaries where each dictionary contains the relative
            differences of the random data from the reference data.
        """

        assert len(self.rand_data_list) == len(self.results)
        if ref_data is None:
            ref_data = self.data_generator.int_mean()
        rel_diffs = [
            {k: v / ref_data[k] - 1 for k, v in data.items()}
            for data in self.rand_data_list
        ]
        return rel_diffs

    def get_df_for_rel_data_diffs_and_results(
        self, ref_data: Optional[dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with relative data differences and results.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with relative data differences and results.
        """

        rel_diffs = self.get_rel_data_diffs(ref_data)
        assert len(rel_diffs) == len(self.results)
        for idx, result in enumerate(self.results):
            rel_diffs[idx]["objective_value"] = result
        df_data = pd.DataFrame(rel_diffs)
        return df_data

    @property
    def num_sims(self) -> int:
        """
        Returns the number of simulation runs.

        Returns
        -------
        int
            The number of simulation runs.
        """
        return len(self.results)

    @property
    def mean(self) -> float:
        """
        Computes the mean of the simulation results.

        Returns
        -------
        float
            The mean of simulation results, or None if no simulations were run.
        """
        return np.mean(self.results) if len(self.results) > 0 else None

    @property
    def variance(self) -> float:
        """
        Computes the variance of the simulation results.

        Returns
        -------
        float
            The variance of simulation results, or None if no sims. were run.
        """
        return np.var(self.results) if len(self.results) > 0 else None
