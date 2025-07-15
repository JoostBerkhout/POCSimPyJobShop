import math
import time
from typing import Dict, List, Optional, Tuple, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from sklearn.linear_model import LinearRegression

from pyjobshop import Result
from simpyjobshop.modeling import (
    find_result_for_expected_data,
    find_solution_for_other_data,
)
from simpyjobshop.problems.Problem import Problem
from simpyjobshop.Simulator import Simulator
from simpyjobshop.SolutionCallback import SolutionCallback
from simpyjobshop.utils import find_schedule_per_resource, init_wandb


class IterDetOptConfig(TypedDict):
    num_sims: int
    frac_budget_each_det_opt: float
    frac_budget_final_elites_sim: float
    init_quantile_level: float  # probability to base init quantiles on
    min_quantile_level: float  # minimum probability for quantiles
    max_quantile_level: float  # maximum probability for quantiles


def iter_det_opt(
    problem: Problem,
    simh_config: IterDetOptConfig,
    exp_config: dict[str, int],
    wandb_config: dict[str, str] | None = None,
) -> Tuple[SolutionCallback, Result, dict[str, float]]:
    """
    First applies deterministic optimization to find a promising solution.
    Then it simulates that solution and checks which parameters are crucial
    for the objective value. Finally, it runs deterministic optimization again
    where the most crucial parameters get a larger quantile.

    Parameters
    ----------
    problem : Problem
        The optimization problem instance (e.g., HybridFlowShop).
    simh_config : StandardSimheuristicConfig
        Configuration for the simheuristic (e.g., number of simulations).
    exp_config : dict[str, int]
        Configuration for the experiment (e.g., time limit).
    wandb_config : dict[str, str], optional
        Configuration for Weights & Biases logging, by default None.

    Returns
    -------
    callback : SolutionCallback
        A callback object that contains the found elite solutions.
    result : Result
        A Result object containing the best found solution and additional
        information about the solver run.
    durations : dict[str, float]
        The durations of the experiment phases in seconds. It does not include
        the loading and closing duration of wandb if used.
    """

    if exp_config["use_wandb"]:
        problem_name = problem.__class__.__name__
        save_config = {"exp_config": exp_config, "simh_config": simh_config}
        init_wandb(problem_name, save_config, wandb_config)

    start_time_exp = time.time()
    durations = {}

    # Init
    time_limit = exp_config["time_limit"]
    num_sims = simh_config["num_sims"]
    time_each_det_opt = simh_config["frac_budget_each_det_opt"] * time_limit
    time_final_sims = simh_config["frac_budget_final_elites_sim"] * time_limit
    total_time_det_opt = time_limit - time_final_sims

    # Algorithm init
    data_generator = problem.build_data_generator()
    param_names = list(data_generator.random_only().keys())
    quantile_levels = {
        k: simh_config["init_quantile_level"] for k in param_names
    }
    # (callback used for uniformity to store solutions:)
    callback = SolutionCallback(problem=problem, num_sims=num_sims)
    best_solution = None
    quantiles_history = QuantilesHistory(param_names)  # to study quantiles

    while time.time() - start_time_exp < total_time_det_opt:
        # Generate problem data and build model
        data = data_generator.quantiles_by_dict(quantile_levels)
        quantiles_history.add({k: data[k] for k in param_names})
        model = problem.concrete_model(data)

        # Solve the problem with deterministic optimization
        if best_solution is not None:
            best_solution, _ = find_solution_for_other_data(
                best_solution, problem, data
            )  # not always needed, but simplifies code
        result = model.solve(
            display=True,
            time_limit=time_each_det_opt,
            num_workers=exp_config["num_workers"],
            initial_solution=best_solution,
        )
        best_solution = result.best
        best_obj_val = result.objective

        # Simulate best solution
        sim = Simulator(problem, best_solution)
        sim.simulate_and_store_data(num_sims)

        # Store solution
        _result = find_result_for_expected_data(best_solution, problem)
        _schedule = find_schedule_per_resource(best_solution)
        if callback.solutions.is_new_schedule(_schedule):
            callback.solutions.add(
                solution=_result.best,
                simulator=sim,
                objective=_result.objective,  # Note: Objective value for mean
                metadata={"time_spent": time.time() - start_time_exp},
            )
            callback._log_event(
                f"After det. opt. (status = {result.status}): objective = "
                f"{best_obj_val}, objective for mean-DCOP = "
                f"{_result.objective}, simulated mean = {sim.mean}."
            )
        else:
            callback._log_event(
                f"After det. opt. (status = {result.status}): "
                f"no new solution found"
            )

        # Find parameter quantiles based on simulation results
        df_rel_diffs = sim.get_df_for_rel_data_diffs_and_results(data)
        quantile_levels = get_param_quantile_levels(
            quantile_levels,
            df_rel_diffs,
            simh_config["min_quantile_level"],
            simh_config["max_quantile_level"],
        )

    print(
        f"Number of quantiles changed "
        f"{len(quantiles_history.changed_params)}"
    )
    for _param in quantiles_history.changed_params:
        print(_param, quantiles_history.history[_param])
    # quantiles_history.plot(max_cols=4, figsize=(15, 12))

    # Log times
    time_spent = time.time() - start_time_exp
    durations["solver_phase_dur"] = time_spent

    # Final simulation of the elite solutions for the remaining time
    callback._log_event("Starting final simulation phase...")
    rem_time_final_sims = min(time_final_sims, time_limit - time_spent)
    callback.solutions.simulate_to_time_limit(rem_time_final_sims)

    # Log times
    exp_duration = time.time() - start_time_exp
    durations["final_sim_phase_dur"] = exp_duration - time_spent
    durations["total_exp_dur"] = exp_duration

    if exp_config["use_wandb"]:
        wandb.finish()

    return callback, result, durations


def get_param_quantile_levels(
    previous_quantile_levels: dict[str, float],
    df_rel_diffs: pd.DataFrame,
    min_quant_lvl: float,
    max_quant_lvl: float,
) -> dict[str, float]:
    """
    Helper function:
    Estimates the importance of each parameter based on simulation results
    using linear regression, and assigns adjusted quantile levels accordingly
    for robust deterministic modeling.

    Parameters
    ----------
    previous_quantile_levels : dict[str, float]
        A dictionary mapping parameter names to previous quantile levels.
    df_rel_diffs : pd.DataFrame
        DataFrame where each row is a simulation instance. Must include a
        column "objective_value" (the dependent variable), and other columns
        represent relative deviations of parameters.
    min_quant_lvl : float
        Minimum quantile level to assign to parameters (e.g., 0.5).
    max_quant_lvl : float
        Maximum quantile level to assign to parameters (e.g., 0.9).

    Returns
    -------
    quantiles_lvls : dict[str, float]
        A dictionary mapping each parameter name to an adjusted quantile level
        based on its estimated (>= 0) linear impact on the objective value.
    """
    dep_var = df_rel_diffs["objective_value"]
    indep_vars = df_rel_diffs.drop(columns=["objective_value"])

    lr_model = LinearRegression()
    lr_model.fit(indep_vars, dep_var)

    params = indep_vars.columns
    max_coefs = [max(coef, 0) for coef in lr_model.coef_]
    sum_coefs = sum(max_coefs)
    scaled_coefs = [coef / sum_coefs for coef in max_coefs]
    quant_lvls = [
        min(min_quant_lvl + 2 * sc, max_quant_lvl) for sc in scaled_coefs
    ]  # the 2 needed to make a difference in the Poisson distribution
    quantile_lvls = dict(zip(params, quant_lvls, strict=False))

    # Average quantile levels with previous ones
    quantile_lvls = {
        k: (previous_quantile_levels[k] + v) / 2
        for k, v in quantile_lvls.items()
    }

    # Uncomment the following to study the coefficients and quantiles
    # df_coefs_params = pd.DataFrame({
    #     "param": indep_vars.columns,
    #     "coefficient": lr_model.coef_,
    #     "max(coefficient, 0)": max_coefs,
    #     "scaled_coefficient": scaled_coefs,
    #     "quantiles_vals": quantile_lvls,
    #     })

    return quantile_lvls


class QuantilesHistory:
    """
    Stores and plots quantile levels over time for multiple parameters.

    Parameters
    ----------
    names : List[str]
        List of parameter names to track.
    """

    def __init__(self, names: List[str]) -> None:
        self.history: Dict[str, List[Optional[float]]] = {
            name: [] for name in names
        }

    def add(self, quantile_dict: Dict[str, float]) -> None:
        """
        Add a new set of quantile levels for all tracked parameters.

        Parameters
        ----------
        quantile_dict : Dict[str, float]
            Dictionary mapping parameter names to quantile levels.
        """
        assert quantile_dict.keys() == self.history.keys()
        for name in self.history:
            self.history[name].append(quantile_dict[name])

    @property
    def changed_params(self) -> list[str]:
        """
        List of parameter names whose quantile levels have changed over time.

        Returns
        -------
        list of str
            Parameter names with non-constant quantile level histories.
        """
        return [
            param_name
            for param_name, q_lvls in self.history.items()
            if len(q_lvls) > 1 and any(x != q_lvls[0] for x in q_lvls[1:])
        ]

    def plot(
        self,
        max_cols: int = 4,
        figsize: tuple = (15, 8),
        show_changed_only: bool = True,
    ) -> None:
        """
        Plot quantile level histories for all parameters.

        Parameters
        ----------
        max_cols : int, optional
            Maximum number of subplot columns, by default 4.
        figsize : tuple, optional
            Size of the entire figure in inches, by default (15, 8).
        show_changed_only : bool, optional
            If True, only plot parameters that have changed over time,
            by default True.
        """
        if show_changed_only:
            param_names = self.changed_params
        else:
            param_names = list(self.history.keys())
        num_params = len(param_names)
        num_cols = min(max_cols, num_params)
        num_rows = math.ceil(num_params / num_cols)

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=figsize, squeeze=False
        )

        for idx, name in enumerate(param_names):
            row, col = divmod(idx, num_cols)
            ax = axes[row][col]
            ax.plot(self.history[name], marker="o", linestyle="-")
            ax.set_title(name)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Quantile level")
            ax.grid(True)

        # Hide unused subplots
        for idx in range(num_params, num_rows * num_cols):
            row, col = divmod(idx, num_cols)
            fig.delaxes(axes[row][col])

        plt.tight_layout()
        plt.show()
