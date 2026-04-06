import time
from typing import Tuple, TypedDict

import wandb

from pyjobshop import Result, SolveStatus
from simpyjobshop.EliteSet import EliteSet
from simpyjobshop.problems import Problem
from simpyjobshop.Simulator import Simulator
from simpyjobshop.SolutionCallback import SolutionCallback
from simpyjobshop.utils import init_wandb


class SimulateNearOptimumConfig(TypedDict):
    det_repr: float | str  # float in (0, 1) for a quantile or str "mean"
    num_sims: int  # number of simulations to run for each new best found
    max_size_elite_set: int  # maximum size of the elite set to keep
    frac_budget_final_elites_sim: float  # fraction of time budget final sims


def simulate_near_optimum(
    problem: Problem,
    simh_config: SimulateNearOptimumConfig,
    exp_config: dict[str, int],
    wandb_config: dict[str, str] | None = None,
) -> Tuple[SolutionCallback, Result, dict[str, float]]:
    """
    Iteratively applies deterministic optimization to structurally find all
    top solutions and simulate those.

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
    det_repr = simh_config["det_repr"]
    num_sims = simh_config["num_sims"]
    max_size_elite_set = simh_config["max_size_elite_set"]
    time_final_sims = simh_config["frac_budget_final_elites_sim"] * time_limit
    time_det_opt = time_limit - time_final_sims

    # Generate problem data and build model
    data_generator = problem.build_data_generator()
    data = data_generator.by_key(det_repr)
    model = problem.concrete_model(data)

    # Init elite set and callback (for uniformity and logging events)
    elite_set = EliteSet()
    callback = SolutionCallback(problem=problem, num_sims=num_sims)
    callback.solutions = elite_set
    remaining_time = time_det_opt - (time.time() - start_time_exp)

    # Keep finding top solutions until time limit is reached
    while remaining_time > 0:
        # Solve the problem with deterministic optimization
        callback._log_event(f"remaining time: { remaining_time}s")
        start_time_solver = time.time()
        excl_sols = [es.solution for es in elite_set]
        result = model.solve(
            display=False,
            time_limit=remaining_time,
            num_workers=exp_config["num_workers"],
            exclude_solutions=excl_sols,
            initial_solution=None  # excl_sols[-1] if len(excl_sols) > 0 else None,
        )
        solver_time_spent = time.time() - start_time_solver
        callback._log_event(f"Time spend in solver: {solver_time_spent}s")
        solution_found_status = {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
        if result.status not in solution_found_status:
            break  # No more solutions
        best_solution = result.best
        best_obj_val = result.objective

        # Log information
        callback._log_event("Start simulating new solution.")

        # Simulate best solution
        sim = Simulator(problem, best_solution)
        sim.simulate_and_store_data(num_sims)

        # Add the best solution to the elite set
        elite_set.add(
            solution=best_solution,
            simulator=sim,
            objective=best_obj_val,  # Note: Objective value for deter. repr.
            metadata={"time_spent": time.time() - start_time_exp},
        )

        # Log information
        callback._log_event(
            f"New top solution found with determ. obj. val.: {best_obj_val} "
            f"and stoch. obj. val: {sim.mean}."
        )

        if len(elite_set) >= max_size_elite_set:
            callback._log_event(
                "Elite set reached maximum size: terminate determ. opt."
            )
            break

        # Update remaining time
        remaining_time = time_det_opt - (time.time() - start_time_exp)

    # Log info and times
    remaining_time = time_det_opt - (time.time() - start_time_exp)
    if remaining_time <= 0:
        callback._log_event(
            "Time limit reached during deterministic optimization phase. "
            f"Last optimization status is {result.status}."
        )
    else:
        callback._log_event(
            f"Deterministic optimization phase finished with status "
            f"{result.status} while there was {remaining_time}s left."
        )
    time_spent = time.time() - start_time_exp
    durations["solver_phase_dur"] = time_spent

    # Final simulation of the elite solutions for the remaining time
    rem_time_final_sims = min(time_final_sims, time_limit - time_spent)
    elite_set.simulate_to_time_limit(rem_time_final_sims)

    # Log times
    exp_duration = time.time() - start_time_exp
    durations["final_sim_phase_dur"] = exp_duration - time_spent
    durations["total_exp_dur"] = exp_duration

    if exp_config["use_wandb"]:
        wandb.finish()

    return callback, result, durations
