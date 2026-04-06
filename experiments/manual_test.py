"""Messy code to do some testing. """
import time

from matplotlib import pyplot as plt

import numpy as np

from pyjobshop import Solution, ProblemData
from pyjobshop.plot import plot_machine_gantt
from pyjobshop.plot.plot_machine_gantt import (
    plot_machine_gantt_for_OS,
    plot_machine_gantt_with_tardiness,
    )
from simpyjobshop.modeling import (
    find_result_for_expected_data,
    find_solution_for_other_data,
    )
from simpyjobshop.problems import (
    OpenShop, HybridFlowShop, ParallelMachines, OpenShopNoTard,
    ParallelMachinesSmallRegrTest,
    )
from simpyjobshop.problems import OpenShopNoTard, OpenShop
from simpyjobshop.problems.JobShop import JobShop, JobShopNoTard
from simpyjobshop.problems.JobShopEasy import JobShopEasy
from simpyjobshop.problems.OpenShopEasy import OpenShopEasy
from experiments.utils.SimheuristicSpec import SimheuristicSpec
from simpyjobshop.problems.ParallelMachinesSmall import ParallelMachinesSmall
from simpyjobshop.simheuristics import (
    DynamicSimheuristicConfig,
    SimulateLastSolutionsConfig,
    StandardSimheuristicConfig,
    standard_simheuristic, DeterministicOptimizationConfig,
    SimulateNearOptimumConfig, simulate_near_optimum,
    deterministic_optimization, dynamic_simheuristic, IterDetOptConfig,
    iter_det_opt,
    )

problem_seed = 1
# problem = ParallelMachines(seed=problem_seed)
# problem = ParallelMachinesSmall(seed=problem_seed)
problem = ParallelMachinesSmallRegrTest()
# problem = OpenShop(seed=problem_seed)
# problem = OpenShopNoTard(seed=problem_seed)
# problem = HybridFlowShop(seed=problem_seed)
# problem = FlexibleJobShop(seed=problem_seed)
# problem = OpenShop(seed=problem_seed)
# problem = JobShop(seed=problem_seed)
# problem = JobShopNoTard(seed=problem_seed)
# problem = JobShopEasy(seed=problem_seed)
# problem = OpenShopEasy(seed=problem_seed)

exp_config = {
    "time_limit": 10,
    "num_workers": 16,  # (per instance)
    "use_wandb": False,
    # only relevant for cli_run_experiments.py and submit_slurm_job.py:
    "num_rand_experiments": None,
    "num_parallel_instances": None,
    "num_sims_for_true_expec_objective": None,
    }
# exp_config["objective_bound"] = 5  # the solver will try to find all sols

# Simheuristic configurations
det_opt_config: DeterministicOptimizationConfig = {
    "det_repr": 0.8,
    }
std_simh_config: StandardSimheuristicConfig = {
    "det_repr": "mean",
    "num_sims": 100,
    "max_size_elite_set": 10**10,
    "frac_budget_before_sims": 0,
    "frac_budget_final_elites_sim": 0,
    }
dyn_simh_config: DynamicSimheuristicConfig = {
    "num_sims": 20,
    "max_size_elite_set": 5,
    "score_finding_new_elite": 1,
    "score_finding_new_best": 2,
    "init_score": 1,
    "max_time_per_cp_solve": 5,
    "consider_mean": True,
    "quantiles": [0.6, 0.7, 0.8],
    "frac_budget_before_sims": 0 / exp_config["time_limit"],
    "frac_budget_final_elites_sim": 0 / exp_config["time_limit"],
    # frac_budget_final_elites_sim should be enough to sim. final
    }
sim_last_config: SimulateLastSolutionsConfig = {
    "det_repr": "mean",
    "max_size_elite_set": 10,
    "frac_budget_final_elites_sim": 0.4,  # should be enough to sim. final
    }
near_opt_sim_config: SimulateNearOptimumConfig = {
    "det_repr": "mean",
    "num_sims": 40,
    "max_size_elite_set": 10 ** 10,
    "frac_budget_final_elites_sim": 0,
    }
# iter_det_opt_config: IterDetOptConfig = {
#     "num_sims": 100,
#     "frac_budget_each_det_opt": 0.10,  # number of top parameters to adjust
#     "frac_budget_final_elites_sim": 0,  # fraction of the time limit for each det opt
#     }
simheuristics: list[SimheuristicSpec] = [
    SimheuristicSpec("std_simh", standard_simheuristic, std_simh_config),
    # SimheuristicSpec("det_opt", deterministic_optimization, det_opt_config),
    # SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config),
    # SimheuristicSpec("dyn_simh", dynamic_simheuristic, dyn_simh_config),
    SimheuristicSpec("near_opt_sim", simulate_near_optimum, near_opt_sim_config),
    # SimheuristicSpec("iter_det_opt", iter_det_opt, iter_det_opt_config),
    ]
simheuristic = simheuristics[0]

wandb_config = (
    {
        "project_name": "Test",
        "job_name": simheuristic.name,
        "run_name": f"{problem.__class__}_{simheuristic.name}_seed_{problem_seed}",
        }
    if exp_config["use_wandb"]
    else None
)

callback, last_CP_results, durations = simheuristic.fun(
    problem,
    simheuristic.config,
    exp_config,
    wandb_config=wandb_config,
    )
callback.print_log()

# Find elite
elite_solutions = callback.solutions
if elite_solutions.all_simulated():
    elite = elite_solutions.get_best_mean_solution()
elif elite_solutions.none_simulated():
    elite = elite_solutions.get_best_deterministic_solution()
else:
    raise Exception(
        "Some elites are simulated and some not. This is not expected."
        )

# elite_solutions.get_best_mean_solution().simulator.simulate(10000)

# # Plot all Gantt charts
# _plot_machine_gantt = plot_machine_gantt_for_OS if isinstance(problem, OpenShop) else plot_machine_gantt_with_tardiness
# data_generator = problem.build_data_generator()
# data = data_generator.int_mean()
# model = problem.concrete_model(data)
# for es in elite_solutions:
#     # Find results for elite with expected data
#     exp_data_result = find_result_for_expected_data(es.solution, problem)
#     es.objective = exp_data_result.objective
#     if len(elite_solutions) > 5:
#         break  # skip plotting for large number of solutions
#     _plot_machine_gantt(
#         es.solution,
#         model.data(),
#         plot_labels=False,
#         title=f"CP solution with obj val {es.objective} "
#               f"and mean obj val {es.simulator.mean}"
#         )
#     plt.show()
#
# elite_solutions.print_summary()
#
# # Plot objective value vs. mean objective value
# plt.figure()
# mean_obj_vals = [es.simulator.mean for es in elite_solutions]
# obj_vals = [es.objective for es in elite_solutions]
# plt.plot(obj_vals, mean_obj_vals, "o")
# plt.xlabel("Objective value")
# plt.ylabel("Mean objective value")
# # plt.xlim(8000, 11000)
# # plt.ylim(8800, 9300)
# plt.title("Objective value vs. mean objective value")
# plt.grid()
# plt.show()
#
#
# def get_all_completion_times(solution: Solution, data: ProblemData):
#     """
#     Get all completion times from a solution.
#     """
#     completion_times = [0 for _ in range(data.num_jobs)]
#     for task, task_data in zip(solution.tasks, data.tasks):
#         job = task_data.job
#         completion_times[job] = max(completion_times[job], task.end)
#     return completion_times
#
# def get_due_date_slack(data: ProblemData, completion_times: list[int]):
#     """
#     Get all due date slack from a solution.
#     """
#     slack = [
#         data.jobs[job].due_date - compl_time
#         for (job, compl_time) in enumerate(completion_times)
#         ]
#     return slack
#
#
# # Plot best solutions
# best_DCOP_solution = elite_solutions[np.argmin(obj_vals)]
# best_SCOP_solution = elite_solutions[np.argmin(mean_obj_vals)]
# for es in [best_DCOP_solution, best_SCOP_solution]:
#     _plot_machine_gantt(
#         es.solution,
#         model.data(),
#         plot_labels=False,
#         title=f"CP solution with obj val {es.objective} "
#               f"and mean obj val {es.simulator.mean}"
#         )
#     plt.show()
#
# # Plot slacks in scatter plot
# plt.figure()
# compl_best_DCOP_solution = get_all_completion_times(best_DCOP_solution.solution, model.data())
# compl_best_SCOP_solution = get_all_completion_times(best_SCOP_solution.solution, model.data())
# slack_best_DCOP_solution = get_due_date_slack(model.data(), compl_best_DCOP_solution)
# slack_best_SCOP_solution = get_due_date_slack(model.data(), compl_best_SCOP_solution)
# plt.scatter(slack_best_DCOP_solution, slack_best_SCOP_solution)
# plt.xlabel("Slack of DCOP solution")
# plt.ylabel("Slack of SCOP solution")
# plt.title("Slack for each job for DCOP vs. SCOP solutions")
# # also draw the line y=x
# plt.plot(
#     [min(slack_best_DCOP_solution), max(slack_best_SCOP_solution)],
#     [min(slack_best_DCOP_solution), max(slack_best_SCOP_solution)],
#     color="red",
#     linestyle="--",
#     )
# plt.grid()
# plt.show()
#
# # Plot simulation results in scatter plot
# plt.figure()
# sims_best_DCOP_solution = best_DCOP_solution.simulator.results
# sims_best_SCOP_solution = best_SCOP_solution.simulator.results
# plt.scatter(sims_best_DCOP_solution, sims_best_SCOP_solution)
# plt.xlabel("Simulation result of DCOP solution")
# plt.ylabel("Simulation result of SCOP solution")
# plt.title("Simulation results of DCOP vs. SCOP solutions")
# # also draw the line y=x
# plt.plot(
#     [min(sims_best_DCOP_solution), max(sims_best_SCOP_solution)],
#     [min(sims_best_DCOP_solution), max(sims_best_SCOP_solution)],
#     color="red",
#     linestyle="--",
# )
# plt.grid()
# plt.show()

# # Plot Gantt charts for different simulated data
# for seed in [0]:
#     data = data_generator.random()
#     model = problem.concrete_model(data)
#     for sol in [best_DCOP_solution.solution, best_SCOP_solution.solution]:
#         sol, obj = find_solution_for_other_data(sol, problem, data)
#         _plot_machine_gantt(
#             sol,
#             model.data(),
#             plot_labels=False,
#             title=f"CP solution with obj val {obj} "
#             )
#         plt.show()

# # Estimate simulation time
# num_sims = 100 - elite.simulator.num_sims
# start_time = time.time()
# elite.simulator.simulate(num_sims=num_sims)
# duration = time.time() - start_time
# print(f"Time per simulation: {duration / num_sims:.2f} seconds")

# # Test impact of parallel simulation
# from simpyjobshop.Simulator import Simulator
# # Time the simulation time without and with parallel
# num_sims = 80
# start_time = time.time()
# sim_1 = Simulator(problem, best_DCOP_solution.solution)
# sim_1.parallel_simulate(num_sims, num_workers=8)
# elapsed_time_parallel = time.time() - start_time
# start_time = time.time()
# sim_2 = Simulator(problem, best_DCOP_solution.solution)
# sim_2.simulate(num_sims)
# elapsed_time_serial = time.time() - start_time
# assert sim_1.results == sim_2.results, "Simulation results do not match!"
# print(f"Elapsed time parallel: {elapsed_time_parallel:.2f} seconds")
# print(f"Elapsed time serial: {elapsed_time_serial:.2f} seconds")



