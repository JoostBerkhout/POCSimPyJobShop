from experiments.configs.base_exp_config import exp_config
from experiments.utils.SimheuristicSpec import SimheuristicSpec
from simpyjobshop.simheuristics import (
    # SimulateLastSolutionsConfig,
    # simulate_last_solutions,
    # StandardSimheuristicConfig,
    # standard_simheuristic,
    DynamicSimheuristicConfig,
    dynamic_simheuristic,
)

# Simheuristic configurations
# det_opt_conf_1: DeterministicOptimizationConfig = {
#     "det_repr": "mean",
# }
# det_opt_conf_2: DeterministicOptimizationConfig = {
#     "det_repr": 0.6,
# }
# det_opt_conf_3: DeterministicOptimizationConfig = {
#     "det_repr": 0.7,
# }
# det_opt_conf_4: DeterministicOptimizationConfig = {
#     "det_repr": 0.8,
# }
# sim_last_config_1: SimulateLastSolutionsConfig = {
#     "det_repr": "mean",
#     "max_size_elite_set": 5,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# sim_last_config_2: SimulateLastSolutionsConfig = {
#     "det_repr": "mean",
#     "max_size_elite_set": 10,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# sim_last_config_3: SimulateLastSolutionsConfig = {
#     "det_repr": 0.6,
#     "max_size_elite_set": 5,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# sim_last_config_4: SimulateLastSolutionsConfig = {
#     "det_repr": 0.6,
#     "max_size_elite_set": 10,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# sim_last_config_5: SimulateLastSolutionsConfig = {
#     "det_repr": 0.7,
#     "max_size_elite_set": 5,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# sim_last_config_6: SimulateLastSolutionsConfig = {
#     "det_repr": 0.7,
#     "max_size_elite_set": 10,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# sim_last_config_7: SimulateLastSolutionsConfig = {
#     "det_repr": 0.8,
#     "max_size_elite_set": 5,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# sim_last_config_8: SimulateLastSolutionsConfig = {
#     "det_repr": 0.8,
#     "max_size_elite_set": 10,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# stand_simh_configs: list[StandardSimheuristicConfig] = []
# idx = 1
# time_limit = exp_config["time_limit"]
# for num_sims in [20, 40]:
#     for pre_sim in [0, 10]:
#         for det_repr in [0.7, 0.8]:
#             stand_simh_configs.append(
#                 {
#                     "det_repr": det_repr,
#                     "num_sims": num_sims,
#                     "max_size_elite_set": 10,
#                     "frac_budget_before_sims": pre_sim / time_limit,
#                     "frac_budget_final_elites_sim": 30 / time_limit,
#                 }
#             )
#             # print(f"Std. simheuristic {idx} & {det_repr}-quantiles "
#             #       f"& {pre_sim}s & {num_sims} \\\\ ")  # for LaTeX table
#             idx += 1
dyn_simh_configs: list[DynamicSimheuristicConfig] = []
time_limit = exp_config["time_limit"]
idx = 1
for max_time_per_cp_solve in [45, 60]:
    for final_sim_time in [15, 30]:
        dyn_simh_configs.append(
            {
                "num_sims": 20,
                "max_size_elite_set": 10,
                "score_finding_new_elite": 1,
                "score_finding_new_best": 3,
                "init_score": 1,
                "max_time_per_cp_solve": max_time_per_cp_solve,
                "consider_mean": False,
                "quantiles": [0.6, 0.7, 0.8],
                "frac_budget_before_sims": 0,
                "frac_budget_final_elites_sim": final_sim_time / time_limit,
            }
        )
        # print(
        #     f"Dyn. simh. longer {idx} & {max_time_per_cp_solve}s &"
        #     f" {final_sim_time}s\\\\ "
        # )  # LaTeX
        idx += 1

simheuristics: list[SimheuristicSpec] = [
    # SimheuristicSpec("det_opt", deterministic_optimization, det_opt_conf),
    # SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config),
    # SimheuristicSpec(
    #     f"std_simh_{idx + 1}", standard_simheuristic, stand_simh_config
    # )
    # for idx, stand_simh_config in enumerate(stand_simh_configs)
    SimheuristicSpec(
        f"dyn_simh_{idx + 1}", dynamic_simheuristic, dyn_simh_config
    )
    for idx, dyn_simh_config in enumerate(dyn_simh_configs)
    # SimheuristicSpec("dyn_simh", dynamic_simheuristic, dyn_simh_config),
]
