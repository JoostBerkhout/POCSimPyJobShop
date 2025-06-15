from experiments.configs.base_exp_config import exp_config
from experiments.utils.SimheuristicSpec import SimheuristicSpec
from simpyjobshop.simheuristics import (
    SimulateLastSolutionsConfig,
    simulate_last_solutions,
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
sim_last_config_1: SimulateLastSolutionsConfig = {
    "det_repr": "mean",
    "max_size_elite_set": 5,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
sim_last_config_2: SimulateLastSolutionsConfig = {
    "det_repr": "mean",
    "max_size_elite_set": 10,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
sim_last_config_3: SimulateLastSolutionsConfig = {
    "det_repr": 0.6,
    "max_size_elite_set": 5,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
sim_last_config_4: SimulateLastSolutionsConfig = {
    "det_repr": 0.6,
    "max_size_elite_set": 10,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
sim_last_config_5: SimulateLastSolutionsConfig = {
    "det_repr": 0.7,
    "max_size_elite_set": 5,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
sim_last_config_6: SimulateLastSolutionsConfig = {
    "det_repr": 0.7,
    "max_size_elite_set": 10,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
sim_last_config_7: SimulateLastSolutionsConfig = {
    "det_repr": 0.8,
    "max_size_elite_set": 5,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
sim_last_config_8: SimulateLastSolutionsConfig = {
    "det_repr": 0.8,
    "max_size_elite_set": 10,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
# stand_simh_config: StandardSimheuristicConfig = {
#     "det_repr": "mean",
#     "num_sims": 20,
#     "max_size_elite_set": 5,
#     "frac_budget_before_sims": 10 / exp_config["time_limit"],
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }
# dyn_simh_config: DynamicSimheuristicConfig = {
#     "num_sims": 20,
#     "max_size_elite_set": 5,
#     "score_finding_new_elite": 1,
#     "score_finding_new_best": 2,
#     "init_score": 1,
#     "max_time_per_cp_solve": 30,
#     "consider_mean": True,
#     "quantiles": [0.6, 0.7, 0.8],
#     "frac_budget_before_sims": 10 / exp_config["time_limit"],
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
#     # frac_budget_final_elites_sim should be enough to sim. final
# }

simheuristics: list[SimheuristicSpec] = [
    # SimheuristicSpec("det_opt", deterministic_optimization, det_opt_conf),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config_1),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config_2),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config_3),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config_4),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config_5),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config_6),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config_7),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config_8),
    # SimheuristicSpec("std_simh", standard_simheuristic, stand_simh_config),
    # SimheuristicSpec("dyn_simh", dynamic_simheuristic, dyn_simh_config),
]
