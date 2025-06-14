from experiments.utils.SimheuristicSpec import SimheuristicSpec
from simpyjobshop.simheuristics import (
    DeterministicOptimizationConfig,
    deterministic_optimization,
)

# Simheuristic configurations
det_opt_conf_1: DeterministicOptimizationConfig = {
    "det_repr": "mean",
}
det_opt_conf_2: DeterministicOptimizationConfig = {
    "det_repr": 0.6,
}
det_opt_conf_3: DeterministicOptimizationConfig = {
    "det_repr": 0.7,
}
det_opt_conf_4: DeterministicOptimizationConfig = {
    "det_repr": 0.8,
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
# sim_last_config: SimulateLastSolutionsConfig = {
#     "det_repr": "mean",
#     "max_size_elite_set": 5,
#     "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
# }

simheuristics: list[SimheuristicSpec] = [
    SimheuristicSpec("det_opt_1", deterministic_optimization, det_opt_conf_1),
    SimheuristicSpec("det_opt_2", deterministic_optimization, det_opt_conf_2),
    SimheuristicSpec("det_opt_3", deterministic_optimization, det_opt_conf_3),
    SimheuristicSpec("det_opt_4", deterministic_optimization, det_opt_conf_4),
    # SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config),
    # SimheuristicSpec("std_simh", standard_simheuristic, stand_simh_config),
    # SimheuristicSpec("dyn_simh", dynamic_simheuristic, dyn_simh_config),
]
