from experiments.configs.base_exp_config import exp_config
from experiments.utils.SimheuristicSpec import SimheuristicSpec
from simpyjobshop.simheuristics import (
    DynamicSimheuristicConfig,
    SimulateLastSolutionsConfig,
    StandardSimheuristicConfig,
    deterministic_optimization,
    dynamic_simheuristic,
    simulate_last_solutions,
    standard_simheuristic,
)

# Simheuristic configurations
stand_simh_config: StandardSimheuristicConfig = {
    "num_sims": 20,
    "max_size_elite_set": 5,
    "frac_budget_before_sims": 10 / exp_config["time_limit"],
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}
dyn_simh_config: DynamicSimheuristicConfig = {
    "num_sims": 20,
    "max_size_elite_set": 5,
    "score_finding_new_elite": 1,
    "score_finding_new_best": 2,
    "init_score": 1,
    "max_time_per_cp_solve": 30,
    "consider_mean": True,
    "quantiles": [0.6, 0.7, 0.8],
    "frac_budget_before_sims": 10 / exp_config["time_limit"],
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
    # frac_budget_final_elites_sim should be enough to sim. final
}
sim_last_config: SimulateLastSolutionsConfig = {
    "max_size_elite_set": 5,
    "frac_budget_final_elites_sim": 30 / exp_config["time_limit"],
}

simheuristics: list[SimheuristicSpec] = [
    SimheuristicSpec("det_opt", deterministic_optimization, {}),
    SimheuristicSpec("sim_last", simulate_last_solutions, sim_last_config),
    SimheuristicSpec("std_simh", standard_simheuristic, stand_simh_config),
    SimheuristicSpec("dyn_simh", dynamic_simheuristic, dyn_simh_config),
]
