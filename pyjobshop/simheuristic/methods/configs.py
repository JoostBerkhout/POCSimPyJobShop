dcop_config: dict[str, int] = {
        "num_jobs": 20,
        "num_stages": 20,
        "num_sims": 25,
        "num_sims_long": 5000,
        "method": "dcop"
    }

adaptive_config: dict = {
    "num_jobs": 20,
    "num_stages": 20,
    "num_sims": 25,
    "num_sims_long": 5000,
    "max_size_elite_set": 5,
    "score_finding_new_elite": 1,
    "score_finding_new_best": 2,
    "init_score": 1,
    "max_time_per_cp_solve": 30,
    "enumerate": 0,
    "method": "adaptive",
    "consider_mean": int(True),
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

standard_config: dict = {
                "num_jobs": 20,
                "num_stages": 20,
                "num_sims": 25,
                "num_sims_long": 5000,
                "enumerate": 0,
                "method": "standard"
            }