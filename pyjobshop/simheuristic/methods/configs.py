dcop_config: dict[str, int] = {
        "num_jobs": 20,
        "num_stages": 20,
        "num_sims": 25,
        "num_sims_long": 5000,
        "method": "dcop"
    }

adaptive_config: dict[str, int] = {
    "num_jobs": 20,
    "num_stages": 20,
    "num_sims": 25,
    "num_sims_long": 5000,
    "max_size_elite_set": 5,
    "score_finding_new_elite": 1,
    "score_finding_new_best": 2,
    "init_score": 1,
    "max_time_per_cp_solve": 30,
    "consider_mean": int(True),
    "enumerate": 0,
    "method": "adaptive"
}

standard_config: dict[str, int] = {
                "num_jobs": 20,
                "num_stages": 20,
                "num_sims": 25,
                "num_sims_long": 5000,
                "enumerate": 0,
                "method": "standard"
            }