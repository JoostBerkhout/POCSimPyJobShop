exp_config = {
    "time_limit": 5,
    "num_workers": 1,  # (per instance)
    "use_wandb": True,
    # only relevant for cli_run_experiments.py and submit_slurm_job.py:
    "num_rand_experiments": 5,
    "num_parallel_instances": 5,
    "num_sims_for_true_expec_objective": 10,
    }
