exp_config = {
    "time_limit": 300,
    "num_workers": 8,  # (per instance)
    "use_wandb": True,
    # only relevant for cli_run_experiments.py and submit_slurm_job.py:
    "num_rand_experiments": 50,
    "num_parallel_instances": 24,
    "num_sims_for_true_expec_objective": 1000,
}
