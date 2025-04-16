from multiprocessing import cpu_count


def check_cpu_use(
    num_parallel_instances: int,
    num_workers_per_instance: int,
    num_cpus: int | None = None,
):
    """
    Checks that parallel instances and their workers fit within CPU limits.
    """
    total_workers = num_parallel_instances * num_workers_per_instance
    num_cpus = cpu_count() if num_cpus is None else num_cpus
    msg = "Too many parallel instances and workers for available CPU cores."
    assert total_workers <= num_cpus, msg
