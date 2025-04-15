from multiprocessing import cpu_count


def check_cpu_use(num_parallel_instances: int, num_workers_per_instance: int):
    """
    Checks that parallel instances and their workers fit within CPU limits.
    """
    total_workers = num_parallel_instances * num_workers_per_instance
    assert (
        total_workers <= cpu_count()
    ), "Too many parallel instances and workers for available CPU cores."
