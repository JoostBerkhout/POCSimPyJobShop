import importlib
from typing import Any


def load_module_or_default(module_path: str, fallback_module: str) -> Any:
    """
    Import a module by path, falling back to a default if not found.

    Parameters
    ----------
    module_path : str
        Full dotted path of the preferred module to import.
    fallback_module : str
        Full dotted path of the fallback module to use if the preferred
        module cannot be imported.

    Returns
    -------
    Any
        The imported module object.
    """
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError:
        return importlib.import_module(fallback_module)


def get_configs(problem_name: str) -> tuple[dict, list]:
    """
    Load experiment and simheuristic configuration for a problem.

    This function attempts to load problem-specific experiment and simheuristic
    configurations. If not found, it falls back to default base configurations.

    Parameters
    ----------
    problem_name : str
        Name of the problem (must match the folder/module name in
        `experiments/configs/`).

    Returns
    -------
    tuple of (dict, list)
        - exp_config : dict
            Dictionary with experiment-level configuration.
        - simheuristics : list
            List of Simheuristic instances or configuration dicts for the
            problem's simheuristics.
    """
    base_path = "experiments.configs"
    problem_path = f"{base_path}.{problem_name}"

    exp_module = load_module_or_default(
        f"{problem_path}.exp_config", f"{base_path}.base_exp_config"
    )
    simh_module = load_module_or_default(
        f"{problem_path}.simh_configs", f"{base_path}.base_simh_configs"
    )

    return exp_module.exp_config, simh_module.simheuristics
