# POCSimPyJobShop

POCSimPyJobShop is a Python library for solving stochastic scheduling problems with constraint programming, using PyJobShop and simulation. This proof-of-concept (POC) implementation is still in an early stage of development. This repository contains the code used to produce the numerical results and analysis for the Winter Simulation Conference 2026 submission. For an updated version of the code, with syntax aligned with the submission example, please refer to https://github.com/JoostBerkhout/SimPyJobShop.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To get started:

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.
2. Clone the repository and install all dependencies:
   ```bash
   git clone https://github.com/JoostBerkhout/POCSimPyJobShop.git
   cd POCSimPyJobShop
   uv sync
   ```
3. Run any script or notebook with `uv run`, e.g.:
   ```bash
   uv run pytest
   ```

## Experiment data

The experiment result CSV files are available on Zenodo:
[https://doi.org/10.5281/zenodo.19662639](https://doi.org/10.5281/zenodo.19662639).
Download and extract them into the `experiments/results/` folder to reproduce
the analyses in the notebooks.

## Structure of code

The codebase is currently organized into the following main components:
- `simpyjobshop`: Contains the main library code for SimPyJobShop:
  - Root contains the main library code, which is used to solve stochastic scheduling problems.
  - `problems`: Contains the stochastic scheduling problems to be solved.
  - `simheuristics`: Contains the simheuristics for solving the stochastic scheduling problems.
- `pyjobshop`: Contains the PyJobShop library code, which is used as a dependency for SimPyJobShop.
- `experiments`: Contains scripts for running experiments and generating results:
  - Root contains the main scripts for running experiments locally and on the cluster. Detailed in the following section. 
  - `configs`: Contains configuration files for the experiments.
  - `results`: Contains the results of the experiments.
    - `notebooks`: Contains Jupyter notebooks for analyzing the results of the experiments:
      - `all_results_analyzer.ipynb`: Analyzes all the results of the experiments and generates plots and tables for the paper. It combines all the results from the following notebooks that analyzed different parameter settings for different approaches:
        - `param_tuning_det_opt.ipynb`: Analyzes the results of the parameter tuning for the deterministic optimization.
        - `param_tuning_sim_last.ipynb`: Analyzes the results of the approach that simulates the best solutions found.
        - `param_tuning_std_simh.ipynb`: Analyzes the results of the standard simheuristic.
        - `param_tuning_dyn_simh.ipynb`: Analyzes the results of the dynamic simheuristic.
  - `utils`: Contains utility functions for the experiments.
- `tests`: Contains unit tests for the library. It uses the same structure as the `simpyjobshop` directory, with each module having a corresponding test module.

## Details about `simpyjobshop`

The following details apply to this repo. Note that the new version uses a different setup as described in https://github.com/JoostBerkhout/SimPyJobShop.

This package provides code for solving stochastic PyJobShop scheduling problems
using simheuristics. By fixing solutions in PyJobShop, it allows to "simulate"
solutions for randomly generated data.

For each stochastic PyJobShop problem the user wants to solve with
simheuristics, the user has to make a class in subfolder `problems` that
inherits `Problem`. The user has to overwrite:

- `Problem.concrete_model()` with the concrete model for concrete `data`.
- `Problem.distribution_data()` that returns the distribution and constant data.

It tries to follow the model and data separation paradigm
from algebraic modeling languages. See the `problems` folder for examples.

## Details about experiments

- To replicate an experiment on Snellius: look at the date-and-time stamp of the results .csv file. Then go back to the latest commit before that date-and-time stamp. That gives all the code and settings used for that particular experiment.
- If there is a folder in experiments/configs folder with a specific problem name, then that folder contains the configuration files for that problem. The configuration files are used to run the experiments on Snellius.
- Steps on Snellius to run experiments for `SomeProblemName`:
    1. Go to the right folder with the root of the repository.
    2. `git pull` to get the latest code.
    3. Run `uv run experiments/submit_slurm_job.py --problem SomeProblemName --mock true` to check whether the instructions seem to be in order.
    4. Run `uv run experiments/submit_slurm_job.py --problem SomeProblemName` to submit the job to the cluster.
    5. Wait for the job to finish. You can check the status of the job with `squeue -u <your_username>`.
    6. Once the job is finished, you can find the results in the `experiments/results` folder. The results are stored in a .csv file with the name of the problem and the date-and-time stamp of when the experiment was run.
- As an alternative to `uv run` individually for each problem, you can also run `bash experiments/batch_submit_slurm_job.sh` to submit multiple problems at once.
- An inconsistency slipped into the experiments that does not have severe consequences but is worth mentioning: For the experiments w/ tardiness, the duration means are uniformly drawn from {0, 1, ..., 14}. In contrast, for the experiments w/o tardiness, the duration means are uniformly drawn from {1, 2, ..., 15}. The latter is more consistent as it generates a Poisson distribution with mean > 0, whereas the first generates Poisson distributions with mean = 0, leading to deterministic durations. Since it only affects around 7% of the task durations, it does not have a severe impact on the results. However, for future experiments, it is recommended to use the latter approach for consistency.