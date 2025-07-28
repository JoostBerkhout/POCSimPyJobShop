# SimPyJobShop

SimPyJobShop is a Python library for solving stochastic scheduling problems with constraint programming making use of PyJobShop and simulation.

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
  - `notebooks`: Contains Jupyter notebooks for analyzing the results of the experiments.
  - `utils`: Contains utility functions for the experiments.
- `tests`: Contains unit tests for the library. It uses the same structure as the `simpyjobshop` directory, with each module having a corresponding test module.

## Details about `simpyjobshop`

This package provides code for solving stochastic PyJobShop scheduling problems
using simheuristics. By fixing solutions in PyJobShop, it allows to "simulate"
solutions for randomly generated data.

For each stochastic PyJobShop problem the user wants to solve with
simheuristics, the user has to make a class in subfolder `problems` that
inherits `Problem`. The user has to overwrite:

- `Problem.conrete_model()` with the concrete model for concrete `data`.
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

# README.md file for PyJobShop below

![PyJobShop logo](docs/source/assets/images/logo.svg)

[![PyPI](https://img.shields.io/pypi/v/PyJobShop?style=flat-square)](https://pypi.org/project/pyjobshop/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](https://github.com/PyJobShop/PyJobShop/)
[![CI](https://img.shields.io/github/actions/workflow/status/PyJobShop/PyJobShop/.github%2Fworkflows%2FCI.yml?style=flat-square)](https://github.com/PyJobShop/PyJobShop/)
[![DOC](https://img.shields.io/readthedocs/pyjobshop?style=flat-square)](https://pyjobshop.org)
[![Codecov](https://img.shields.io/codecov/c/github/PyJobShop/PyJobShop?style=flat-square)](https://app.codecov.io/gh/PyJobShop/PyJobShop/)

PyJobShop is a Python library for solving scheduling problems with constraint programming.
It currently supports the following scheduling problems:

- **Resource environments:** single machines, parallel machines, hybrid flow shops, open shops, job shops, flexible job shops, renewable resources and non-renewable resources.
- **Constraints:** release dates, deadlines, due dates, multiple modes, sequence-dependent setup times, no-wait, blocking, and arbitrary precedence constraints.
- **Objective functions:** minimizing makespan, total flow time, number of tardy jobs, total tardiness, total earliness, maximum tardiness and maximum lateness.

You can find PyJobShop on the Python Package Index under the name `pyjobshop`. 
To install it, simply run:

``` shell
pip install pyjobshop
```

The documentation is available [here](https://pyjobshop.org/).

> [!TIP]
> If you are new to scheduling or constraint programming, you might benefit from first reading the [introduction to scheduling](https://pyjobshop.org/stable/setup/intro_to_scheduling.html) and [introduction to constraint programming](https://pyjobshop.org/stable/setup/intro_to_cp.html) pages.

## Constraint programming solvers
PyJobShop uses [OR-Tools'](https://github.com/google/or-tools) CP-SAT solver as its default constraint programming solver.
We also provide support for CP Optimizer. 
See [our documentation](https://pyjobshop.org/stable/setup/installation.html) for instructions on how to install PyJobShop with CP Optimizer.

## Examples
We provide example notebooks that show how PyJobShop may be used to solve scheduling problems.

- A short tutorial and introduction to PyJobShop's modeling interface, available [here](https://pyjobshop.org/stable/examples/simple_example.html). This is a great way to get started with PyJobShop.
- Notebooks solving the classical machine scheduling problems such as the hybrid flow shop ([here](https://pyjobshop.org/stable/examples/hybrid_flow_shop.html)) and the flexible job shop problem ([here](https://pyjobshop.org/stable/examples/flexible_job_shop.html)).
- A notebook showing how to solve different project scheduling problems, [here](https://pyjobshop.org/stable/examples/project_scheduling.html).

## Contributing
We are very grateful for any contributions you are willing to make. 
Please have a look [here](https://pyjobshop.org/stable/dev/contributing.html) to get started. 
If you aim to make a large change, it is helpful to discuss the change first in a new GitHub issue. Feel free to open one!

## Getting help
Feel free to open an issue or a new discussion thread here on GitHub.
Please do not e-mail us with questions, modeling issues, or code examples.
Those are much easier to discuss via GitHub than over e-mail.
When writing your issue or discussion, please follow the instructions [here](https://pyjobshop.org/stable/setup/getting_help.html).
