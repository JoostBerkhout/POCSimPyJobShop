# Simheuristic for PyJobShop

This package provides code for solving stochastic PyJobShop scheduling problems
using simheuristics. By fixing solutions in PyJobShop, it allows to "simulate"
solutions for randomly generated data. 

For each stochastic PyJobShop problem the user wants to solve with 
simheuristics, the user has to make a class in subfolder `problems` that
inherits `Problem`. The user has to overwrite:

- `Problem.conrete_model()` with the concrete model for concrete `data`.
- `Problem.distribution_data()` that returns the distribution and constant data.

It tries to follow the model and data separation paradigm
from algebraic modeling languages.

The script `test_simheuristic.py` is used to test the code and demonstrates
how it can be used for simulation and optimization.

## Discussed design with Leon Lan on 3-12-2024

```
data = ProblemData()
pi1 = solve(data)

for scenario in scenarios:
# Fix ordering of new solution
new_constraints = new_constraints(pi1, data)
constraints = data.constraints.deepcopy()
constraints |= new_costraints # based on solution ordering constraints

    # Define all new modes based on solution & sample
    modes = [Mode(task=1, duration=sample[idx]) for idx in range(data.num_tasks)]
    setup_times = ...

    new_data = data.replace(constraints=constraints, modes=modes, setup_times)
    solve(new_data)


# -> E[f(pi_1, X)]

```
