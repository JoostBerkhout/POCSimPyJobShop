# Simheuristic for PyJobShop

This package provides code for solving stochastic PyJobShop scheduling problems
using simheuristics. It tries to follow the model and data separation paradigm 
from algebraic modeling languages. 

For each problem the user wants to solve with simheuristics, the user has to 
specify a problem `.py` script in the `problems` subpackage. This script should
contain a function that generates a concrete model based on data, and a 
data function that simulates problem data (of a type that can be used in the
model function to make a concrete model).

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
