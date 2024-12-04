# Simheuristic for PyJobShop

This package provide code for a simheuristic approach in combination with
PyJobShop. It tries to follow the model and data separation paradigm as
used in algebraic modeling languages. The user can specify the scheduling model 
in `modeling.model_builder` and the corresponding (simulation) data in 
`DataGeneratorBuilder`. The scheduling model `modeling.model_builder` will
then load the data via a `DataGenerator` build via `DataGeneratorBuilder`. 
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
