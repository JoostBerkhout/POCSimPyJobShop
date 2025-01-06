import pandas as pd

from pyjobshop.simheuristic.methods.run_dcop import run_dcop
from pyjobshop.simheuristic.methods.run_adaptive import run_adaptive
from pyjobshop.simheuristic.methods.run_standard import run_standard
from pyjobshop.simheuristic.methods.configs import dcop_config, standard_config, adaptive_config

configs = [adaptive_config, dcop_config, standard_config]
summarized_data = []
output_file = "results/debug_results.csv"

# Run al methods with different time limits
for seed in [20]:
    print(f'\nStart with seed {seed}')
    for time_limit in [120]:
        for config in configs:
            config["time_limit"] = time_limit
            config['seed'] = seed
            config["num_sims_long"] = 100
            if config["method"] == "dcop":
                data = run_dcop(dcop_config)
            elif config["method"] == "standard":
                data = run_standard(config)
            elif config["method"] == "adaptive":
                data = run_adaptive(config)
            else:
                raise NotImplementedError

            print(f'data {data}')

            summarized_data = summarized_data + data
            summarized_data_df = pd.DataFrame(summarized_data)
            summarized_data_df.to_csv(output_file)

