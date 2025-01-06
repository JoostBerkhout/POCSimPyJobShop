import pandas as pd

from pyjobshop.simheuristic.methods.run_dcop import run_dcop
from pyjobshop.simheuristic.methods.run_adaptive import run_adaptive
from pyjobshop.simheuristic.methods.run_standard import run_standard
from pyjobshop.simheuristic.methods.configs import dcop_config, standard_config, adaptive_config

summarized_data = []
output_file = "results/new_results_adaptive.csv"

config = adaptive_config

# Run al methods with different time limits
for seed in [200, 120, 6, 53, 13]:
    print(f'\nStart with seed {seed}')
    for time_limit in [300, 600, 1800]:
        for strategy in ["adaptive", "mean"]:
            if strategy == "mean":
                config["consider_mean"] = int(True)
                config["strategies"] = []
            elif strategy == "adaptive":
                config["consider_mean"] = int(True)
                config["strategies"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            else:
                config["consider_mean"] = int(False)
                config["strategies"] = [strategy]

            config["time_limit"] = time_limit
            config['seed'] = seed
            data = run_adaptive(config)

            data[0]["strategy"] = strategy
            print(f'data')
            summarized_data = summarized_data + data
            summarized_data_df = pd.DataFrame(summarized_data)
            summarized_data_df.to_csv(output_file)