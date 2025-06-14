#!/bin/bash

# List of problem class names
PROBLEMS=(
    OpenShop
    OpenShopFullStoch
    OpenShopNoTard
    OpenShopFullStochNoTard
    FlexibleJobShop
    FlexibleJobShopFullStoch
    FlexibleJobShopNoTard
    FlexibleJobShopFullStochNoTard
    HybridFlowShop
    HybridFlowShopFullStoch
    HybridFlowShopNoTard
    HybridFlowShopFullStochNoTard
    JobShop
    JobShopFullStoch
    JobShopNoTard
    JobShopFullStochNoTard
    ParallelMachines
    ParallelMachinesFullStoch
    ParallelMachinesNoTard
    ParallelMachinesFullStochNoTard
)

# Loop over each problem and submit the job
for PROBLEM in "${PROBLEMS[@]}"; do
    echo "Submitting job for $PROBLEM..."
    uv run experiments/submit_slurm_job.py --problem "$PROBLEM" --mock true
done
