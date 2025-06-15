#!/bin/bash

# List of problem class names
PROBLEMS=(
    OpenShopNoTard
    OpenShopFullStochNoTard
    FlexibleJobShopNoTard
    FlexibleJobShopFullStochNoTard
    HybridFlowShopNoTard
    HybridFlowShopFullStochNoTard
    JobShopNoTard
    JobShopFullStochNoTard
    ParallelMachinesNoTard
    ParallelMachinesFullStochNoTard
)

# Loop over each problem and submit the job
for PROBLEM in "${PROBLEMS[@]}"; do
    echo "Submitting job for $PROBLEM..."
    uv run experiments/submit_slurm_job.py --problem "$PROBLEM"
done
