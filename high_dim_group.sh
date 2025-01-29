#!/bin/bash
# How to use in bash: 1. chmod +x high_dim_group.sh 2. ./high_dim_group.sh
# device: A800 80G

# Group 1 - Split into two tasks
echo "Running Group 1 - Task 1..."
python results_full_history/Grad_Dependent_Nonlinear/60d/experiment_run.py &
wait
echo "Group 1 - Task 1 completed."

echo "Running Group 1 - Task 2..."
python results/Grad_Dependent_Nonlinear/60d/experiment_run.py &
wait
echo "Group 1 - Task 2 completed."

# Group 2 - Split into two tasks
echo "Running Group 2 - Task 1..."
python results/Grad_Dependent_Nonlinear/80d/experiment_run.py &
wait
echo "Group 2 - Task 1 completed."

echo "Running Group 2 - Task 2..."
python results_full_history/Linear_HJB/60d/experiment_run.py &
wait
echo "Group 2 - Task 2 completed."

# Group 3 - Split into two tasks
echo "Running Group 3 - Task 1..."
python results_full_history/Grad_Dependent_Nonlinear/80d/experiment_run.py &
wait
echo "Group 3 - Task 1 completed."

echo "Running Group 3 - Task 2..."
python results/Linear_HJB/60d/experiment_run.py &
wait
echo "Group 3 - Task 2 completed."

# Group 4 - Split into two tasks
echo "Running Group 4 - Task 1..."
python results/Linear_HJB/80d/experiment_run.py &
wait
echo "Group 4 - Task 1 completed."

echo "Running Group 4 - Task 2..."
python results_full_history/Linear_HJB/80d/experiment_run.py &
wait
echo "Group 4 - Task 2 completed."
