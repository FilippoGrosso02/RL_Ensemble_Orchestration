#!/bin/bash
#SBATCH --job-name=test_rl_job_parallel  
#SBATCH --account=project_2012457
#SBATCH --output=ppo_output.log        # Standard output log
#SBATCH --error=ppo_error.log          # Error log
#SBATCH --partition=small            # Change this to your HPC partition
#SBATCH --nodes=4                     # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=16           # CPU cores per task
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --time=24:00:00                # Max execution time (hh:mm:ss)

# Load necessary modules (Modify as needed)

module load python-data

# Activate virtual environment (Modify if needed)
source ~/venvs/rl_env/bin/activate  

# Run PPO training script with SLURM parameters
python train_ppo.py \
    --path_name "ppo_test2" \
    --model_path "simulation_policy.zip" \
    --config_dir "profile/config_profile" \
    --n_envs 4 \
    --total_timesteps 100000 \
    --learning_rate 3e-4 \
    --n_steps 2048 \
    --batch_size 64 \
    --clip_range 0.2
