#!/bin/bash
#SBATCH --job-name=ppo_test_highclip  
#SBATCH --account=project_2012457
#SBATCH --output=ppo_output.log        # Standard output log
#SBATCH --error=ppo_error.log          # Error log
#SBATCH --partition=small              # Partition (adjust if needed)
#SBATCH --nodes=4                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=32             # CPU cores per task
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --time=24:00:00                # Max execution time (hh:mm:ss)

# Load necessary modules
module load python-data

# Activate virtual environment
source ~/venvs/rl_env/bin/activate  

# Run PPO training script with custom parameters
python train.py \
    --path_name "ppo_test_highclip" \
    --model_path "simulation_policy.zip" \
    --config_dir "profile/config_profile" \
    --n_envs 4 \
    --total_timesteps 5120 \
    --learning_rate 3e-4 \
    --n_steps 128 \
    --batch_size 64 \
    --clip_range 0.4 \
    --n_epochs 10 \
    --net_arch "[256, 256]"
