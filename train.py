import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from managers.enviroment_manager import SimulationEnv
import os
import time
import shutil

# Argument parser for SLURM parameters
parser = argparse.ArgumentParser(description="Train PPO model with SLURM")
parser.add_argument('--path_name', type=str, default="ppo", help="Path name for the Simulation Environment (default: 'ppo1')")
parser.add_argument("--model_path", type=str, default="simulation_policy.zip", help="Path to save the trained model")
parser.add_argument("--config_dir", type=str, default="profile/config_profile", help="Directory for config files")
parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--total_timesteps", type=int, default=2560, help="Total training timesteps")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--n_steps", type=int, default=128, help="Number of steps per update")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping range for PPO updates")

args = parser.parse_args()

# Clean config directory
def clean_dir(config_dir):
    if os.path.exists(config_dir):
        shutil.rmtree(config_dir)
        print("Directory cleaned")
    os.makedirs(config_dir, exist_ok=True)

clean_dir(args.config_dir)

# Log the argument values to a file
result_dir = os.path.join("results", args.path_name)
os.makedirs(result_dir, exist_ok=True)
def log_args(args, config_dir):
    log_file = os.path.join(config_dir, "arguments.txt")
    with open(log_file, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Arguments logged to {log_file}")

log_args(args, result_dir)

# Create environment
ind = 0 # The index of the enviroment
def create_env():
    global ind
    ind += 1  # Increment the index
    return SimulationEnv(path_name=args.path_name, env_index= ind, num_inferences=100, parallel_workers=1)
       
env = make_vec_env(create_env, n_envs=args.n_envs)

# Create the Tensorboard directory
log_dir = "logs"

# Initialize PPO model
model = PPO(
    "MlpPolicy", env,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    clip_range=args.clip_range,
    verbose=1,
    tensorboard_log=log_dir
)

start_time = time.time()
model.learn(total_timesteps=args.total_timesteps * args.n_envs)
end_time = time.time()

print(f"Training time: {end_time - start_time:.2f} seconds")


