import argparse
import os
import time
import shutil
import ast

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from managers.enviroment_manager import SimulationEnv

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
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for each PPO update")
parser.add_argument("--net_arch", type=str, default="[256, 256]", help="Network architecture as a list or dict, e.g. '[64, 64]' or '[{\"pi\": [64], \"vf\": [64]}]'")

args = parser.parse_args()

# Convert net_arch string to Python object
try:
    net_arch = ast.literal_eval(args.net_arch)
except Exception as e:
    raise ValueError(f"Invalid net_arch format: {args.net_arch}\nError: {e}")

# Clean config directory
def clean_dir(config_dir):
    if os.path.exists(config_dir):
        shutil.rmtree(config_dir)
        print("Directory cleaned")
    os.makedirs(config_dir, exist_ok=True)

clean_dir(f"{args.config_dir}_{args.path_name}")

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
ind = 0
def create_env():
    global ind
    ind += 1
    return SimulationEnv(path_name=args.path_name, env_index=ind, num_inferences=100, parallel_workers=1)

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
    n_epochs=args.n_epochs,
    policy_kwargs=dict(net_arch=net_arch),
    verbose=1,
    tensorboard_log=log_dir
)

# Train the model
start_time = time.time()
model.learn(total_timesteps=args.total_timesteps * args.n_envs)
end_time = time.time()

print(f"Training time: {end_time - start_time:.2f} seconds")
