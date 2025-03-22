from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from managers.enviroment_manager import SimulationEnv
import argparse
import os
import torch
import time

parser = argparse.ArgumentParser(description="Train or load a PPO agent.")
parser.add_argument('--path_name', type=str, default="ppo",
                    help="Path name for the Simulation Environment (default: 'ppo1')")
args = parser.parse_args()

MODEL_PATH = "simulation_policy.zip"

# Create 4 parallel instances
def create_env():
    return SimulationEnv(path_name=args.path_name, num_inferences=100, parallel_workers=4)

env = make_vec_env(create_env, n_envs=4)

# Load an existing model or create a new one
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("No saved model found. Training a new one...")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256)

start_time = time.time()

# Training agent
model.learn(total_timesteps=2560)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time:.2f} seconds")

# Save the trained model
# model.save("simulation_policy.zip")
