import argparse
import os
from stable_baselines3 import PPO
from managers.enviroment_manager import SimulationEnv

parser = argparse.ArgumentParser(description="Train or load a PPO agent.")
parser.add_argument('--path_name', type=str, default="ppo",
                    help="Path name for the Simulation Environment (default: 'ppo')")
args = parser.parse_args()

MODEL_PATH = "simulation_policy.zip"

# Create the environment with the provided path_name
env = SimulationEnv(path_name=args.path_name, num_inferences=100)

if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("No saved model found. Training a new one...")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256)

# Train the agent
model.learn(total_timesteps=5120)

# Save the trained model
model.save("simulation_policy")