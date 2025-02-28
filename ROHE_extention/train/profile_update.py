import os
import sys
from stable_baselines3 import PPO
import time
import logging

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)
from simulation.profile_manager import ProfileManager



        
sim_config = {
    "config_path": "config/sim_config.yaml",
    "profile_path": "data/model_profile/model_profile.yaml",
    "label_path": "data/raw_sample/file_label.csv",
    "model_record_path": "data/record",
    "output_path": "results/eemls_inferences/"
}

profile_manager = ProfileManager(sim_config)
profile_manager.run()

while True:
    logging.debug("Update profile")
    time.sleep(10)