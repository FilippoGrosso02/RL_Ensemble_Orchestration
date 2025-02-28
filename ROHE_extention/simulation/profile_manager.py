from rohe.common import rohe_utils
import pandas as pd
from threading import Timer
import random
import yaml
import os
import logging
NUM_INF = 1000
DEFAULT_INTERVAL = 10

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)



class ProfileManager():
    def __init__(self, config):
        self.sim_config_path = config["config_path"]
        self.model_record_path = config["model_record_path"]
        self.profile_path = config["profile_path"]
        self.label_path = config["label_path"]
        self.data_label = pd.read_csv(os.path.join(parent_dir, self.label_path)).groupby("label")
        self.labels = self.data_label.groups.keys()
        self.model_profile = rohe_utils.load_config(os.path.join(parent_dir, self.profile_path))
        self.sim_config = {}
        self.all_model_names = self.model_profile.keys()
        self.model_data = {}
        for model_name in self.all_model_names:
            self.model_data[model_name] = pd.read_csv(os.path.join(parent_dir, self.model_record_path +"/"+ model_name + ".csv"))
        self.update_flag = False
    
    def update_profile(self):
        self.sim_config = rohe_utils.load_config(os.path.join(parent_dir, self.sim_config_path))
        distribution = self.sim_config["distribution"]
        distribution_keys = list(distribution.keys())
        distribution_weights = list(distribution.values())
        
        # random list of 1000 inference keys following the distribution
        inference_keys = random.choices(distribution_keys, distribution_weights, k=NUM_INF)
        inf_files = []
        for key in inference_keys:
            df_file = self.data_label.get_group(key)
            file_name = random.choice(df_file["file_name"].values)
            inf_files.append(file_name)
        for model_name, model_df in self.model_data.items():
            # only select row has file_name in inf_files
            model_df = model_df[model_df["file_name"].isin(inf_files)]
            # create a new column for model_df following condition: if "label" is in str of "inference" column then 1 else 0
            model_df["accuracy"] = model_df.apply(lambda x: 1 if str(x["label"]) in str(x["inference"]) else 0, axis=1)
            # calculate accuracy
            accuracy = float(model_df["accuracy"].mean())
            # calculate min response time
            min_response_time = float(model_df["response_time"].min())
            # calculate max response time
            max_response_time = float(model_df["response_time"].max())
            # calculate avg response time
            avg_response_time = float(model_df["response_time"].mean())
            # update to model profile
            self.model_profile[model_name]["overall_accuracy"] = accuracy
            self.model_profile[model_name]["response_time"]["cuda"]["max"] = max_response_time
            self.model_profile[model_name]["response_time"]["cuda"]["min"] = min_response_time
            self.model_profile[model_name]["response_time"]["cuda"]["avg"] = avg_response_time
        self.save_yaml(os.path.join(parent_dir, self.profile_path), self.model_profile)
        logging.debug("Update profile for model")
        if self.update_flag:
            Timer(DEFAULT_INTERVAL, self.update_profile).start()
        
        
    def save_yaml(self, path: str, data: dict):
        """Save YAML data to the given path."""
        try:
            with open(path, 'w') as file:
                yaml.safe_dump(data, file, default_flow_style=False)
        except Exception as e:
            logging.debug(f"Error saving YAML file: {e}")
    
    def run(self):
        self.update_flag = True
        self.update_profile()