from managers.model_rl import ServiceConfig, ProcessingService, ServiceType, InferenceService, EnsembleService
from managers.config_manager import ConfigManager

import logging
import os
import time
import pandas as pd
import random
import copy
from collections import Counter
from rohe.common import rohe_utils as utils
from IPython.display import clear_output

import yaml
import numpy as np


class StateManager():
    def __init__(self, current_dir):
        # Initialize necessary variables
        
        self.current_dir = current_dir
        self.parent_dir = os.path.dirname(current_dir)
        self.CONFIG_PATH = os.path.join(current_dir, "sim_config.yaml")
        self.PROFILE_PATH = os.path.join(current_dir, "profile/model_profile/model_profile.yaml")
        self.FILE_DATA_PATH = os.path.join(current_dir, "file_label.csv")
        self.MODEL_DATA_PATH = os.path.join(current_dir, "profile/processed")
        self.FILE_DATA_PATH = os.path.join(current_dir, "file_label.csv")
        self.THROUGHPUT_REQUIREMENT = 15
        self.step_index = 0
        
        self.profile_data = utils.load_config(self.PROFILE_PATH)
        self.data_file_label = pd.read_csv(self.FILE_DATA_PATH).groupby("label")
        self.labels = list(self.data_file_label.groups.keys())
        self.model_profile_data = {}

        # Input data example
        self.init_data = {
            "input": {
                "file_name": "n01560419_3101",
                "image_height": 224,
                "image_width": 224,
            }
        }
        # Global variables
        self.sim_config = None
        self.distribution_keys = None
        self.distribution_weights = None
        self.processing_service = None
        self.ensemble_service = None
        self.total_energy_consumption = 0
        
        self.setup_config()
        
        self.state_lenght = self.setup_state()

        


    def setup_state(self):
        state = self.get_state()
        state_vector = self.flatten_structured_state(state)

        return len(state_vector)


    def setup_config(self):
        """
        Reloads the initial YAML configuration file and sets up data,
        including extracting model names and ensuring 'ensemble' is first.
        """
        # Reload the initial YAML configuration file
        initial_yaml_file_path = os.path.join(self.current_dir, "initial_sim_config.yaml")

        # Load contents from the initial YAML file
        with open(initial_yaml_file_path, "r") as initial_file:
            initial_yaml_content = yaml.safe_load(initial_file)

        # Write the loaded contents into the target YAML file
        with open(self.CONFIG_PATH, "w") as target_file:
            yaml.dump(initial_yaml_content, target_file, default_flow_style=False)

        print(f"Contents of {initial_yaml_file_path} have been dumped into {self.CONFIG_PATH}.")

        # Load necessary data
        self.profile_data = utils.load_config(self.PROFILE_PATH)
        self.data_file_label = pd.read_csv(self.FILE_DATA_PATH).groupby("label")
        self.labels = list(self.data_file_label.groups.keys())

        # Extract model names from profile_data and ensure 'ensemble' is first
        model_names = list(self.profile_data.keys())  # Extract model names
        if "ensemble" in model_names:
            model_names.remove("ensemble")  # Ensure 'ensemble' isn't duplicated
        self.model_names = ["ensemble"] + model_names  # Prepend 'ensemble'

        self.model_profile_data = {}

        print(f"Model order set: {self.model_names}")


    
    #def state_step():

    def config_pipeline(self):           
        """Configure the pipeline based on the simulation config."""
        # Load the simulation config from the model rl 
        self.sim_config = utils.load_config(self.CONFIG_PATH)

        # Input distribution
        distribution = self.sim_config["distribution"]
        self.distribution_keys = list(distribution.keys())
        self.distribution_weights = list(distribution.values())

        # Create processing service
        if "processing" in self.sim_config:
            processing_config = self.sim_config["processing"]
            processing_config["service_type"] = ServiceType.PROCESSING.value
            processing_service_config = ServiceConfig.model_validate(processing_config)
            self.processing_service = ProcessingService(processing_service_config)

        # Create ensemble service
        if "ensemble" in self.sim_config:
            ensemble_config = self.sim_config["ensemble"]
            ensemble_config["service_type"] = ServiceType.ENSEMBLE.value
            ensemble_service_config = ServiceConfig.model_validate(ensemble_config)
            self.ensemble_service = EnsembleService(ensemble_service_config)

        # Add inference services to ensemble
        if "inference" in self.sim_config:
            inference_configs = self.sim_config["inference"]
            for model_name, model_config in inference_configs.items():
                model_config["throughput"] = self.profile_data[model_name]["throughput"]
                model_config["energy"] = self.profile_data[model_name]["energy"]
                model_config["response_time"] = self.profile_data[model_name]["response_time"]
                model_config["service_type"] = ServiceType.INFERENCE.value
                model_config["data_path"] = str(self.MODEL_DATA_PATH)
                model_config["throughput_requirement"] = int(self.sim_config["throughput_requirement"])
                inference_service_config = ServiceConfig.model_validate(model_config)
                inference_service = InferenceService(inference_service_config)
                self.ensemble_service.add_model(inference_service)

    def random_data_metrics(self, num_samples = 100):
        # Perform the inference on a random data sampled from the distribution

        self.step_index

        model_count = 0
        # Prepare input data: 224-224
        data = copy.deepcopy(self.init_data)

        # Select the input image
        selected_key = random.choices(self.distribution_keys, weights=self.distribution_weights, k=1)[0]
        if selected_key in self.labels:
            df_file = self.data_file_label.get_group(selected_key)
            file_name = random.choice(df_file["file_name"].values)
        data["input"]["file_name"] = file_name
        data["label"] = selected_key

        # Execute services
        data = self.processing_service.execute(data)
        data = self.ensemble_service.execute(data)
        

        # Update model profiles
        for model_name, inferences in data["ml_inference"].items():
            model_count += 1
            response_time = data["response_time"]["inference"].get(model_name, 0)

            i_label = data["label"]
            i_accuracy = 1 if i_label in inferences else 0
            i_confidence = inferences.get(i_label, 0)
            model_contribution = data["contribution"].get(model_name, 0)

            # Save to profile data
            data_dict = {
                "label": [i_label],
                "accuracy": [i_accuracy],
                "confidence": [i_confidence],
                "response_time": [response_time],
                "contribution": [model_contribution],
            }
            result_df = pd.DataFrame(data_dict)
            if model_name not in self.model_profile_data:
                self.model_profile_data[model_name] = {"data_frame": result_df}
            else:
                self.model_profile_data[model_name]["data_frame"] = pd.concat(
                    [self.model_profile_data[model_name]["data_frame"], result_df], ignore_index=True
                )

            # Ensure the output directory exists
            output_dir = f"results/reinf_learning_inference/"
            os.makedirs(output_dir, exist_ok=True)

            # Save the last 10,000 rows to CSV
            if (self.step_index % num_samples == 0):
                self.model_profile_data[model_name]["data_frame"].tail(num_samples).to_csv(
                    f"{output_dir}{model_name}_inference.csv",     
                    

                )

        
        return data

    def get_state(self, num_samples=100):
        """
        Generates a structured state vector with a fixed model order.
        If a model is missing, its slot is replaced with a padding entry.
        """

        self.config_pipeline()

        for i in range(num_samples):
            data = self.random_data_metrics(num_samples)
        
        

        energy_report = self.ensemble_service.energy_estimate()
        ensemble_state = {
            "total_energy_consumption": self.total_energy_consumption + energy_report["ensemble"],
            "ensemble_size": len(self.ensemble_service.ensemble),
        }

        # Ensure self.model_order exists (should be set in setup_config)
        if not hasattr(self, "model_names"):
            raise AttributeError("self.model_order is not defined. Make sure to call setup_config() first.")

        # Model-level metrics (fixed order)
        model_states = {}

        for model_name in self.model_names:  # Iterate in locked order
            
            if (model_name in self.model_profile_data and model_name in self.ensemble_service.ensemble.keys()) or model_name == "ensemble":
                model_data = self.model_profile_data[model_name]

                if "data_frame" in model_data:
                    recent_df = model_data["data_frame"].tail(num_samples)

                    # Calculate metrics
                    accuracy = recent_df["accuracy"].mean()
                    confidence = recent_df["confidence"].mean()
                    avg_response_time = recent_df["response_time"].mean()
                    max_response_time = recent_df["response_time"].max()
                    contribution = recent_df["contribution"].mean()
                else:
                    # If model has no recorded data, use default values
                    accuracy = confidence = avg_response_time = max_response_time = contribution = 0.0
            else:
                # If model is missing, fill with padding values
                accuracy = confidence = avg_response_time = max_response_time = contribution = 0.0

            # Store model metrics in a fixed order
            model_states[model_name] = {
                "accuracy": accuracy,
                "confidence": confidence,
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "contribution": contribution,
            }

        # Input state (flattened input metrics)
        input_state = {
            "input_file_length": len(data["input"]["file_name"]),  # Example input length
            "image_height": data["input"]["image_height"],
            "image_width": data["input"]["image_width"],
        }

        structured_state = {
            "ensemble_state": ensemble_state,
            "model_states": model_states,
            "input_state": input_state
        }

        if hasattr(self, "distribution_weights") and isinstance(self.distribution_weights, (list, np.ndarray)):
            structured_state["distribution_weights"] = list(map(np.float32, self.distribution_weights))

        return structured_state

    def flatten_structured_state(self, structured_state):

        ensemble_state_vector = [
            float(structured_state["ensemble_state"].get("total_energy_consumption", 0.0)),
            float(structured_state["ensemble_state"].get("ensemble_size", 0))
        ]

        # Include distribution weights if available
        #if hasattr(self, "distribution_weights") and isinstance(self.distribution_weights, (list, np.ndarray)):
        #    ensemble_state_vector.extend(map(np.float32, self.distribution_weights))

        # Flatten model states
        model_states_vector = []
        for metrics in structured_state["model_states"].values():
            model_states_vector.extend([
                np.float32(metrics.get("accuracy", 0.0)),
                np.float32(metrics.get("confidence", 0.0)),
                np.float32(metrics.get("avg_response_time", 0.0)),
                np.float32(metrics.get("max_response_time", 0.0)),
                np.float32(metrics.get("contribution", 0.0))
            ])

        # Flatten input state
        input_state_vector = [
            np.float32(structured_state["input_state"].get("input_file_length", 0)),
            np.float32(structured_state["input_state"].get("image_height", 0)),
            np.float32(structured_state["input_state"].get("image_width", 0))
        ]

        distribution_weights = structured_state["distribution_weights"]
        if distribution_weights:
            ensemble_state_vector.extend(distribution_weights)

        # Combine all parts into a single flattened array
        flattened_state = np.concatenate([
            np.array(ensemble_state_vector, dtype=np.float32),
            np.array(model_states_vector, dtype=np.float32),
            np.array(input_state_vector, dtype=np.float32)
        ])

        return flattened_state


