import logging
import numpy as np
import os
import sys
from rohe.common import rohe_utils
import pandas as pd
import random


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)
from simulation.rohe_base_model import ServiceConfig, ProcessingService, ServiceType, InferenceService, EnsembleService


class EEMLSSimulation():
    def __init__(self, sim_config):
        self.config_path = os.path.join(parent_dir, sim_config["config_path"])
        self.profile_path = os.path.join(parent_dir, sim_config["profile_path"])
        self.label_path = os.path.join(parent_dir, sim_config["label_path"])
        self.model_record_path = os.path.join(parent_dir, sim_config["model_record_path"])
        self.output_path = os.path.join(parent_dir, sim_config["output_path"])
        self.throughput_requirement = 15
        self.num_models = sim_config["num_models"]
        self.max_models = sim_config["max_models"]
        self.min_models = sim_config["min_models"]
        self.step_index = 0
        
        self.model_profile_data = rohe_utils.load_config(self.profile_path)
        self.label_data = pd.read_csv(self.label_path).groupby("label")
        self.labels = self.label_data.groups.keys()
        
        model_names = list(self.model_profile_data.keys())
        if "ensemble" in model_names:
            model_names.remove("ensemble")
        self.model_names = ["ensemble"] + model_names
        self.model_profile_rt = {}
        
        self.sim_config = None
        self.distribution_keys = None
        self.distribution_weights = None
        self.processing_service = None
        self.ensemble_service = None
        self.total_energy_consumption = 0
        self.structured_state = {}
        self.flatten_state = []
        self.state_length = self.get_state_length()
    
    def get_state_length(self):
        state = self.step_inference()
        flat_state = self.flatten_structured_state(state)
        return len(flat_state)
        
    def init_pipeline(self):
        """Configure the pipeline based on the simulation config."""
        # Load the simulation config from the model rl 
        self.sim_config = rohe_utils.load_config(self.config_path)

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
                model_config["throughput"] = self.model_profile_data[model_name]["throughput"]
                model_config["energy"] = self.model_profile_data[model_name]["energy"]
                model_config["response_time"] = self.model_profile_data[model_name]["response_time"]
                model_config["service_type"] = ServiceType.INFERENCE.value
                model_config["data_path"] = str(self.model_record_path)
                model_config["throughput_requirement"] = int(self.sim_config["throughput_requirement"])
                inference_service_config = ServiceConfig.model_validate(model_config)
                inference_service = InferenceService(inference_service_config)
                self.ensemble_service.add_model(inference_service)
    
    def inference(self, num_inferences=100):
        self.step_index += 1
        # init default data
        data = {
            "input": {
                "file_name": "n01560419_3101",
                "image_height": 224,
                "image_width": 224,
            }
        }
        # Select the input image
        selected_key = random.choices(self.distribution_keys, weights=self.distribution_weights, k=1)[0]
        if selected_key in self.labels:
            df_file = self.label_data.get_group(selected_key)
            file_name = random.choice(df_file["file_name"].values)
        data["input"]["file_name"] = file_name
        data["label"] = selected_key
        
        data = self.processing_service.execute(data)
        data = self.ensemble_service.execute(data)
        
        # Update model profiles
        for model_name, inferences in data["ml_inference"].items():
            
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
            if model_name not in self.model_profile_rt:
                self.model_profile_rt[model_name] = {"data_frame": result_df}
            else:
                self.model_profile_rt[model_name]["data_frame"] = pd.concat(
                    [self.model_profile_rt[model_name]["data_frame"], result_df], ignore_index=True
                )


            os.makedirs(self.output_path, exist_ok=True)

            # Save the last 10,000 rows to CSV
            if (self.step_index % num_inferences == 0):
                self.model_profile_rt[model_name]["data_frame"].tail(num_inferences).to_csv(
                    f"{self.output_path}{model_name}_inference.csv")
        
        return data
        
    
    def step_inference(self, num_inferences=100):
        self.init_pipeline()
        self.step_index = 0
        for i in range(num_inferences):
            data = self.inference(num_inferences)
            
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
        
        
        for i in range(self.max_models+1):
            if i < len(self.ensemble_service.ensemble.keys()):
                model_name = list(self.ensemble_service.ensemble.keys())[i]
            elif i == len(self.ensemble_service.ensemble.keys()):
                model_name = "ensemble"
            else:
                model_name = f"empty_{i}"
                
            if (model_name in self.model_profile_rt and model_name in self.ensemble_service.ensemble.keys()) or model_name == "ensemble":
                
                model_data = self.model_profile_rt[model_name]

                if "data_frame" in model_data:
                    recent_df = model_data["data_frame"].tail(num_inferences)

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
        self.structured_state = structured_state
        print("STATE: ", structured_state)
        return structured_state
    
    def flatten_structured_state(self, structured_state=None):
        if structured_state is None:
            structured_state = self.structured_state
        ensemble_state_vector = [
            float(structured_state["ensemble_state"].get("total_energy_consumption", 0.0)),
            float(structured_state["ensemble_state"].get("ensemble_size", 0))
        ]

        # Include distribution weights if available
        #if hasattr(self, "distribution_weights") and isinstance(self.distribution_weights, (list, np.ndarray)):
        #    ensemble_state_vector.extend(map(np.float32, self.distribution_weights))

        # Flatten model states
        model_states_vector = []
        # print(structured_state["model_states"])
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
            np.float32(structured_state["input_state"].get("image_height", 0)),
            np.float32(structured_state["input_state"].get("image_width", 0))
        ]

        distribution_weights = structured_state["distribution_weights"]
        if distribution_weights:
            ensemble_state_vector.extend(distribution_weights)

        # Combine all parts into a single flattened array
        flattened_state = np.concatenate([
            np.array(ensemble_state_vector, dtype=np.float32),
            np.array(input_state_vector, dtype=np.float32),
            np.array(model_states_vector, dtype=np.float32)
        ])
        return flattened_state