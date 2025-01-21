from pydantic import BaseModel
from enum import Enum
from typing import Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import copy
import pandas as pd
import yaml

logging.basicConfig(
    format="%(asctime)s:%(levelname)s -- %(message)s", level=logging.INFO
)

def convert_inference_to_dict(inference_str):
    if str(inference_str) == 'nan':
        return {}
    items = str(inference_str).split('_')
    return {items[i]: float(items[i + 1]) for i in range(0, len(items), 2)}

class ServiceType(str, Enum):
    """
    Define service type
    """
    PROCESSING = "processing"
    ENSEMBLE = "ensemble"
    INFERENCE = "inference"

class RandomData(BaseModel):
    data_range: dict
    data_size: Optional[int] = 5000
    current_index: Optional[int] = 0
    possitive_flag: Optional[bool] = True
    data_array: Optional[Any] = None
    
    def init_data(self):
        min_val = self.data_range["min"]
        max_val = self.data_range["max"]
        avg_val = self.data_range["avg"]
        if "base_scale" in self.data_range:
            base_scale = self.data_range["base_scale"]
        else:
            base_scale = 10
        if "spike_scale" in self.data_range:
            spike_scale = self.data_range["spike_scale"]
        else:
            spike_scale = 2
        if "noise_scale" in self.data_range:
            noise_scale = self.data_range["noise_scale"]
        else:
            noise_scale = 20
        
        # Base usage using normal distribution
        base_usage = np.random.normal(loc=avg_val, scale=max((max_val - min_val) / base_scale, 1e-6), size=self.data_size)
        
        # Spikes using a normal distribution centered around 0
        scale_spikes = max((max_val - avg_val) / spike_scale, 1e-6)
        spikes = np.random.choice([0, 1], size=self.data_size, p=[0.95, 0.05]) * np.random.normal(loc=0, scale=scale_spikes, size=self.data_size)
        
        # Noise using a small normal distribution centered around 0
        noise = np.random.normal(loc=0, scale=max((max_val - min_val) / noise_scale, 1e-6), size=self.data_size)
        
        self.data_array = base_usage + spikes + noise
        self.data_array = np.clip(self.data_array, min_val/2, 2*max_val)  # 

        if self.possitive_flag:
            self.data_array = np.clip(self.data_array, 0, 4*max_val)  # Ensure part_data is never lower than 0
    
    def get_randome_data(self):
        self.current_index = (self.current_index + 1) % self.data_size
        return self.data_array[self.current_index]
    
    def plot_data(self):
        if self.data_array is None:
            logging.debug("Data array is not initialized")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.data_array, label='Random Data')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Random Data Plot')
        plt.legend()
        plt.show()
        
class ServiceConfig(BaseModel):
    response_time_range: Optional[dict] = {}
    response_time_sim: Optional[int] = 1
    host: Optional[dict] = None
    service_name: Optional[str] = None
    service_type: Optional[str] = None
    mlmodel_name: Optional[str] = None
    data_path: Optional[str] = None
    ensemble_config_path: Optional[str] = None
    throughput: Optional[dict] = None
    throughput_requirement: Optional[int] = None
    response_time: Optional[dict] = None
    energy: Optional[dict] = None
    explainability: Optional[int] = 0
    
    
class BaseService(ABC):
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.response_time_generator = None
        if int(self.config.response_time_sim) == 1:
            self.response_time_generator = RandomData(data_range=self.config.response_time_range)
            self.response_time_generator.init_data()
    
    @abstractmethod
    def execute(self, data: dict):
        pass

class ProcessingService(BaseService):
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
    
    def execute(self, data: dict):
        service_type = self.config.service_type
        service_name = self.config.service_name
        if self.response_time_generator is not None:
            response_time = self.response_time_generator.get_randome_data()
            logging.debug(f"Response time on {service_name}: {response_time}")
        else:
            response_time = -1
        if service_type is not None:
            if service_type == ServiceType.PROCESSING.value:
                if "response_time" not in list(data.keys()):
                    data["response_time"] = {}
                if service_type not in list(data["response_time"].keys()):
                    data["response_time"][service_type] = {}
                data["response_time"][service_type][service_name] = response_time
        return data

class InferenceService(BaseService):
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.df = None
        if self.config.data_path is not None and self.config.mlmodel_name is not None:
            data_path = os.path.join(self.config.data_path, f"{self.config.mlmodel_name}.csv")
            self.df = pd.read_csv(data_path)
        self.current_inferences = 0
        if self.config.host is None:
            self.config.host = {"cpu": 0, "cuda": 1, "tpu": 0}
        if "cpu" not in list(self.config.host.keys()):
            self.config.host["cpu"] = 0
        if "cuda" not in list(self.config.host.keys()):
            self.config.host["cuda"] = 0
        if "tpu" not in list(self.config.host.keys()):
            self.config.host["tpu"] = 0
        self.current_energy_consumption = 0
        self.last_energy_consumption = 0
        self.last_inferences = 0
        self.response_time_generator_cpu = None
        self.response_time_generator_cuda = None
        self.response_time_generator_tpu = None
        cpu_response_time_range = copy.deepcopy(self.config.response_time_range)
        cpu_response_time_range["max"] = self.config.response_time["cpu"]["max"]
        cpu_response_time_range["min"] = self.config.response_time["cpu"]["min"]
        cpu_response_time_range["avg"] = self.config.response_time["cpu"]["avg"]
        cuda_response_time_range = copy.deepcopy(self.config.response_time_range)
        cuda_response_time_range["max"] = self.config.response_time["cuda"]["max"]
        cuda_response_time_range["min"] = self.config.response_time["cuda"]["min"]
        cuda_response_time_range["avg"] = self.config.response_time["cuda"]["avg"]
        tpu_response_time_range = copy.deepcopy(self.config.response_time_range)
        tpu_response_time_range["max"] = self.config.response_time["tpu"]["max"]
        tpu_response_time_range["min"] = self.config.response_time["tpu"]["min"]
        tpu_response_time_range["avg"] = self.config.response_time["tpu"]["avg"]
        if int(self.config.response_time_sim) == 1:
            self.response_time_generator_cpu = RandomData(data_range=cpu_response_time_range)
            self.response_time_generator_cpu.init_data()
            self.response_time_generator_cuda = RandomData(data_range=cuda_response_time_range)
            self.response_time_generator_cuda.init_data()
            self.response_time_generator_tpu = RandomData(data_range=tpu_response_time_range)
            self.response_time_generator_tpu.init_data()
    
    def reset(self):
        self.current_inferences = 0
        self.current_energy_consumption = 0
        self.last_energy_consumption = 0
        self.last_inferences = 0
    
    def energy_estimate(self):
        self.current_throughput = 0
        for key, value in self.config.host.items():
            i_throughput = self.config.throughput[key]
            self.current_throughput += (i_throughput*value)
        if self.current_throughput < self.config.throughput_requirement:
            a_instance = int((self.config.throughput_requirement - self.current_throughput)/self.config.throughput["cuda"]) +1
            self.config.host["cuda"] += a_instance
        energy_report = {}
        instances = self.config.host
        energy_report["instances"] = instances
        total_instances = instances["cpu"] + instances["cuda"] + instances["tpu"]
        inference_diff = self.current_inferences - self.last_inferences
        
        cpu_infernces = int(instances["cpu"] * inference_diff / total_instances)
        tpu_infernces = int(instances["tpu"] * inference_diff / total_instances)
        cuda_infernces = inference_diff - cpu_infernces - tpu_infernces
        
        energy_per_inference_cpu = self.config.energy["cpu"]/(self.config.throughput["cpu"]*3600)
        cpu_inference_energy = cpu_infernces * energy_per_inference_cpu
        energy_per_inference_cuda = self.config.energy["cuda"]/(self.config.throughput["cuda"]*3600)
        cuda_inference_energy = cuda_infernces * energy_per_inference_cuda
        energy_per_inference_tpu = self.config.energy["tpu"]/(self.config.throughput["tpu"]*3600)
        tpu_inference_energy = tpu_infernces * energy_per_inference_tpu
        
        if self.config.explainability == 1:
            cpu_inference_energy = 1.5*cpu_inference_energy
            cuda_inference_energy = 1.5*cuda_inference_energy
            tpu_inference_energy = 1.5*tpu_inference_energy
            
        self.last_energy_consumption = self.current_energy_consumption
        self.current_energy_consumption = cpu_inference_energy + cuda_inference_energy + tpu_inference_energy
        energy_report["energy"] = {"cpu": cpu_inference_energy, 
                                   "cuda": cuda_inference_energy,
                                   "tpu": tpu_inference_energy,
                                   "total": self.current_energy_consumption}
        self.last_inferences = self.current_inferences
        return energy_report
    
    def execute(self, data: dict):
        self.current_inferences += 1
        service_type = self.config.service_type
        mlmodel_name = self.config.mlmodel_name
        if self.response_time_generator is not None:
            instances = self.config.host
            total_instances = instances["cpu"] + instances["cuda"] + instances["tpu"]
            rand_int = np.random.randint(0, total_instances)
            if rand_int < instances["cpu"]:
                response_time = self.response_time_generator_cpu.get_randome_data()
            elif rand_int < instances["cpu"] + instances["tpu"]:
                response_time = self.response_time_generator_tpu.get_randome_data()  
            else:
                response_time = self.response_time_generator_cuda.get_randome_data()
                
            if self.config.explainability == 1:
                response_time = 2*response_time
            logging.debug(f"Response time on {mlmodel_name}: {response_time}")
        else:
            response_time = -1
        if service_type is not None:
            if service_type == ServiceType.INFERENCE.value:
                if "response_time" not in list(data.keys()):
                    data["response_time"] = {}
                if service_type not in list(data["response_time"].keys()):
                    data["response_time"][service_type] = {}
                data["response_time"][service_type][mlmodel_name] = response_time
                
        if self.df is not None:
            file_name = data["input"]["file_name"]
            inference_rows = self.df[self.df['file_name'] == file_name]
            inference_dict = {}
            if len(inference_rows) > 0:
                for index, row in inference_rows.iterrows():
                    inference_dict = convert_inference_to_dict(row['inference'])
                    break
            logging.debug(f"Inference result: {inference_dict}")
            
            if "ml_inference" not in list(data.keys()):
                data["ml_inference"] = {}
            data["ml_inference"][mlmodel_name] = inference_dict
            if "explainability" not in list(data.keys()):
                data["explainability"] = {}
            data["explainability"][mlmodel_name] = self.config.explainability
        return data
    
class EnsembleService(BaseService):
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.ensemble = {}
    
    def add_model(self, model: InferenceService):
        mlmodel_name = model.config.mlmodel_name
        self.ensemble[mlmodel_name] = model

    def load_model_from_yaml(self, model_name: str, throughput_req, yaml_path: str):
        try:
            with open(yaml_path, 'r') as file:
                model_profiles = yaml.safe_load(file)
            
            # Check if the specified model exists in the YAML file
            if model_name not in model_profiles:
                logging.error(f"Model '{model_name}' not found in YAML file: {yaml_path}")
                return
            
            # Retrieve the configuration for the specified model
            model_config = model_profiles[model_name]
            
            # Prepare response time and throughput data
            response_time_data = model_config.get("response_time", {})
            throughput_data = model_config.get("throughput", {})
            energy_data = model_config.get("energy", {})
            
            # Create the ServiceConfig for the model
            service_config = ServiceConfig(
                service_name=model_name,
                mlmodel_name=model_name,
                service_type=ServiceType.INFERENCE.value,
                response_time={
                    "cpu": {
                        "max": response_time_data.get("cpu", {}).get("max", 1),
                        "min": response_time_data.get("cpu", {}).get("min", 0.1),
                        "avg": response_time_data.get("cpu", {}).get("avg", 0.5),
                    },
                    "cuda": {
                        "max": response_time_data.get("cuda", {}).get("max", 1),
                        "min": response_time_data.get("cuda", {}).get("min", 0.1),
                        "avg": response_time_data.get("cuda", {}).get("avg", 0.5),
                    },
                    "tpu": {
                        "max": response_time_data.get("tpu", {}).get("max", 1),
                        "min": response_time_data.get("tpu", {}).get("min", 0.1),
                        "avg": response_time_data.get("tpu", {}).get("avg", 0.5),
                    },
                },
                response_time_range={
                    "max": response_time_data.get("cpu", {}).get("max", 1),
                    "min": response_time_data.get("cpu", {}).get("min", 0.1),
                    "avg": response_time_data.get("cpu", {}).get("avg", 0.5),
                },
                response_time_sim=1,
                host={
                    "cpu": response_time_data.get("cpu", {}).get("host", 1),
                    "cuda": response_time_data.get("cuda", {}).get("host", 1),
                    "tpu": response_time_data.get("tpu", {}).get("host", 1),
                },
                throughput={
                    "cpu": throughput_data.get("cpu", 0),
                    "cuda": throughput_data.get("cuda", 0),
                    "tpu": throughput_data.get("tpu", 0),
                },
                throughput_requirement=throughput_req,
                energy={
                    "cpu": energy_data.get("cpu", 0),
                    "cuda": energy_data.get("cuda", 0),
                    "tpu": energy_data.get("tpu", 0),
                },
                explainability=model_config.get("explainability", 0)
            )

            # Print the loaded service configuration
            print(service_config)
            
            # Instantiate the InferenceService and add it to the ensemble
            inference_service = InferenceService(config=service_config)
            self.add_model(inference_service)

            logging.info(f"Model '{model_name}' successfully loaded from YAML file.")
        
        except FileNotFoundError:
            logging.error(f"YAML file not found: {yaml_path}")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
    
    def load_models_from_yaml(self, throughput_req, yaml_path: str):
        try:
            with open(yaml_path, 'r') as file:
                model_profiles = yaml.safe_load(file)
            
            if not model_profiles:
                logging.warning(f"No models found in YAML file: {yaml_path}")
                return

            # Iterate over each model in the YAML and load them using load_model_from_yaml
            for model_name in model_profiles.keys():
                self.load_model_from_yaml(model_name=model_name, throughput_req=throughput_req, yaml_path=yaml_path)

            
            logging.info("All models loaded from YAML file.")
        
        except FileNotFoundError:
            logging.error(f"YAML file not found: {yaml_path}")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
    
    def remove_model_from_ensemble(self, model_name: str):
        try:
            # Check if the ensemble contains the model
            model_found = False
            for inference_service in self.ensemble:
                if inference_service.config.mlmodel_name == model_name:
                    # Remove the model from the ensemble
                    self.ensemble.remove(inference_service)
                    model_found = True
                    logging.info(f"Model '{model_name}' successfully removed from the ensemble.")
                    break
            
            if not model_found:
                logging.warning(f"Model '{model_name}' not found in the ensemble.")
        
        except Exception as e:
            logging.error(f"An error occurred while removing model '{model_name}': {e}")
            
    def energy_estimate(self):
        ensemble_energy = 0
        energy_report = {}
        for mlmodel_name, model in self.ensemble.items():
            energy_report[mlmodel_name] = model.energy_estimate()
            ensemble_energy += energy_report[mlmodel_name]["energy"]["total"]
        energy_report["ensemble"] = ensemble_energy
        return energy_report

    def execute(self, data: dict):
        for mlmodel_name, model in self.ensemble.items():
            data = model.execute(data)
        
        service_type = self.config.service_type
        service_name = self.config.service_name
        if self.response_time_generator is not None:
            response_time = self.response_time_generator.get_randome_data()
            logging.debug(f"Response time on {service_name}: {response_time}")
        else:
            response_time = -1
            logging.warning("Response time generator of ensemble is not initialized")
        
        max_inference_time = 0
        if "response_time" in list(data.keys()):
            response_time_data = data["response_time"]
            inference_type = ServiceType.INFERENCE.value
            if inference_type in list(response_time_data.keys()):
                inferenct_time_data = response_time_data[inference_type]
                for inference_name, inference_time in inferenct_time_data.items():
                    if inference_time > max_inference_time:
                        max_inference_time = inference_time
        response_time += max_inference_time
        if service_type is not None:
            if service_type == ServiceType.ENSEMBLE.value:
                if "response_time" not in list(data.keys()):
                    data["response_time"] = {}
                if "inference" not in list(data["response_time"].keys()):
                    data["response_time"]["inference"] = {}
                data["response_time"]["inference"]["ensemble"] = response_time
        
        max_prop = 0
        max_key = None
        # Initialize a dictionary to store the sum of probabilities for each class
        sum_probabilities = {}
        # Initialize a dictionary to store the count of occurrences for each class
        count_probabilities = {}
        
        if "ml_inference" in list(data.keys()):
            inference_data = data["ml_inference"]
            for mlmodel_name, inference_dict in inference_data.items():
                for key, value in inference_dict.items():
                    if value > max_prop:
                        max_prop = value
                        max_key = key
                    
                    if key not in sum_probabilities:
                        sum_probabilities[key] = 0
                        count_probabilities[key] = 0
                    sum_probabilities[key] += value
                    count_probabilities[key] += 1
                        
            if max_key is not None:
                inference_data["ensemble"] = {max_key: max_prop}
                
            # Calculate the average probability for each class
            avg_probabilities = {class_name: sum_probabilities[class_name] / count_probabilities[class_name]
                                for class_name in sum_probabilities}

            # Find the class with the highest average probability
            if not avg_probabilities:
                # Handle the case where avg_probabilities is empty
                highest_avg_class = ""
                avg_probabilities = {}  # Ensure avg_probabilities is an empty dictionary
            else:
                highest_avg_class = max(avg_probabilities, key=avg_probabilities.get)

            # Construct the aggregated result
            # aggregated_result = {
            #     "decision": highest_avg_class,
            #     "avg_prediction": avg_probabilities,
            # }
            if "contribution" not in list(data.keys()):
                data["contribution"] = {}
            if highest_avg_class != "":
                for mlmodel_name, inference_dict in inference_data.items():
                    if highest_avg_class in list(inference_dict.keys()):
                        data["contribution"][mlmodel_name] = inference_dict[highest_avg_class]
                    else:
                        data["contribution"][mlmodel_name] = 0
                inference_data["ensemble"] = {highest_avg_class: avg_probabilities[highest_avg_class]}
            else:
                inference_data["ensemble"] = {}         
            inf_explainability = data["explainability"]
            inf_explainability["ensemble"] = 0
            for mlmodel_name, explainability in inf_explainability.items():
                if explainability == 1:
                    mlmodel_inf = inference_data[mlmodel_name]
                    if highest_avg_class in list(mlmodel_inf.keys()):
                        inf_explainability["ensemble"] = 1
                        break
            
            
        return data
        