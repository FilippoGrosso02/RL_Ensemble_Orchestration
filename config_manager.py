import yaml
import random

class ConfigManager:
    def __init__(self, model_profile_path: str, target_yaml_path: str):
        self.model_profile_path = model_profile_path
        self.target_yaml_path = target_yaml_path
    
    # LOADING AND SAVING THE YAML 
    
    def load_yaml(self, path: str):
        """Load YAML file from the given path."""
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None

    def save_yaml(self, path: str, data: dict):
        """Save YAML data to the given path."""
        try:
            with open(path, 'w') as file:
                yaml.safe_dump(data, file, default_flow_style=False)
        except Exception as e:
            print(f"Error saving YAML file: {e}")


    ### BASIC CONFIG OPERATIONS
    
    def format_model_data(self, model_name: str, model_data: dict) -> dict:
        """Format model data to match the required structure with default values."""
        default_response_time_range = {
            "avg": 0.3,
            "base_scale": 40,
            "max": 0.5,
            "min": 0.1,
            "noise_scale": 20,
            "spike_scale": 3
        }

        return {
            model_name: {
                "explainability": model_data.get("explainability", 0),
                "host": model_data.get("host", {"cpu": 1}),
                "mlmodel_name": model_name,
                "response_time_range": model_data.get("response_time_range", default_response_time_range),
                "response_time_sim": model_data.get("response_time_sim", 1),
                "service_name": model_name,
            }
        }
   

    def add_model(self, model_name: str):
        """Add a single model to the target YAML."""
        model_profiles = self.load_yaml(self.model_profile_path)
        if model_profiles is None or model_name not in model_profiles:
            print(f"Model '{model_name}' not found in model profiles.")
            return

        # Format model data
        #print( model_profiles[model_name])
        formatted_model = self.format_model_data(model_name, model_profiles[model_name])


        # Load the target YAML
        target_yaml = self.load_yaml(self.target_yaml_path) or {}
        if 'inference' not in target_yaml:
            target_yaml['inference'] = {}

        # Add the model to the target YAML
        target_yaml['inference'].update(formatted_model)
        self.save_yaml(self.target_yaml_path, target_yaml)

        #print(formatted_model)
        print(f"Model '{model_name}' successfully added to the target YAML.")


    def remove_model(self, model_name: str):
        """Remove a single model from the target YAML."""
        target_yaml = self.load_yaml(self.target_yaml_path)
        if target_yaml is None or 'inference' not in target_yaml or model_name not in target_yaml['inference']:
            print(f"Model '{model_name}' not found in the target YAML.")
            return

        # Remove the model from the target YAML
        del target_yaml['inference'][model_name]
        self.save_yaml(self.target_yaml_path, target_yaml)
        print(f"Model '{model_name}' successfully removed from the target YAML.")
    
    
    def add_all_models(self):
        """Add all models from the model profiles to the target YAML."""
        model_profiles = self.load_yaml(self.model_profile_path)
        if model_profiles is None:
            print("No model profiles found.")
            return

        target_yaml = self.load_yaml(self.target_yaml_path) or {}
        if 'inference' not in target_yaml:
            target_yaml['inference'] = {}

        # Format and add all models to the target YAML
        for model_name, model_data in model_profiles.items():
            formatted_model = self.format_model_data(model_name, model_data)
            target_yaml['inference'].update(formatted_model)

        self.save_yaml(self.target_yaml_path, target_yaml)
        print("All models successfully added to the target YAML.")
    
    def add_random_model(self):
        """Add a random model from the model profiles to the target YAML."""
        model_profiles = self.load_yaml(self.model_profile_path)
        if model_profiles is None or not model_profiles:
            print("No models available in the model profiles.")
            return

        # Pick a random model name
        model_name = random.choice(list(model_profiles.keys()))
        self.add_model(model_name)

    def remove_random_model(self):
        """Remove a random model from the target YAML."""
        target_yaml = self.load_yaml(self.target_yaml_path)
        if target_yaml is None or 'inference' not in target_yaml or not target_yaml['inference']:
            print("No models available in the target YAML to remove.")
            return

        # Pick a random model name to remove
        model_name = random.choice(list(target_yaml['inference'].keys()))
        self.remove_model(model_name)


    ### MODEL SCORING 

    def calculate_model_score(self, model_data: dict, weights: dict) -> float:
        """Calculate the score for a model based on the given weights."""
        accuracy = model_data.get("overall_accuracy", 0.0)
        confidence = model_data.get("confidence", 0.0)
        latency = model_data.get("response_time_range", {}).get("avg", 0.5)
        energy = model_data.get("energy", {}).get("cpu", 5.0)
        explainability = model_data.get("explainability", 0)
        
        ## NEED TO NORMALIZE
        # Normalize metrics if needed (e.g., latency and energy)
        latency = max(0.1, latency)  # Avoid division by zero

        # Calculate score using weighted sum
        score = (
            weights["accuracy"] * accuracy +
            weights["confidence"] * confidence +
            weights["explainability"] * explainability -
            weights["latency"] * latency -
            weights["energy"] * energy
        )
        return score

    def select_best_model(self, weights: dict):
        """Select the best model based on scoring."""
        # Load model profiles and target YAML
        model_profiles = self.load_yaml(self.model_profile_path)
        target_yaml = self.load_yaml(self.target_yaml_path)

        if not model_profiles:
            print("No model profiles found.")
            return None

        # Extract models already present in the target YAML
        present_models = target_yaml.get('inference', {}).keys() if target_yaml and 'inference' in target_yaml else []

        # Compute scores for models not already in the target YAML
        model_scores = {}
        for model_name, model_data in model_profiles.items():
            if model_name in present_models:  # Skip models already in the target YAML
                continue
            score = self.calculate_model_score(model_data, weights)
            model_scores[model_name] = score

        if not model_scores:
            print("No eligible models to score.")
            return None

        # Find the model with the highest score
        best_model_name = max(model_scores, key=model_scores.get)
        print(f"Best model: {best_model_name} with score: {model_scores[best_model_name]}")
        return best_model_name


    def add_best_model(self, weights: dict):
        """Add the best model to the target YAML based on scoring."""
        best_model_name = self.select_best_model(weights)
        if best_model_name:
            self.add_model(best_model_name)

    def remove_worst_model(self, weights: dict):
        """Remove the worst model from the target YAML based on scoring."""
        # Load model profiles and target YAML
        model_profiles = self.load_yaml(self.model_profile_path)
        target_yaml = self.load_yaml(self.target_yaml_path)

        if not target_yaml or 'inference' not in target_yaml:
            print("No models available in the target YAML to remove.")
            return

        present_models = target_yaml['inference'].keys()

        if not model_profiles:
            print("No model profiles found.")
            return

        # Compute scores for models in the target YAML
        model_scores = {}
        for model_name in present_models:
            if model_name not in model_profiles:  # Ensure profile exists for the model
                continue
            model_data = model_profiles[model_name]
            score = self.calculate_model_score(model_data, weights)
            model_scores[model_name] = score

        if not model_scores:
            print("No models with valid scores found.")
            return

        # Find the model with the lowest score
        worst_model_name = min(model_scores, key=model_scores.get)
        print(f"Worst model: {worst_model_name} with score: {model_scores[worst_model_name]}")

        # Remove the worst model from the target YAML
        self.remove_model(worst_model_name)

