import logging
import os
import numpy as np

from managers.state_manager import StateManager 
from managers.visualization_manager import VisualizationManager, DashApp
from managers.objective import rl_reward_estimation
from managers.config_manager import ConfigManager

import gym
from gym import spaces

class SimulationEnv(gym.Env):

    def __init__(self, path_name):
        super(SimulationEnv, self).__init__()

        current_dir = os.getcwd()   
        
        parent_dir = os.path.dirname(current_dir)
        CONFIG_PATH = os.path.join(current_dir, "sim_config.yaml")
        PROFILE_PATH = os.path.join(current_dir, "profile/model_profile/model_profile.yaml")
        self.CONTRACT_PATH = os.path.join(current_dir, "config/contract_metrics.json")

        self.state_manager = StateManager(current_dir)
        self.config_manager = ConfigManager(PROFILE_PATH, CONFIG_PATH)
        self.visualization_manager = VisualizationManager(current_dir, path_name)
        self.dash_app = DashApp(self.visualization_manager)

        self.dash_app.run()

        # Define constants
        self.state_size = self.state_manager.state_lenght # Change this to be dynamic based on the enviroment
        # Observation space: Fixed size of 55
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(self.state_size,), dtype=np.float32
        )

        # Action space (example: add/remove/replace a model)
        self.action_space = spaces.Discrete(4)

        self.weights = {
            "accuracy": 1.0,
            "confidence": 0.0,
            "latency": 0.0,
            "energy": 0.0,
            "explainability": 0.0
        }

    def reset(self):
        # Reset the environment to an initial state
        # Define Init Data:
        
        self.current_state = self.state_manager.get_state()  # Get initial state
        
        return self.state_manager.flatten_structured_state(self.current_state)

    def step(self, action):
        
        if action == 0:
            self.apply_action("keep_ensemble")
        elif action == 1:
            self.apply_action("add_model")
        elif action == 2:
            self.apply_action("replace_model")
        elif action == 3:
            self.apply_action("remove_model")
      
        state = self.state_manager.get_state()
        print("STATE: ", state)

        # Simulate a step in the environment
        self.current_state = state
        reward = self._calculate_reward(self.current_state)

        state["reward"] = reward
        state["action"] = action

        self.visualization_manager.reward_list.append(reward)
        self.visualization_manager.add_state_to_csv(state)
        self.dash_app.update_graph()

        print("REWARD: ", reward)

        done = False  # Define termination condition if applicable
        
        return  self.state_manager.flatten_structured_state(self.current_state), reward, done, {}


    def _calculate_reward(self, state):

        # TO REDO
        model_states = state["model_states"]["ensemble"]
        performance_metrics = {}
        performance_metrics["accuracy"] = model_states["accuracy"]
        performance_metrics["confidence"] = model_states["confidence"]
        performance_metrics["explainability"] = 1.0
        performance_metrics["energy"] = state["ensemble_state"]["total_energy_consumption"]
        performance_metrics["latency"] = model_states["avg_response_time"]
         
        reward = rl_reward_estimation(performance_metrics, self.CONTRACT_PATH)

        return reward
    
    def apply_action(self, action):
    
        weights = self.weights
        manager = self.config_manager
        if action == "keep_ensemble":
            logging.info("Action: Keeping the ensemble")
        
        elif action == "add_model":
            manager.add_best_model(weights)
            logging.info("Action: Adding a model")
            # Logic for adding a new model (Placeholder)
        elif action == "replace_model":
            logging.info("Action: Replacing a model")
            manager.remove_worst_model(weights)
            manager.add_best_model(weights)
            # Logic for replacing a model (Placeholder)
        elif action == "remove_model":
            manager.remove_worst_model(weights)

        elif action == "add_random_model":
            manager.add_random_model()
        elif action == "remove_random_model":
            manager.remove_random_model()
        elif action == "replace_random_model":
            manager.remove_random_model()
            manager.add_random_model()

        else:
            logging.warning("Unknown action")