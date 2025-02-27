import logging
import numpy as np
import os
import gym
from gym import spaces
import sys

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)
from rutils import rl_reward_estimation
from RL.visualize import VisualizationManager, DashApp
from RL.ensemble import RoheEnsemble
from simulation.eemls import EEMLSSimulation


    

class EEMLSEnv(gym.Env):
    def __init__(self, env_config):
        super(EEMLSEnv, self).__init__()
        
        sim_config = env_config["sim_config"]
        self.eemlse_simulation = EEMLSSimulation(sim_config)
        self.experiment_id = env_config["experiment_id"]
        self.ensemble_manager = RoheEnsemble(os.path.join(parent_dir, sim_config["profile_path"]), os.path.join(parent_dir, sim_config["config_path"]))
        self.visualization_manager = VisualizationManager(parent_dir, self.experiment_id)
        self.dash_app = DashApp(self.visualization_manager)
        self.contract_path = os.path.join(parent_dir, env_config["contract_path"])
        
        self.dash_app.run()
        self.min_models = env_config["min_models"]
        self.max_models = env_config["max_models"]
        self.num_models = env_config["num_models"]
        self.num_inferences = env_config["num_inferences"]
        self.state_size = self.eemlse_simulation.state_length
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(11)
        self.weights = env_config["weights"]
        
    def reset(self):
        # Reset the environment to an initial state
        # Define Init Data:
        
        self.current_state = self.eemlse_simulation.step_inference()  # Get initial state
        return self.eemlse_simulation.flatten_structured_state(self.current_state)
    
    def step(self, action):
        
        """ OLD 
        if action == 0 :
            self.apply_action("keep_ensemble")
        elif action == 1 and self.num_models < self.max_models:
            self.apply_action("add_model")
        elif action == 2:
            self.apply_action("replace_model")
        elif action == 3 and self.num_models > self.min_models: 
            self.apply_action("remove_model")
        else:
            self.apply_action("keep_ensemble")
        """
        self.apply_action(action)
        state = self.eemlse_simulation.step_inference(num_inferences = self.num_inferences)
        print("STATE: ", state)

        # Simulate a step in the environment
        self.current_state = state
        reward = self._calculate_reward(self.current_state)

        state["reward"] = reward
        state["action"] = action
        self.num_models = state["ensemble_state"]["ensemble_size"]
        state["models"] = list(self.eemlse_simulation.ensemble_service.ensemble.keys())

        self.visualization_manager.reward_list.append(reward)
        self.visualization_manager.add_state_to_csv(state)
        self.dash_app.update_graph()

        print("REWARD: ", reward)

        done = False  # Define termination condition if applicable
        
        return  self.eemlse_simulation.flatten_structured_state(self.current_state), reward, done, {}


    def _calculate_reward(self, state):

        # TO REDO
        model_states = state["model_states"]["ensemble"]
        performance_metrics = {}
        performance_metrics["accuracy"] = model_states["accuracy"]
        performance_metrics["confidence"] = model_states["confidence"]
        performance_metrics["explainability"] = 1.0
        performance_metrics["energy"] = state["ensemble_state"]["total_energy_consumption"]
        performance_metrics["latency"] = model_states["avg_response_time"]
         
        reward = rl_reward_estimation(performance_metrics, self.contract_path)

        return reward
    
    def apply_action_old(self, action):
    
        weights = self.weights
        eemls_ensemble = self.ensemble_manager
        if action == "keep_ensemble":
            logging.info("Action: Keeping the ensemble")
        
        elif action == "add_model":
            eemls_ensemble.add_best_model(weights)
            logging.info("Action: Adding a model")
            # Logic for adding a new model (Placeholder)
        elif action == "replace_model":
            logging.info("Action: Replacing a model")
            eemls_ensemble.remove_worst_model(weights)
            eemls_ensemble.add_best_model(weights)
            # Logic for replacing a model (Placeholder)
        elif action == "remove_model":
            eemls_ensemble.remove_worst_model(weights)

        elif action == "add_random_model":
            eemls_ensemble.add_random_model()
        elif action == "remove_random_model":
            eemls_ensemble.remove_random_model()
        elif action == "replace_random_model":
            eemls_ensemble.remove_random_model()
            eemls_ensemble.add_random_model()

        else:
            logging.warning("Unknown action")

    def apply_action(self, action):
        """
        Apply an action based on the given numeric value:
        - 0 to 4: Add a model corresponding to the index.
        - 5 to 9: Remove a model corresponding to the index - 5.
        - 10: Keep the ensemble as it is.
        """
        
        weights = self.weights
        eemls_ensemble = self.ensemble_manager

        if 0 <= action <= 4:
            logging.info(f"Action: Adding model at index {action}")
            eemls_ensemble.add_model_by_index(action)

        elif 5 <= action <= 9:
            model_index = action - 5
            logging.info(f"Action: Removing model at index {model_index}")
            if (self.num_models > 1): eemls_ensemble.remove_model_by_index(model_index)

        elif action == 10:
            logging.info("Action: Keeping the ensemble (no changes)")

        else:
            logging.warning("Unknown action")