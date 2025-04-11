# Designing the RL enviroment

## Problems

- The reward is very dependent on chance, need a big sample size to see the true effects of adding or removing a model (100-1000), this is computationally expensive but might be a good task for parallelization

- Computing 1000 simulated inferences is time consuming, CSC resources might be a good way to speed up the training process

## The RL architecture

### Goal
The RL model learns to select the best set of models out of a ML model list to optimize a metrics profile in ML inference.

### Actions
The action space of the model is: one action for adding each model in the list, one action for removing each model, one action to keep the ensemble as is

### Steps
Each step the model performs:
- one action on the ensemble (either adding a model, removing it or keeping it as is)
- a certain number of inferences with that particular ensemble (we set the default to 100 inferences) 
- compute the reward over these inferences based on the predefined weights in the contract (accuracy, energy estimate, etc..)

### Updates
* This section holds for the PPO architecture

The policy of the RL model is updated every x steps based on the actions and rewards observed

## Tests

We want to see if the RL method can learn the optimal ensemble over time without knowing anything a priori about the models and converge to the ideal ensmeble (calculated with an algorithm that knows the model performances over the whole set of inputs)

If this is the case the RL model could be applied to situations where the algorithmic scoring isn't ideal, for example when the distribution of labels in the input changes  the optimal set might also need to change

Overall a "learning" RL model for ensembe orchestration would prvide a more versatile and out of the box solution compared to having to score and test each model before running the inference. 

### Parameters to test

### **1. learning_rate (`Learning Rate`)**  
Controls how much the model updates its weights; a lower value makes training more stable but slower, while a higher value speeds up training but risks instability.

### **2. `n_steps` (Rollout Buffer Size)**  
Defines how many steps the agent collects before updating; larger values lead to more stable training but higher memory usage.

### **3. `batch_size` (Minibatch Size)**  
Determines the number of samples used per update; smaller sizes improve exploration, while larger sizes lead to smoother updates.

### **4. `clip_range` (Clipping Range for Policy Updates)**  
Controls how much the policy can change in each update to maintain stability; typically set to `0.1`–`0.2`.

### **5. `net_arch` (Neural Network Architecture)**  
Specifies the number of neurons per layer in the policy and value networks; deeper networks improve learning in complex environments.

### **6. `n_envs` (Number of Parallel Environments)**  
Defines the number of environments running in parallel; higher values speed up training but increase resource usage.
