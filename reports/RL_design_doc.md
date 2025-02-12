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

