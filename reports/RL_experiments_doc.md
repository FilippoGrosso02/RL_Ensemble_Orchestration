# Performing experiments to check the RL model learning capabilities

# The accuracy test

- Target: 100% accuracy

- Total_num_models = 5

- Num-Actions = 11

- List of models =  ["EfficientNetV2S", "DenseNet121", "ResNet50", "MobileNetV2", "NASNetLarge"]

Best performing models: 1. EfficientNetV2S, 2. Nas NETLarge

We want to check if overtime the model learns to include the models with the highest accuracy and discard the ones with less accuracy over time.

## Strategy:

- To follow the normalization of reward we set the 0 reward to an average value of 0.9 so that bad results will be negative

- Each step corresponds to 100 inferences


## Example Test 1: standard parameters (256 num_steps, 20 model updates)

Results are found in the /results/ppo/reinforcement_learning_acc_1302 file


# PPO hyper-parameters

We want to see how varying these hyperparameters affect the convergence of the model:

- Learning rate: the step size for the optimizer

- Batch size: the amount of steps before eeach update in the model's policy

- Gamma: discount factor on future rewards

- Clip_range: clipping the gradient

## Parallelization: number of enviroments

We want to explore the possibility of increasing the number of enviroments to speed up the training process





