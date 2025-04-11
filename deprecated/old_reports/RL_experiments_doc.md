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


### PPO Hyperparameter Sensitivity Test Plan

1. **Baseline Run**  
   `--n_envs 4 --total_timesteps 5120 --learning_rate 3e-4 --n_steps 128 --batch_size 64 --clip_range 0.2 --n_epochs 10 --net_arch "[256, 256]"`

---

#### Architecture Tests

2. Shallow network  
   `--net_arch "[64, 64]"`

3. Deep network  
   `--net_arch "[512, 512, 256]"`

4. Narrow network  
   `--net_arch "[32, 32]"`

5. Separate actor/critic  
   `--net_arch "[{'pi': [64, 64], 'vf': [128, 128]}]"`

6. Small policy, big critic  
   `--net_arch "[{'pi': [32], 'vf': [256, 256]}]"`

---

#### Epochs and Update Strategy

7. Fewer epochs  
   `--n_epochs 3`

8. More epochs  
   `--n_epochs 20`

9. Many small updates  
   `--n_steps 64 --batch_size 32`

10. Fewer large updates  
    `--n_steps 256 --batch_size 128`

---

#### Learning Rate Sensitivity

11. Very small LR  
    `--learning_rate 1e-5`

12. Small LR  
    `--learning_rate 1e-4`

13. Large LR  
    `--learning_rate 1e-3`

14. Very large LR (may cause instability)  
    `--learning_rate 5e-3`

---

#### Clip Range & Stability

15. Small clip range  
    `--clip_range 0.1`

16. Large clip range  
    `--clip_range 0.3`

---

#### Parallelization Impact

17. Fewer envs (less exploration per step)  
    `--n_envs 1`

18. More envs (faster wall time, more diverse samples)  
    `--n_envs 8`

---

19. Even more envs (faster wall time, more diverse samples)  
    `--n_envs 16`

---





