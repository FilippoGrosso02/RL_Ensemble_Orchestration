# Reinforcement Learning Ensemble Orcherstration
The goal of this research is to develop a reinforcement learning (RL) component that can dynamically orchestrate a machine learning (ML) ensemble, optimizing for user-defined metrics. 

Specifically, the RL algorithm will help the ensemble adapt to changing input data distributions in the inference pipeline. 

This RL system will be a modular add-on that can be configured through a text file, allowing it to integrate with various cloud-based pipelines.

In this research, the performance of various RL models for orchestration will be evaluated and compared against traditional ensemble orchestration methods, including:
- Constant ensembles
- Random action
- Algorithmic scoring

![Graph Placeholder](#) <!-- Add your graph here -->

## Components
The main components of the RL add-on:

1. **State Manager**
   - **Function 1: Config Pipeline**
     - Configures the pipeline based on the settings in the provided config file.
   - **Function 2: Get State**
     - Retrieves the current state at any given moment and sends it to the RL model.

2. **RL Manager**
   - Selects and runs different RL models, utilizing the State Manager to get the current state during each step.

3. **Visualizer**
   - Collects the outputs and generates visualizations and metrics, allowing for the evaluation of different RL policies within the cloud environment.

4. **Utils**
   - **Config Manager**
     - Modifies the YAML file, enabling various actions to be called by the RL model.
   - **Model RL**
     - Simulates ML inference based on the content specified in the YAML file.

## Reward System and User-Defined Metrics
The reward of the RL model is defined by the following formula:

**Reward = Σ(αᵢ * f(xᵢ))**

Where:
- **αᵢ** is a set of weights describing what metrics the user prioritizes.
- **f(xᵢ)** is a function applied to the metrics that normalizes their value to [0, 1].

### Inference Metrics
The inference metrics considered for now are:
- **Accuracy**
- **Confidence**
- **Explainability**
- **Latency**
- **Energy**
