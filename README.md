# Reinforcement Learning ML Ensemble Orcherstration
The goal of this research is to develop a **reinforcement learning (RL) component** that can dynamically orchestrate a **machine learning (ML) ensemble**, optimizing for user-defined metrics. 

Specifically, the RL algorithm will help the ensemble adapt to changing input data distributions in the inference pipeline. 

This RL system will be a **modular add-on** that can be configured through a text file, allowing it to integrate with various cloud-based pipelines.

In this research, the performance of various RL models for orchestration will be evaluated and compared against traditional ensemble orchestration methods, including:
- Constant ensemble
- Random action
- Algorithmic scoring

![RL System Architecture](images/RL_Orchestrator01.png)

## Prerequisites
The user needs ML ensemble inference pipeline with:
1. **Input Data**: for an optimal performance the data should labeled and provide an estimate of the distributions of labels in the data at any given moment.
2. **Ml Ensemble**: a set of containerized ML models, for now the user needs to run each model individually before setting up the pipeline to get an estimate of the model's performance metrics.
3. **Config File**: where the user specifies the structure of the cloud pipeline, its components and the utilized model.


## RL System components
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

## Reward Function and User-Defined Metrics
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

## Potential future research

- Developing **parallelized** reinforcement learning (RL) algorithms for improved scalability and efficiency.
- Extending RL-based orchestration to manage other cloud-based systems, such as IoT devices, drones, and similar distributed architectures.
