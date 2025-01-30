import csv
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import threading
import pandas as pd


class VisualizationManager:
    def __init__(self, current_dir, csv_path=f"default"):

        # Construct the full directory path
        self.csv_dir = os.path.join(current_dir, "results", csv_path)
        self.csv_path = os.path.join(self.csv_dir, "reinforcement_learning.csv")

        # Ensure the directory exists
        os.makedirs(self.csv_dir, exist_ok=True)

        # Remove the file if it already exists
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

        # Create a new CSV file with headers
        with open(self.csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self._get_headers())

        # Initialize reward list
        self.reward_list = []

    def _get_headers(self):

        headers = [
            "total_energy_consumption",
            "ensemble_size",
            "input_file_length",
            "image_height",
            "image_width",
            "ensemble_accuracy",
            "ensemble_confidence",
            "ensemble_avg_response_time",
            "ensemble_max_response_time",
            "ensemble_contribution",
            "reward",
            "action",
            "distribution_weights"
        ]

        return headers



    def add_state_to_csv(self, state):

        # Flatten the state dictionary into a single row
        row = list(self.flatten_state(state).values())

        # Write the row to the CSV file
        with open(self.csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)


    def flatten_state(self,state, parent_key='', sep='_'):
        """
        STATE INFORMATION
        The state is composed of the following components and keys:

        1. Ensemble State (`ensemble_state`):
        - `total_energy_consumption`: 0.01894598299398171
        - `ensemble_size`: 14

        2. Model States (`model_states`):
        - Contains individual model details with the following keys:
            - `accuracy`: Model's accuracy in current inference tasks.
            - `confidence`: Confidence level of the model's predictions.
            - `avg_response_time`: Average response time of the model (in seconds).
            - `max_response_time`: Maximum response time observed for the model (in seconds).
            - `contribution`: Model's contribution to the ensemble.
        - Example Models:
            - **InceptionResNetV2**:
                - `accuracy`: 0.88
                - `confidence`: 0.8021299481391907
                - `avg_response_time`: 2.0684926527844554
                - `max_response_time`: 2.140760123997036
                - `contribution`: 0.8157378768920899
            - **MobileNetV2**:
                - `accuracy`: 0.6666666666666666
                - `confidence`: 0.515729824701945
                - `avg_response_time`: 0.06555019974484746
                - `max_response_time`: 0.10884077820269697
                - `contribution`: 0.515729824701945
            - **ResNet50V2**:
                - `accuracy`: 1.0
                - `confidence`: 0.5565559466679891
                - `avg_response_time`: 0.11347026663042319
                - `max_response_time`: 0.12229664638153938
                - `contribution`: 0.5565559466679891
            - **... (other models follow the same structure)**:
            - **Ensemble Summary** (`ensemble`):
                - `accuracy`: 0.96
                - `confidence`: 0.82377298942463
                - `avg_response_time`: 6.255583806121645
                - `max_response_time`: 7.168263599886896
                - `contribution`: 0.940839307308197

        3. Input State (`input_state`):
        - `input_file_length`: 15
        - `image_height`: 224
        - `image_width`: 224

        4. Distribution Weights (`distribution_weights`):
        - A list of weights applied to models during inference.
        - Example: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, ..., 0.0]


        """
        flattened_state_with_weights = {
        'total_energy_consumption': state['ensemble_state']['total_energy_consumption'],
        'ensemble_size': state['ensemble_state']['ensemble_size'],
        'input_file_length': state['input_state']['input_file_length'],
        'image_height': state['input_state']['image_height'],
        'image_width': state['input_state']['image_width'],
        'ensemble_accuracy': state['model_states']['ensemble']['accuracy'],
        'ensemble_confidence': state['model_states']['ensemble']['confidence'],
        'ensemble_avg_response_time': state['model_states']['ensemble']['avg_response_time'],
        'ensemble_max_response_time': state['model_states']['ensemble']['max_response_time'],
        'ensemble_contribution': state['model_states']['ensemble']['contribution'],
        'reward' : state['reward'],
        'action' : state['action'],
        'distribution_weights': state['distribution_weights']  # Add distribution weights
        }


        return flattened_state_with_weights




class DashApp:
    def __init__(self, viz_manager, title="Real-Time Reward Visualization"):

        self.viz_manager = viz_manager
        self.app = dash.Dash(__name__)
        self.title = title
        self._setup_layout()


    def _setup_layout(self):

        self.app.layout = html.Div([
            html.H1(self.title),
            dcc.Graph(id="reward-graph"),
            dcc.Interval(id="interval-update", interval=100, n_intervals=0)  # Poll every 100ms
        ])
    
    def _generate_figure(self):
        rewards = self.viz_manager.reward_list
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=list(range(len(rewards))),
            y=rewards,
            mode="lines+markers",
            name="Rewards"
        ))
        figure.update_layout(
            title="Rewards Over Time",
            xaxis_title="Episode",
            yaxis_title="Reward",
            template="plotly_dark"
        )
        return figure
    
    def update_graph(self):

        self.current_figure = self._generate_figure()
        # Access the dcc.Graph component and update its figure
        self.app.layout.children[1].figure = self.current_figure
    
    def run(self, debug=True, **kwargs):

        self.app.run_server(debug=True, host="127.0.0.1", port=8000)
        # find it here:
        # http://127.0.0.1:8000

