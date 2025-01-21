import csv
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import threading



class VisualizationManager:
    def __init__(self, csv_path=f"results/reinforcement_learning.csv"):
        """
        Initialize the VisualizationManager.

        Args:
            csv_path (str): Path to the CSV file where states will be saved.
        """
        self.csv_path = csv_path

        # Check if the file exists; if not, create it with headers
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self._get_headers())

        self.reward_list = []

    def _get_headers(self):
        """
        Define the headers for the CSV file.

        Returns:
            list: List of column headers.
        """
        headers = [
            "total_energy_consumption",
            "ensemble_size",
            "input_file_length",
            "image_height",
            "image_width",
        ]

        # Add headers for up to 10 models with 5 metrics each
        for i in range(10):
            headers.extend([f"model_{i}_metric_{j}" for j in range(5)])

        return headers

    def add_state_to_csv(self, state):
        """
        Append a state dictionary as a row in the CSV file.

        Args:
            state (dict): The state dictionary containing metrics.
        """
        # Flatten the state dictionary into a single row
        row = self._flatten_state(state)

        # Write the row to the CSV file
        with open(self.csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def _flatten_state(self, state):
        """
        Flatten the state dictionary into a list for CSV storage.

        Args:
            state (dict): The state dictionary containing metrics.

        Returns:
            list: Flattened state as a single row.
        """
        ensemble_state = state["ensemble_state"]
        model_states = state["model_states"]
        input_state = state["input_state"]

        # Extract ensemble and input state metrics
        row = [
            ensemble_state.get("total_energy_consumption", 0.0),
            ensemble_state.get("ensemble_size", 0),
            input_state[0],  # input_file_length
            input_state[1],  # image_height
            input_state[2],  # image_width
        ]

        # Flatten model states (ensure fixed size for 10 models)
        for model in model_states[:10]:
            row.extend(model)

        # Add padding for missing models
        for _ in range(10 - len(model_states)):
            row.extend([0.0] * 5)

        return row



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

