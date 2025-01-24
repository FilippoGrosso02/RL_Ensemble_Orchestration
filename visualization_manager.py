import csv
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import threading
import pandas as pd


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


    def flatten_dict(self,d, parent_key='', sep='_'):
        """Recursively flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def flatten_state_to_csv(self, state_dict):

        """Flattens the given state dictionary and writes it to a CSV file."""
        # Only consider the 'ensemble' model state
        filtered_state = {
            'ensemble_state': state_dict.get('ensemble_state', {}),
            'model_states': {
                'ensemble': state_dict.get('model_states', {}).get('ensemble', {})
            },
            'input_state': state_dict.get('input_state', {}),
            'distribution_weights': state_dict.get('distribution_weights', [])
        }

        # Flatten the filtered state dictionary
        flattened_state = self.flatten_dict(filtered_state)

        # Convert the flattened dictionary into a DataFrame
        df = pd.DataFrame([flattened_state])

        # Write the DataFrame to a CSV file
        try:
            df.to_csv(self.csv_path, mode='a', header=not pd.io.common.file_exists(self.csv_path), index=False)
            print(f"Flattened state appended to {self.csv_path}")
        except Exception as e:
            print(f"Error appending to CSV: {e}")



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

