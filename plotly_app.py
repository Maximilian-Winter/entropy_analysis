import os
import json
import datetime
import numpy as np
import pandas as pd
from collections import Counter

import dash
from dash import Dash, html, dcc, Output, Input, State
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Function to load generation results
def load_generation_results(folder_name):
    with open(os.path.join(folder_name, 'generation_results.json'), 'r') as f:
        generation_results = json.load(f)
    return generation_results


# Function to load metadata
def load_metadata(folder_name):
    with open(os.path.join(folder_name, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    return metadata


# Function to create entropy over time figure
def create_entropy_over_time_figure(generation_results):
    steps = list(range(1, len(generation_results['step_analyses']) + 1))
    logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
    attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=logits_entropies, mode='lines+markers', name='Logits Entropy'))
    fig.add_trace(go.Scatter(x=steps, y=attention_entropies, mode='lines+markers', name='Attention Entropy'))
    fig.update_layout(title='Entropy Over Time', xaxis_title='Generation Step', yaxis_title='Entropy')
    return fig


# Function to create model states figure
def create_model_states_figure(generation_results):
    steps = list(range(1, len(generation_results['step_analyses']) + 1))
    states = [step['model_state'] for step in generation_results['step_analyses']]

    state_to_num = {
        'Very Uncertain': 0,
        'Uncertain': 1,
        'Slightly Uncertain': 2,
        'Exploring': 3,
        'Balanced': 4,
        'Focusing': 5,
        'Confident': 6,
        'Highly Confident': 7,
        'Overconfident': 8,
        'Very Overconfident': 9
    }
    numeric_states = [state_to_num.get(state, -1) for state in states]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=numeric_states, mode='lines+markers', name='Model State'))
    fig.update_layout(title='Model State Over Time', xaxis_title='Generation Step', yaxis_title='Model State',
                      yaxis=dict(tickmode='array', tickvals=list(state_to_num.values()),
                                 ticktext=list(state_to_num.keys())))
    return fig


# Function to create entropy distribution figure
def create_entropy_distribution_figure(generation_results):
    logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
    attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Logits Entropy Distribution', 'Attention Entropy Distribution'))
    fig.add_trace(go.Histogram(x=logits_entropies, nbinsx=20, name='Logits Entropy'), row=1, col=1)
    fig.add_trace(go.Histogram(x=attention_entropies, nbinsx=20, name='Attention Entropy'), row=1, col=2)
    fig.update_layout(title='Entropy Distribution', showlegend=False)
    return fig


# Function to create surprisal over time figure
def create_surprisal_over_time_figure(generation_results):
    steps = list(range(1, len(generation_results['step_analyses']) + 1))
    surprisal_values = [step['surprisal'] for step in generation_results['step_analyses']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=surprisal_values, mode='lines+markers', name='Surprisal'))
    fig.update_layout(title='Surprisal Over Time', xaxis_title='Generation Step',
                      yaxis_title='Surprisal (Negative Log Probability)')
    return fig


# Function to create entropy correlation figure
def create_entropy_correlation_figure(generation_results):
    logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
    attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

    correlation = np.corrcoef(logits_entropies, attention_entropies)[0, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=logits_entropies, y=attention_entropies, mode='markers', name='Entropy Scatter'))
    fig.update_layout(title=f'Logits vs Attention Entropy (Correlation: {correlation:.2f})',
                      xaxis_title='Logits Entropy', yaxis_title='Attention Entropy')
    return fig


# Function to create entropy gradient figure
def create_entropy_gradient_figure(generation_results):
    logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
    attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

    logits_gradient = np.gradient(logits_entropies)
    attention_gradient = np.gradient(attention_entropies)

    steps = list(range(1, len(logits_gradient) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=logits_gradient, mode='lines+markers', name='Logits Entropy Gradient'))
    fig.add_trace(go.Scatter(x=steps, y=attention_gradient, mode='lines+markers', name='Attention Entropy Gradient'))
    fig.update_layout(title='Entropy Gradient Over Time', xaxis_title='Generation Step', yaxis_title='Entropy Gradient')
    return fig


# Function to create rolling entropy figure
def create_rolling_entropy_figure(generation_results, window=5):
    logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
    attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]
    steps = list(range(1, len(generation_results['step_analyses']) + 1))

    rolling_logits = pd.Series(logits_entropies).rolling(window=window).mean().tolist()
    rolling_attention = pd.Series(attention_entropies).rolling(window=window).mean().tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=rolling_logits, mode='lines+markers', name='Rolling Logits Entropy'))
    fig.add_trace(go.Scatter(x=steps, y=rolling_attention, mode='lines+markers', name='Rolling Attention Entropy'))
    fig.update_layout(title=f'Rolling Entropy Over Time (Window = {window})', xaxis_title='Generation Step',
                      yaxis_title='Rolling Entropy')
    return fig


# Function to create model state distribution figure
def create_model_state_distribution_figure(generation_results):
    states = [step['model_state'] for step in generation_results['step_analyses']]
    state_counts = Counter(states)
    states_list = list(state_counts.keys())
    counts = list(state_counts.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(x=states_list, y=counts))
    fig.update_layout(title='Distribution of Model States', xaxis_title='Model State', yaxis_title='Frequency')
    return fig


# Function to create the Dash app
def create_app(generation_results, metadata):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1('Model Generation Metrics Visualization'),
        dcc.Tabs([
            dcc.Tab(label='Entropy Over Time', children=[
                dcc.Graph(figure=create_entropy_over_time_figure(generation_results))
            ]),
            dcc.Tab(label='Model States Over Time', children=[
                dcc.Graph(figure=create_model_states_figure(generation_results))
            ]),
            dcc.Tab(label='Entropy Distribution', children=[
                dcc.Graph(figure=create_entropy_distribution_figure(generation_results))
            ]),
            dcc.Tab(label='Surprisal Over Time', children=[
                dcc.Graph(figure=create_surprisal_over_time_figure(generation_results))
            ]),
            dcc.Tab(label='Entropy Correlation', children=[
                dcc.Graph(figure=create_entropy_correlation_figure(generation_results))
            ]),
            dcc.Tab(label='Entropy Gradient Over Time', children=[
                dcc.Graph(figure=create_entropy_gradient_figure(generation_results))
            ]),
            dcc.Tab(label='Rolling Entropy Over Time', children=[
                dcc.Graph(figure=create_rolling_entropy_figure(generation_results))
            ]),
            dcc.Tab(label='Model State Distribution', children=[
                dcc.Graph(figure=create_model_state_distribution_figure(generation_results))
            ]),
        ]),
    ])

    return app


if __name__ == '__main__':
    folder_name = '001/analysis_results_20241015_021216'

    generation_results = load_generation_results(folder_name)
    metadata = load_metadata(folder_name)

    app = create_app(generation_results, metadata)
    app.run_server(debug=True)
