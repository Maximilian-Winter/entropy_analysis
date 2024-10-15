import os
import json

import numpy as np
import pandas as pd
from collections import Counter

import dash
from dash import Dash, html, dcc, Output, Input, State

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer


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
    fig.add_trace(go.Scatter(
        x=steps,
        y=logits_entropies,
        mode='lines+markers',
        name='Logits Entropy',
        customdata=steps,
        hovertemplate='Step: %{customdata}<br>Logits Entropy: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=steps,
        y=attention_entropies,
        mode='lines+markers',
        name='Attention Entropy',
        customdata=steps,
        hovertemplate='Step: %{customdata}<br>Attention Entropy: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(title='Entropy Over Time', xaxis_title='Generation Step', yaxis_title='Entropy')
    return fig


# Function to create model states figure
def create_model_states_figure(generation_results):
    steps = list(range(1, len(generation_results['step_analyses']) + 1))
    states = [step['model_state'] for step in generation_results['step_analyses']]

    state_to_num = {
        'Very Uncertain': 0, 'Uncertain': 1, 'Slightly Uncertain': 2, 'Exploring': 3,
        'Balanced': 4, 'Focusing': 5, 'Confident': 6, 'Highly Confident': 7,
        'Overconfident': 8, 'Very Overconfident': 9
    }
    numeric_states = [state_to_num.get(state, -1) for state in states]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=numeric_states,
        mode='lines+markers',
        name='Model State',
        customdata=steps,
        hovertemplate='Step: %{customdata}<br>State: %{text}<extra></extra>',
        text=states
    ))
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
    fig.add_trace(go.Histogram(
        x=logits_entropies,
        nbinsx=20,
        name='Logits Entropy',
        customdata=list(range(1, len(logits_entropies) + 1)),
        hovertemplate='Step: %{customdata}<br>Logits Entropy: %{x:.2f}<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=attention_entropies,
        nbinsx=20,
        name='Attention Entropy',
        customdata=list(range(1, len(attention_entropies) + 1)),
        hovertemplate='Step: %{customdata}<br>Attention Entropy: %{x:.2f}<extra></extra>'
    ), row=1, col=2)
    fig.update_layout(title='Entropy Distribution', showlegend=False)
    return fig


# Function to create surprisal over time figure
def create_surprisal_over_time_figure(generation_results):
    steps = list(range(1, len(generation_results['step_analyses']) + 1))
    surprisal_values = [step['surprisal'] for step in generation_results['step_analyses']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=surprisal_values,
        mode='lines+markers',
        name='Surprisal',
        customdata=steps,
        hovertemplate='Step: %{customdata}<br>Surprisal: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(title='Surprisal Over Time', xaxis_title='Generation Step',
                      yaxis_title='Surprisal (Negative Log Probability)')
    return fig


# Function to create entropy correlation figure
def create_entropy_correlation_figure(generation_results):
    logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
    attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

    correlation = np.corrcoef(logits_entropies, attention_entropies)[0, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=logits_entropies,
        y=attention_entropies,
        mode='markers',
        name='Entropy Scatter',
        customdata=list(range(1, len(logits_entropies) + 1)),
        hovertemplate='Step: %{customdata}<br>Logits Entropy: %{x:.2f}<br>Attention Entropy: %{y:.2f}<extra></extra>'
    ))
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
    fig.add_trace(go.Scatter(
        x=steps,
        y=logits_gradient,
        mode='lines+markers',
        name='Logits Entropy Gradient',
        customdata=steps,
        hovertemplate='Step: %{customdata}<br>Logits Gradient: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=steps,
        y=attention_gradient,
        mode='lines+markers',
        name='Attention Entropy Gradient',
        customdata=steps,
        hovertemplate='Step: %{customdata}<br>Attention Gradient: %{y:.2f}<extra></extra>'
    ))
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
    fig.add_trace(go.Scatter(
        x=steps,
        y=rolling_logits,
        mode='lines+markers',
        name='Rolling Logits Entropy',
        customdata=steps,
        hovertemplate='Step: %{customdata}<br>Rolling Logits Entropy: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=steps,
        y=rolling_attention,
        mode='lines+markers',
        name='Rolling Attention Entropy',
        customdata=steps,
        hovertemplate='Step: %{customdata}<br>Rolling Attention Entropy: %{y:.2f}<extra></extra>'
    ))
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
    fig.add_trace(go.Bar(
        x=states_list,
        y=counts,
        customdata=states_list,
        hovertemplate='State: %{customdata}<br>Count: %{y}<extra></extra>'
    ))
    fig.update_layout(title='Distribution of Model States', xaxis_title='Model State', yaxis_title='Frequency')
    return fig


# Function to generate text spans with tokens
def generate_text_spans(generation_results, tokenizer):
    step_analyses = generation_results['step_analyses']
    generated_ids = generation_results['generated_ids']

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
    num_generated_tokens = len(step_analyses)
    tokens = tokens[-num_generated_tokens:]

    spans = []
    for idx, token in enumerate(tokens):
        display_token = tokenizer.convert_tokens_to_string([token])
        spans.append(html.Span(display_token, id={'type': 'token-span', 'index': idx}, style={'padding': '0 2px'}))
    return spans


# Function to create the Dash app
def create_app(generation_results, metadata, tokenizer):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1('Model Generation Metrics Visualization'),
        html.Div(id='generated-text', children=generate_text_spans(generation_results, tokenizer),
                 style={'font-size': '18px', 'line-height': '1.5'}),
        dcc.Store(id='hovered-step', data=-1),
        dcc.Tabs([
            dcc.Tab(label='Entropy Over Time', children=[
                dcc.Graph(id='entropy-over-time-graph', figure=create_entropy_over_time_figure(generation_results))
            ]),
            dcc.Tab(label='Model States Over Time', children=[
                dcc.Graph(id='model-states-graph', figure=create_model_states_figure(generation_results))
            ]),
            dcc.Tab(label='Entropy Distribution', children=[
                dcc.Graph(id='entropy-distribution-graph',
                          figure=create_entropy_distribution_figure(generation_results))
            ]),
            dcc.Tab(label='Surprisal Over Time', children=[
                dcc.Graph(id='surprisal-over-time-graph', figure=create_surprisal_over_time_figure(generation_results))
            ]),
            dcc.Tab(label='Entropy Correlation', children=[
                dcc.Graph(id='entropy-correlation-graph', figure=create_entropy_correlation_figure(generation_results))
            ]),
            dcc.Tab(label='Entropy Gradient Over Time', children=[
                dcc.Graph(id='entropy-gradient-graph', figure=create_entropy_gradient_figure(generation_results))
            ]),
            dcc.Tab(label='Rolling Entropy Over Time', children=[
                dcc.Graph(id='rolling-entropy-graph', figure=create_rolling_entropy_figure(generation_results))
            ]),
            dcc.Tab(label='Model State Distribution', children=[
                dcc.Graph(id='model-state-distribution-graph',
                          figure=create_model_state_distribution_figure(generation_results))
            ]),
        ]),
    ])

    # Callback to highlight the corresponding token when hovering over the graphs
    @app.callback(
        Output('generated-text', 'children'),
        [
            Input('entropy-over-time-graph', 'hoverData'),
            Input('model-states-graph', 'hoverData'),
            Input('surprisal-over-time-graph', 'hoverData'),
            Input('entropy-gradient-graph', 'hoverData'),
            Input('rolling-entropy-graph', 'hoverData'),
            Input('entropy-correlation-graph', 'hoverData'),
            Input('entropy-distribution-graph', 'hoverData'),
            Input('model-state-distribution-graph', 'hoverData'),
        ],
        State('generated-text', 'children')
    )
    def highlight_token(entropy_hover, model_states_hover, surprisal_hover, gradient_hover, rolling_hover,
                        correlation_hover, distribution_hover, state_distribution_hover, children):
        ctx = dash.callback_context

        if not ctx.triggered:
            return children

        hoverData = None
        for hover in [entropy_hover, model_states_hover, surprisal_hover, gradient_hover, rolling_hover,
                      correlation_hover, distribution_hover, state_distribution_hover]:
            if hover is not None:
                hoverData = hover
                break

        if hoverData is None:
            return children

        # Extract the step number from customdata
        point = hoverData['points'][0]
        if 'customdata' in point:
            token_index = int(point['customdata']) - 1  # Adjust for zero-based index
        else:
            # Fallback to old method if customdata is not available
            token_index = int(point.get('x', point.get('pointIndex', 0))) - 1

        # Reconstruct the children, updating the style
        new_children = []
        for idx, child in enumerate(children):
            if isinstance(child, dict):
                child_props = child['props']
                token_text = child_props['children']
            else:
                child_props = child.props
                token_text = child.children

            style = child_props.get('style', {}).copy()
            if idx == token_index:
                style['backgroundColor'] = 'yellow'
            else:
                style['backgroundColor'] = 'transparent'

            new_child = html.Span(token_text, style=style, id=child_props.get('id'))
            new_children.append(new_child)
        return new_children

    return app


if __name__ == '__main__':
    folder_name = '001/analysis_results_20241015_021216'

    generation_results = load_generation_results(folder_name)
    metadata = load_metadata(folder_name)

    app = create_app(generation_results, metadata, AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"))
    app.run_server(debug=True)