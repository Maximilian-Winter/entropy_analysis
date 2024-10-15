import datetime
import json
import os
from typing import Dict, List

import numpy as np
import torch
from transformers import PreTrainedTokenizer


def create_timestamped_folder(parent_folder: str = None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{parent_folder}/analysis_results_{timestamp}" if parent_folder is not None else f"analysis_results_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


class NumpyTorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_generation_results(generation_results: Dict, folder_name: str):
    with open(os.path.join(folder_name, 'generation_results.json'), 'w') as f:
        json.dump(generation_results, f, indent=2, cls=NumpyTorchEncoder)


def save_metadata(model_name, device, logits_entropy_thresholds, attn_entropy_thresholds, folder_name: str):
    metadata = {
        "model_name": model_name,
        "device": device,
        "logits_entropy_thresholds": logits_entropy_thresholds,
        "attn_entropy_thresholds": attn_entropy_thresholds,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(os.path.join(folder_name, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_html_report(generation_results: Dict, tokenizer: PreTrainedTokenizer, folder_name: str,
                         output_html_file: str):
    import os
    import json

    generated_ids = generation_results['generated_ids']
    step_analyses = generation_results['step_analyses']

    tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
    num_generated_tokens = len(step_analyses)
    tokens = tokens[-num_generated_tokens:]

    def create_html_tokens():
        html_tokens = []
        for idx, token in enumerate(tokens):
            step_analysis = step_analyses[idx]
            step_analysis['token_idx'] = idx
            data_attributes = ' '.join(
                [f'data-{key.replace("_", "-")}="{str(value).replace('"', '&quot;')}"' for key, value in
                 step_analysis.items()])
            display_token = tokenizer.convert_tokens_to_string([token])
            display_token = display_token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_token = f'<span id="token-{idx}" {data_attributes}>{display_token}</span>'
            html_tokens.append(html_token)
        return ''.join(html_tokens)

    html_content = create_html_tokens()

    # Prepare data for Plotly charts
    steps = list(range(1, len(step_analyses) + 1))
    logits_entropies = [step['logits_entropy'] for step in step_analyses]
    attention_entropies = [step['attention_entropy'] for step in step_analyses]
    surprisal_values = [step['surprisal'] for step in step_analyses]
    max_probabilities = [step['max_probability'] for step in step_analyses]
    model_states = [step['model_state'] for step in step_analyses]

    # Convert data to JSON for JavaScript
    chart_data = json.dumps({
        'steps': steps,
        'logits_entropies': logits_entropies,
        'attention_entropies': attention_entropies,
        'surprisal_values': surprisal_values,
        'max_probabilities': max_probabilities,
        'model_states': model_states
    })

    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Generated Text Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: #ffffff;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            .generated-text {{
                background-color: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 15px;
                margin-bottom: 20px;
                line-height: 1.8;
                max-height: 200px;
                overflow-y: auto;
            }}
            .tooltip {{
                position: absolute;
                background: #ffffff;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 10px;
                display: none;
                z-index: 1000;
                max-width: 300px;
                word-wrap: break-word;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            span {{
                position: relative;
                cursor: pointer;
                padding: 2px 0;
            }}
            span:hover {{
                background-color: #e0f7fa;
            }}
            .chart {{
                width: 100%;
                height: 400px;
                margin-bottom: 30px;
            }}
            .highlight {{
                background-color: #ffff00;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Generated Text Analysis</h1>
            
            <h2>Entropy Over Time</h2>
            <div id="generated-text-entropy" class="generated-text"></div>
            <div id="entropy-chart" class="chart"></div>
            <div id="tooltip-entropy" class="tooltip"></div>

            <h2>Surprisal Over Time</h2>
            <div id="generated-text-surprisal" class="generated-text"></div>
            <div id="surprisal-chart" class="chart"></div>
            <div id="tooltip-surprisal" class="tooltip"></div>

            <h2>Max Probability Over Time</h2>
            <div id="generated-text-max-probability" class="generated-text"></div>
            <div id="max-probability-chart" class="chart"></div>
            <div id="tooltip-max-probability" class="tooltip"></div>

            <h2>Model States Over Time</h2>
            <div id="generated-text-model-states" class="generated-text"></div>
            <div id="model-states-chart" class="chart"></div>
            <div id="tooltip-model-states" class="tooltip"></div>
        </div>
        <script>
        const chartData = {chart_data};

        function createGeneratedText(divId) {{
            const container = document.getElementById(divId);
            container.innerHTML = `{html_content}`;
        }}

        function highlightToken(index, divId) {{
            const container = document.getElementById(divId);
            container.querySelectorAll('span').forEach(span => span.classList.remove('highlight'));
            const tokenSpan = container.querySelector(`#token-${{index}}`);
            if (tokenSpan) {{
                tokenSpan.classList.add('highlight');
                tokenSpan.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            }}
        }}

        function createPlot(divId, xData, yData, name, title, yaxisTitle, textDivId, tooltipId, additionalLayout = {{}}) {{
            createGeneratedText(textDivId);

            const trace = {{
                x: xData,
                y: yData,
                type: 'scatter',
                mode: 'lines+markers',
                name: name,
                hoverinfo: 'x+y',
                line: {{shape: 'spline'}},
            }};

            const layout = {{
                title: title,
                xaxis: {{title: 'Generation Step'}},
                yaxis: {{title: yaxisTitle}},
                hovermode: 'closest',
                ...additionalLayout
            }};

            Plotly.newPlot(divId, [trace], layout);

            document.getElementById(divId).on('plotly_hover', function(data) {{
                const index = data.points[0].pointIndex;
                highlightToken(index, textDivId);
            }});

            const tooltip = document.getElementById(tooltipId);
            document.getElementById(textDivId).addEventListener('mouseover', function(e) {{
                if (e.target.tagName.toLowerCase() === 'span') {{
                    let dataAttributes = e.target.dataset;
                    let tooltipContent = '<ul>';
                    for (let key in dataAttributes) {{
                        tooltipContent += '<li><strong>' + key.replace(/-/g, ' ') + ':</strong> ' + dataAttributes[key] + '</li>';
                    }}
                    tooltipContent += '</ul>';
                    tooltip.innerHTML = tooltipContent;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.pageX + 15) + 'px';
                    tooltip.style.top = (e.pageY + 15) + 'px';
                }}
            }});
            document.getElementById(textDivId).addEventListener('mouseout', function(e) {{
                if (e.target.tagName.toLowerCase() === 'span') {{
                    tooltip.style.display = 'none';
                }}
            }});
        }}

        // Entropy Over Time
        createPlot('entropy-chart', chartData.steps, chartData.logits_entropies, 'Logits Entropy', 'Entropy Over Time', 'Entropy', 'generated-text-entropy', 'tooltip-entropy');
        Plotly.addTraces('entropy-chart', {{
            x: chartData.steps,
            y: chartData.attention_entropies,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Attention Entropy',
            line: {{shape: 'spline'}},
        }});

        // Surprisal Over Time
        createPlot('surprisal-chart', chartData.steps, chartData.surprisal_values, 'Surprisal', 'Surprisal Over Time', 'Surprisal', 'generated-text-surprisal', 'tooltip-surprisal');

        // Max Probability Over Time
        createPlot('max-probability-chart', chartData.steps, chartData.max_probabilities, 'Max Probability', 'Max Probability Over Time', 'Max Probability', 'generated-text-max-probability', 'tooltip-max-probability');

        // Model States Over Time
        createPlot('model-states-chart', chartData.steps, chartData.model_states, 'Model State', 'Model States Over Time', 'Model State', 'generated-text-model-states', 'tooltip-model-states', {{
            yaxis: {{
                tickvals: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                ticktext: ['Very Uncertain', 'Uncertain', 'Slightly Uncertain', 'Exploring', 'Balanced', 'Focusing', 'Confident', 'Highly Confident', 'Overconfident', 'Very Overconfident']
            }}
        }});
        </script>
    </body>
    </html>
    """

    with open(os.path.join(folder_name, output_html_file), 'w', encoding='utf-8') as f:
        f.write(html_page)


def new_generate_html_report(generation_results: Dict, tokenizer: PreTrainedTokenizer, folder_name: str,
                             output_html_file: str):
    import os
    import json
    import numpy as np

    generated_ids = generation_results['generated_ids']
    step_analyses = generation_results['step_analyses']

    tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
    num_generated_tokens = len(step_analyses)
    tokens = tokens[-num_generated_tokens:]

    def create_html_tokens():
        html_tokens = []
        for idx, token in enumerate(tokens):
            step_analysis = step_analyses[idx]
            step_analysis['token_idx'] = idx
            data_attributes = ' '.join(
                [f'data-{key.replace("_", "-")}="{str(value).replace('"', '&quot;')}"' for key, value in
                 step_analysis.items() if not isinstance(value, (list, np.ndarray))])
            display_token = tokenizer.convert_tokens_to_string([token])
            display_token = display_token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_token = f'<span id="token-{idx}" {data_attributes}>{display_token}</span>'
            html_tokens.append(html_token)
        return ''.join(html_tokens)

    html_content = create_html_tokens()

    # Prepare data for Plotly charts
    # Prepare data for Plotly charts
    steps = list(range(1, len(step_analyses) + 1))
    logits_entropies = [step['logits_entropy'] for step in step_analyses]
    attention_entropies = [step['attention_entropy'] for step in step_analyses]
    surprisal_values = [step['surprisal'] for step in step_analyses]
    max_probabilities = [step['max_probability'] for step in step_analyses]

    # New data
    multi_dim_states = [step['multi_dim_state'].tolist() for step in step_analyses]
    entropy_change_rates = [step.get('entropy_change_rate', 0) for step in step_analyses]
    attention_flows = [step['attention_flow'] for step in generation_results['step_analyses']]

    # Process attention flows
    reshaped_flows = []
    for flow in attention_flows:
        if flow.shape[0] == 1:
            seq_len = flow.shape[1]
            reshaped_flow = np.zeros((seq_len, seq_len))
            reshaped_flow[-1, :] = flow[0, :]
        else:
            reshaped_flow = flow
        reshaped_flows.append(reshaped_flow)

    max_seq_len = max(flow.shape[0] for flow in reshaped_flows)
    padded_flows = []
    for flow in reshaped_flows:
        if flow.shape[0] < max_seq_len:
            padded_flow = np.pad(flow, ((0, max_seq_len - flow.shape[0]), (0, max_seq_len - flow.shape[1])), mode='constant')
        else:
            padded_flow = flow[-max_seq_len:, -max_seq_len:]
        padded_flows.append(padded_flow)

    stacked_flows = np.stack(padded_flows)
    attention_flows = np.mean(stacked_flows, axis=0)

    # Convert data to JSON for JavaScript
    chart_data = json.dumps({
        'steps': steps,
        'logits_entropies': logits_entropies,
        'attention_entropies': attention_entropies,
        'surprisal_values': surprisal_values,
        'max_probabilities': max_probabilities,
        'multi_dim_states': multi_dim_states,
        'entropy_change_rates': entropy_change_rates,
        'attention_flows': attention_flows.tolist()
    }, cls=NumpyTorchEncoder)

    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Generated Text Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: #ffffff;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            .generated-text {{
                background-color: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 15px;
                margin-bottom: 20px;
                line-height: 1.8;
                max-height: 200px;
                overflow-y: auto;
            }}
            .tooltip {{
                position: absolute;
                background: #ffffff;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 10px;
                display: none;
                z-index: 1000;
                max-width: 300px;
                word-wrap: break-word;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            span {{
                position: relative;
                cursor: pointer;
                padding: 2px 0;
            }}
            span:hover {{
                background-color: #e0f7fa;
            }}
            .chart {{
                width: 100%;
                height: 400px;
                margin-bottom: 30px;
            }}
            .highlight {{
                background-color: #ffff00;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Generated Text Analysis</h1>
            
            <h2>Entropy Over Time</h2>
            <div id="generated-text-entropy" class="generated-text"></div>
            <div id="entropy-chart" class="chart"></div>
            <div id="tooltip-entropy" class="tooltip"></div>

            <h2>Surprisal Over Time</h2>
            <div id="generated-text-surprisal" class="generated-text"></div>
            <div id="surprisal-chart" class="chart"></div>
            <div id="tooltip-surprisal" class="tooltip"></div>

            <h2>Max Probability Over Time</h2>
            <div id="generated-text-max-probability" class="generated-text"></div>
            <div id="max-probability-chart" class="chart"></div>
            <div id="tooltip-max-probability" class="tooltip"></div>

            <h2>Multi-Dimensional State Analysis</h2>
            <div id="generated-text-multi-dim" class="generated-text"></div>
            <div id="multi-dim-chart" class="chart"></div>
            <div id="tooltip-multi-dim" class="tooltip"></div>

            <h2>Entropy Change Rate</h2>
            <div id="generated-text-entropy-change" class="generated-text"></div>
            <div id="entropy-change-chart" class="chart"></div>
            <div id="tooltip-entropy-change" class="tooltip"></div>

            <h2>Attention Flow</h2>
            <div id="generated-text-attention-flow" class="generated-text"></div>
            <div id="attention-flow-chart" class="chart"></div>
            <div id="tooltip-attention-flow" class="tooltip"></div>
        </div>
        <script>
        const chartData = {chart_data};

        function createGeneratedText(divId) {{
            const container = document.getElementById(divId);
            container.innerHTML = `{html_content}`;
        }}

        function highlightToken(index, divId) {{
            const container = document.getElementById(divId);
            container.querySelectorAll('span').forEach(span => span.classList.remove('highlight'));
            const tokenSpan = container.querySelector(`#token-${{index}}`);
            if (tokenSpan) {{
                tokenSpan.classList.add('highlight');
                tokenSpan.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            }}
        }}

        function createPlot(divId, xData, yData, name, title, yaxisTitle, textDivId, tooltipId, additionalLayout = {{}}) {{
            createGeneratedText(textDivId);

            const trace = {{
                x: xData,
                y: yData,
                type: 'scatter',
                mode: 'lines+markers',
                name: name,
                hoverinfo: 'x+y',
                line: {{shape: 'spline'}},
            }};

            const layout = {{
                title: title,
                xaxis: {{title: 'Generation Step'}},
                yaxis: {{title: yaxisTitle}},
                hovermode: 'closest',
                ...additionalLayout
            }};

            Plotly.newPlot(divId, [trace], layout);

            document.getElementById(divId).on('plotly_hover', function(data) {{
                const index = data.points[0].pointIndex;
                highlightToken(index, textDivId);
            }});

            const tooltip = document.getElementById(tooltipId);
            document.getElementById(textDivId).addEventListener('mouseover', function(e) {{
                if (e.target.tagName.toLowerCase() === 'span') {{
                    let dataAttributes = e.target.dataset;
                    let tooltipContent = '<ul>';
                    for (let key in dataAttributes) {{
                        tooltipContent += '<li><strong>' + key.replace(/-/g, ' ') + ':</strong> ' + dataAttributes[key] + '</li>';
                    }}
                    tooltipContent += '</ul>';
                    tooltip.innerHTML = tooltipContent;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.pageX + 15) + 'px';
                    tooltip.style.top = (e.pageY + 15) + 'px';
                }}
            }});
            document.getElementById(textDivId).addEventListener('mouseout', function(e) {{
                if (e.target.tagName.toLowerCase() === 'span') {{
                    tooltip.style.display = 'none';
                }}
            }});
        }}

        // Entropy Over Time
        createPlot('entropy-chart', chartData.steps, chartData.logits_entropies, 'Logits Entropy', 'Entropy Over Time', 'Entropy', 'generated-text-entropy', 'tooltip-entropy');
        Plotly.addTraces('entropy-chart', {{
            x: chartData.steps,
            y: chartData.attention_entropies,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Attention Entropy',
            line: {{shape: 'spline'}},
        }});
        
        // Entropy Change Rate
        createPlot('entropy-change-chart', chartData.steps, chartData.entropy_change_rates, 'Entropy Change Rate', 'Entropy Change Rate Over Time', 'Change Rate', 'generated-text-entropy-change', 'tooltip-entropy-change');

        // Surprisal Over Time
        createPlot('surprisal-chart', chartData.steps, chartData.surprisal_values, 'Surprisal', 'Surprisal Over Time', 'Surprisal', 'generated-text-surprisal', 'tooltip-surprisal');

        // Max Probability Over Time
        createPlot('max-probability-chart', chartData.steps, chartData.max_probabilities, 'Max Probability', 'Max Probability Over Time', 'Max Probability', 'generated-text-max-probability', 'tooltip-max-probability');

        // Multi-Dimensional State Analysis
        const multiDimTraces = [
            {{x: chartData.steps, y: chartData.multi_dim_states.map(state => state[0]), name: 'Logits Entropy', type: 'scatter', mode: 'lines+markers'}},
            {{x: chartData.steps, y: chartData.multi_dim_states.map(state => state[1]), name: 'Attention Entropy', type: 'scatter', mode: 'lines+markers'}},
            {{x: chartData.steps, y: chartData.multi_dim_states.map(state => state[2]), name: 'Surprisal', type: 'scatter', mode: 'lines+markers'}},
            {{x: chartData.steps, y: chartData.multi_dim_states.map(state => state[3]), name: 'Max Probability', type: 'scatter', mode: 'lines+markers'}}
        ];
        Plotly.newPlot('multi-dim-chart', multiDimTraces, {{
            title: 'Multi-Dimensional State Analysis', 
            xaxis: {{title: 'Generation Step'}}, 
            yaxis: {{title: 'Value'}},
            hovermode: 'closest'
        }});

        // Add hover event for multi-dimensional state chart
        document.getElementById('multi-dim-chart').on('plotly_hover', function(data) {{
            const index = data.points[0].pointIndex;
            highlightToken(index, 'generated-text-multi-dim');
            
            // Update tooltip with all dimension values
            const tooltip = document.getElementById('tooltip-multi-dim');
            const state = chartData.multi_dim_states[index];
            tooltip.innerHTML = `
                <ul>
                    <li><strong>Logits Entropy:</strong> ${{state[0].toFixed(4)}}</li>
                    <li><strong>Attention Entropy:</strong> ${{state[1].toFixed(4)}}</li>
                    <li><strong>Surprisal:</strong> ${{state[2].toFixed(4)}}</li>
                    <li><strong>Max Probability:</strong> ${{state[3].toFixed(4)}}</li>
                </ul>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY + 15) + 'px';
        }});

        // Hide tooltip when not hovering over a point
        document.getElementById('multi-dim-chart').on('plotly_unhover', function() {{
            document.getElementById('tooltip-multi-dim').style.display = 'none';
        }});

        // Add this to create the generated text for multi-dimensional state analysis
        createGeneratedText('generated-text-multi-dim');

        // Attention Flow
        createGeneratedText('generated-text-attention-flow');
        const attentionFlowData = [{{
            z: chartData.attention_flows,
            type: 'heatmap',
            colorscale: 'Viridis'
        }}];
        Plotly.newPlot('attention-flow-chart', attentionFlowData, {{
            title: 'Aggregated Attention Flow',
            xaxis: {{title: 'Source Token'}},
            yaxis: {{title: 'Target Token', autorange: 'reversed'}}
        }});
        
        </script>
    </body>
    </html>
    """

    with open(os.path.join(folder_name, output_html_file), 'w', encoding='utf-8') as f:
        f.write(html_page)


def new_new_generate_html_report(generation_results_list: List[Dict], tokenizer: PreTrainedTokenizer, folder_name: str,
                                 output_html_file: str):
    import os
    import json
    import numpy as np

    def process_single_result(generation_result):
        generated_ids = generation_result['generated_ids']
        step_analyses = generation_result['step_analyses']

        tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
        num_generated_tokens = len(step_analyses)
        tokens = tokens[-num_generated_tokens:]

        def create_html_tokens():
            html_tokens = []
            for idx, token in enumerate(tokens):
                step_analysis = step_analyses[idx]
                step_analysis['token_idx'] = idx
                data_attributes = ' '.join(
                    [f'data-{key.replace("_", "-")}="{str(value).replace('"', '&quot;')}"' for key, value in
                     step_analysis.items() if not isinstance(value, (list, np.ndarray))])
                display_token = tokenizer.convert_tokens_to_string([token])
                display_token = display_token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html_token = f'<span data-token-idx="{idx}" {data_attributes}>{display_token}</span>'
                html_tokens.append(html_token)
            return ''.join(html_tokens)

        html_content = create_html_tokens()

        steps = list(range(1, len(step_analyses) + 1))
        logits_entropies = [step['logits_entropy'] for step in step_analyses]
        attention_entropies = [step['attention_entropy'] for step in step_analyses]
        surprisal_values = [step['surprisal'] for step in step_analyses]
        max_probabilities = [step['max_probability'] for step in step_analyses]
        multi_dim_states = [step['multi_dim_state'].tolist() for step in step_analyses]
        entropy_change_rates = [step.get('entropy_change_rate', 0) for step in step_analyses]
        attention_flows = [step['attention_flow'] for step in step_analyses]

        # Process attention flows
        reshaped_flows = []
        for flow in attention_flows:
            if flow.shape[0] == 1:
                seq_len = flow.shape[1]
                reshaped_flow = np.zeros((seq_len, seq_len))
                reshaped_flow[-1, :] = flow[0, :]
            else:
                reshaped_flow = flow
            reshaped_flows.append(reshaped_flow)

        max_seq_len = max(flow.shape[0] for flow in reshaped_flows)
        padded_flows = []
        for flow in reshaped_flows:
            if flow.shape[0] < max_seq_len:
                padded_flow = np.pad(flow, ((0, max_seq_len - flow.shape[0]), (0, max_seq_len - flow.shape[1])), mode='constant')
            else:
                padded_flow = flow[-max_seq_len:, -max_seq_len:]
            padded_flows.append(padded_flow)

        stacked_flows = np.stack(padded_flows)
        attention_flows = np.mean(stacked_flows, axis=0)

        return {
            'html_content': html_content,
            'steps': steps,
            'logits_entropies': logits_entropies,
            'attention_entropies': attention_entropies,
            'surprisal_values': surprisal_values,
            'max_probabilities': max_probabilities,
            'multi_dim_states': multi_dim_states,
            'entropy_change_rates': entropy_change_rates,
            'attention_flows': attention_flows.tolist()
        }

    processed_results = [process_single_result(result) for result in generation_results_list]

    # Convert data to JSON for JavaScript
    chart_data = json.dumps(processed_results, cls=NumpyTorchEncoder)

    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Generated Text Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: #ffffff;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            .generated-text {{
                background-color: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 15px;
                margin-bottom: 20px;
                line-height: 1.8;
                max-height: 200px;
                overflow-y: auto;
            }}
            .tooltip {{
                position: absolute;
                background: #ffffff;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 10px;
                display: none;
                z-index: 1000;
                max-width: 300px;
                word-wrap: break-word;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            span {{
                position: relative;
                cursor: pointer;
                padding: 2px 0;
            }}
            span:hover {{
                background-color: #e0f7fa;
            }}
            .chart {{
                width: 100%;
                height: 400px;
                margin-bottom: 30px;
            }}
            .highlight {{
                background-color: #ffff00;
            }}
            #result-selector {{
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Generated Text Analysis</h1>
            
            <div id="result-selector">
                <label for="result-select">Select Generation Result:</label>
                <select id="result-select"></select>
            </div>

            <h2>Entropy Over Time</h2>
            <div id="generated-text-entropy" class="generated-text"></div>
            <div id="entropy-chart" class="chart"></div>
            <div id="tooltip-entropy" class="tooltip"></div>

            <h2>Surprisal Over Time</h2>
            <div id="generated-text-surprisal" class="generated-text"></div>
            <div id="surprisal-chart" class="chart"></div>
            <div id="tooltip-surprisal" class="tooltip"></div>

            <h2>Max Probability Over Time</h2>
            <div id="generated-text-max-probability" class="generated-text"></div>
            <div id="max-probability-chart" class="chart"></div>
            <div id="tooltip-max-probability" class="tooltip"></div>

            <h2>Multi-Dimensional State Analysis</h2>
            <div id="generated-text-multi-dim" class="generated-text"></div>
            <div id="multi-dim-chart" class="chart"></div>
            <div id="tooltip-multi-dim" class="tooltip"></div>

            <h2>Entropy Change Rate</h2>
            <div id="generated-text-entropy-change" class="generated-text"></div>
            <div id="entropy-change-chart" class="chart"></div>
            <div id="tooltip-entropy-change" class="tooltip"></div>

            <h2>Attention Flow</h2>
            <div id="generated-text-attention-flow" class="generated-text"></div>
            <div id="attention-flow-chart" class="chart"></div>
            <div id="tooltip-attention-flow" class="tooltip"></div>

        </div>
        <script>
        const chartData = {chart_data};

        function createGeneratedText(divId, html_content) {{
            const container = document.getElementById(divId);
            container.innerHTML = html_content;
        }}

        function highlightToken(index, divId) {{
            const container = document.getElementById(divId);
            container.querySelectorAll('span').forEach(span => span.classList.remove('highlight'));
            const tokenSpan = container.querySelector(`span[data-token-idx="${{index}}"]`);
            if (tokenSpan) {{
                tokenSpan.classList.add('highlight');
                tokenSpan.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            }}
        }}

        function createPlot(divId, xData, yData, name, title, yaxisTitle, textDivId, tooltipId, additionalLayout = {{}}) {{
            const trace = {{
                x: xData,
                y: yData,
                type: 'scatter',
                mode: 'lines+markers',
                name: name,
                hoverinfo: 'x+y',
                line: {{shape: 'spline'}},
            }};

            const layout = {{
                title: title,
                xaxis: {{title: 'Generation Step'}},
                yaxis: {{title: yaxisTitle}},
                hovermode: 'closest',
                ...additionalLayout
            }};

            Plotly.newPlot(divId, [trace], layout);

            document.getElementById(divId).on('plotly_hover', function(data) {{
                const index = data.points[0].pointIndex;
                highlightToken(index, textDivId);
            }});

            const tooltip = document.getElementById(tooltipId);
            document.getElementById(textDivId).addEventListener('mouseover', function(e) {{
                if (e.target.tagName.toLowerCase() === 'span') {{
                    let dataAttributes = e.target.dataset;
                    let tooltipContent = '<ul>';
                    for (let key in dataAttributes) {{
                        tooltipContent += '<li><strong>' + key.replace(/-/g, ' ') + ':</strong> ' + dataAttributes[key] + '</li>';
                    }}
                    tooltipContent += '</ul>';
                    tooltip.innerHTML = tooltipContent;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.pageX + 15) + 'px';
                    tooltip.style.top = (e.pageY + 15) + 'px';
                }}
            }});
            document.getElementById(textDivId).addEventListener('mouseout', function(e) {{
                if (e.target.tagName.toLowerCase() === 'span') {{
                    tooltip.style.display = 'none';
                }}
            }});
        }}

        function updateCharts(resultIndex) {{
            const data = chartData[resultIndex];

            createGeneratedText('generated-text-entropy', data.html_content);
            createGeneratedText('generated-text-surprisal', data.html_content);
            createGeneratedText('generated-text-max-probability', data.html_content);
            createGeneratedText('generated-text-multi-dim', data.html_content);
            createGeneratedText('generated-text-entropy-change', data.html_content);
            createGeneratedText('generated-text-attention-flow', data.html_content);

            // Entropy Over Time
            createPlot('entropy-chart', data.steps, data.logits_entropies, 'Logits Entropy', 'Entropy Over Time', 'Entropy', 'generated-text-entropy', 'tooltip-entropy');
            Plotly.addTraces('entropy-chart', {{
                x: data.steps,
                y: data.attention_entropies,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Attention Entropy',
            }});

            // Entropy Change Rate
            createPlot('entropy-change-chart', data.steps, data.entropy_change_rates, 'Entropy Change Rate', 'Entropy Change Rate Over Time', 'Change Rate', 'generated-text-entropy-change', 'tooltip-entropy-change');

            // Surprisal Over Time
            createPlot('surprisal-chart', data.steps, data.surprisal_values, 'Surprisal', 'Surprisal Over Time', 'Surprisal', 'generated-text-surprisal', 'tooltip-surprisal');

            // Max Probability Over Time
            createPlot('max-probability-chart', data.steps, data.max_probabilities, 'Max Probability', 'Max Probability Over Time', 'Max Probability', 'generated-text-max-probability', 'tooltip-max-probability');

            // Multi-Dimensional State Analysis
            const multiDimTraces = [
                {{x: data.steps, y: data.multi_dim_states.map(state => state[0]), name: 'Logits Entropy', type: 'scatter', mode: 'lines+markers'}},
                {{x: data.steps, y: data.multi_dim_states.map(state => state[1]), name: 'Attention Entropy', type: 'scatter', mode: 'lines+markers'}},
                {{x: data.steps, y: data.multi_dim_states.map(state => state[2]), name: 'Surprisal', type: 'scatter', mode: 'lines+markers'}},
                {{x: data.steps, y: data.multi_dim_states.map(state => state[3]), name: 'Max Probability', type: 'scatter', mode: 'lines+markers'}}
            ];
            Plotly.newPlot('multi-dim-chart', multiDimTraces, {{
                title: 'Multi-Dimensional State Analysis', 
                xaxis: {{title: 'Generation Step'}}, 
                yaxis: {{title: 'Value'}},
                hovermode: 'closest'
            }});

            document.getElementById('multi-dim-chart').on('plotly_hover', function(data) {{
                const index = data.points[0].pointIndex;
                highlightToken(index, 'generated-text-multi-dim');
            }});

            // Attention Flow
            createGeneratedText('generated-text-attention-flow', data.html_content);
            const attentionFlowData = [{{
                z: data.attention_flows,
                type: 'heatmap',
                colorscale: 'Viridis'
            }}];
            Plotly.newPlot('attention-flow-chart', attentionFlowData, {{
                title: 'Aggregated Attention Flow',
                xaxis: {{title: 'Source Token'}},
                yaxis: {{title: 'Target Token', autorange: 'reversed'}}
            }});
        }}

        // Initialize result selector
        const resultSelect = document.getElementById('result-select');
        chartData.forEach((_, index) => {{
            const option = document.createElement('option');
            option.value = index;
            option.text = `Generation Result ${{index + 1}}`;
            resultSelect.appendChild(option);
        }});

        resultSelect.addEventListener('change', function() {{
            updateCharts(this.value);
        }});

        // Initial chart creation
        updateCharts(0);
        
        </script>
    </body>
    </html>
    """

    with open(os.path.join(folder_name, output_html_file), 'w', encoding='utf-8') as f:
        f.write(html_page)