import datetime
import json
import os
from typing import Dict

import torch
from transformers import PreTrainedTokenizer


def create_timestamped_folder(parent_folder: str = None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{parent_folder}/analysis_results_{timestamp}" if parent_folder is not None else f"analysis_results_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_generation_results(generation_results: Dict, folder_name: str):
    results = {}
    for k, v in generation_results.items():
        if isinstance(v, torch.Tensor):
            v = v.tolist()
        results[k] = v
    with open(os.path.join(folder_name, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


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
