import os
from copy import copy

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import logging

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

from utils import new_new_generate_html_report
from utils import create_timestamped_folder, save_generation_results, save_metadata, new_generate_html_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure non-deterministic algorithms are used
torch.use_deterministic_algorithms(False)


class ConfigurableAnalysis:
    """Configuration class for enabling or disabling specific analyses."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled


class EntropyAnalysisConfig:
    """Configuration for the entire entropy analysis."""

    def __init__(self):
        self.logits_entropy = ConfigurableAnalysis()
        self.attention_entropy = ConfigurableAnalysis()
        self.surprisal = ConfigurableAnalysis()
        self.max_probability = ConfigurableAnalysis()


class BaseEntropyAnalysisWrapper(ABC):
    """Abstract base class for entropy analysis wrappers."""

    def __init__(self, model_name: str, device: str = "cuda", config: Optional[EntropyAnalysisConfig] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model = self._load_model(model_name, device)
        self.model_name = model_name
        self.config = config or EntropyAnalysisConfig()

    @abstractmethod
    def _load_model(self, model_name: str, device: str):
        pass

    @staticmethod
    def calculate_entropy(probs: np.ndarray) -> float:
        """Calculate the entropy of a probability distribution."""
        probs = probs[probs > 0]
        return entropy(probs, base=2)

    @staticmethod
    def calculate_sequence_entropy(probs: torch.Tensor) -> List[float]:
        """Calculate entropy for each token in a sequence."""
        return [BaseEntropyAnalysisWrapper.calculate_entropy(token_probs.cpu().numpy()) for token_probs in probs]

    @staticmethod
    def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0,
                              filter_value: float = -float('Inf')) -> torch.Tensor:
        """
        Filter logits using top-k and/or top-p (nucleus) filtering.

        Args:
            logits: Logits distribution shape (batch_size, vocab_size).
            top_k: Keep only top k tokens with highest probability.
            top_p: Keep the top tokens with cumulative probability >= top_p.
            filter_value: The value to replace filtered logits with.

        Returns:
            Filtered logits tensor.
        """
        # Clone logits to avoid modifying the original tensor
        logits = logits.clone()

        # Top-k filtering
        if top_k > 0:
            top_k = min(max(top_k, 1), logits.size(-1))
            kth_values = torch.topk(logits, top_k)[0][..., -1, None]
            indices_to_remove = logits < kth_values
            logits = logits.masked_fill(indices_to_remove, filter_value)

        # Top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter sorted tensors to original indexing
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

            logits = logits.masked_fill(indices_to_remove, filter_value)

        return logits

    def generate_and_analyze(self, input_text: str, max_length: int = 50, temperature: float = 1.0,
                             top_k: int = 0, top_p: float = 0.0) -> Dict:
        """
        Generate text from the input and analyze model state at each step.

        Args:
            input_text: The input text to start generation.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.

        Returns:
            A dictionary containing generation results and analyses.
        """
        self.model.eval()
        generated_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to(self.device)
        past_key_values = None
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"Input Text: {input_text}")

        generation_results = []
        generated_text = input_text
        previous_logits = None

        for step in range(max_length):
            try:
                if past_key_values is not None:
                    input_ids_step = generated_ids[:, -1:]  # Last generated token
                else:
                    input_ids_step = generated_ids

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids_step,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                        output_hidden_states=True
                    )

                past_key_values = outputs.past_key_values
                logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
                attentions = outputs.attentions
                hidden_states = outputs.hidden_states

                # Sampling
                next_token_id = self.sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                # Store the generated token ID
                generated_token_id = next_token_id.item()

                # Analyze step
                step_analysis = self.analyze_step(
                    logits,
                    attentions,
                    hidden_states,
                    previous_logits=previous_logits,
                    generated_token_id=generated_token_id
                )
                generation_results.append(step_analysis)

                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

                generated_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=False)
                generated_text += generated_token

                logger.info(f"Step {step + 1} - Generated Token: {generated_token}")
                # Print step analysis if needed
                # logger.debug(f"Step Analysis: {step_analysis}")
                logger.info("-" * 50)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    logger.info("End of sequence reached.")
                    break

                # Update previous logits
                previous_logits = logits

            except Exception as e:
                logger.error(f"Error in generation step {step + 1}: {e}", exc_info=True)
                break

        logger.info("Final Generated Text:")
        logger.info(generated_text)

        return {
            'generated_ids': generated_ids,
            'generated_text': generated_text,
            'step_analyses': generation_results
        }

    def analyze_step(self, logits: torch.Tensor, attentions: List[torch.Tensor],
                     hidden_states: List[torch.Tensor], previous_logits: Optional[torch.Tensor],
                     generated_token_id: int) -> Dict:
        step_analysis = {}
        logits_entropy = None
        # Logits entropy
        if self.config.logits_entropy.enabled:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            logits_entropy = self.calculate_entropy(probs[0].cpu().numpy())
            step_analysis['logits_entropy'] = logits_entropy

            # Max probability (model's confidence)
            if self.config.max_probability.enabled:
                max_prob = torch.max(probs[0]).item()
                step_analysis['max_probability'] = max_prob

            # Surprisal (negative log probability of the generated token)
            if self.config.surprisal.enabled:
                token_prob = probs[0, generated_token_id].item()
                surprisal = -np.log2(token_prob + 1e-12)
                step_analysis['surprisal'] = surprisal

        # Attention entropy

        if self.config.attention_entropy.enabled:
            last_layer_attentions = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
            head_entropies = []
            for head_attn_weights in last_layer_attentions:  # Iterate over heads
                attn_weights = head_attn_weights[-1, :]  # Get attention for the last token
                entropy_value = self.calculate_entropy(attn_weights.cpu().numpy())
                head_entropies.append(entropy_value)
            attention_entropy = np.mean(head_entropies)
            step_analysis['attention_entropy'] = attention_entropy

        # Multi-dimensional state
        step_analysis['multi_dim_state'] = np.array([
            step_analysis.get('logits_entropy', 0),
            step_analysis.get('attention_entropy', 0),
            step_analysis.get('surprisal', 0),
            step_analysis.get('max_probability', 0)
        ])

        # Entropy change rate (if previous_logits is available)
        if previous_logits is not None and logits_entropy is not None:
            prev_probs = F.softmax(previous_logits[:, -1, :], dim=-1)
            prev_entropy = self.calculate_entropy(prev_probs[0].cpu().numpy())
            entropy_change_rate = logits_entropy - prev_entropy
            step_analysis['entropy_change_rate'] = entropy_change_rate

        if self.config.attention_entropy.enabled:
            last_layer_attentions = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
            if len(last_layer_attentions.shape) == 3:
                attention_flow = last_layer_attentions.mean(dim=0).cpu().numpy()
            else:
                attention_flow = last_layer_attentions.unsqueeze(0).cpu().numpy()
            step_analysis['attention_flow'] = attention_flow
        return step_analysis

    def visualize_multi_dim_state(self, generation_results: Dict, folder_name: str):
        states = np.array([step['multi_dim_state'] for step in generation_results['step_analyses']])

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Multi-dimensional State Analysis')

        metrics = ['Logits Entropy', 'Attention Entropy', 'Surprisal', 'Max Probability']

        for i, (ax, metric) in enumerate(zip(axs.ravel(), metrics)):
            ax.plot(states[:, i])
            ax.set_title(metric)
            ax.set_xlabel('Generation Step')
            ax.set_ylabel('Value')

        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'multi_dim_state.png'))
        plt.close()

    def visualize_entropy_change_rate(self, generation_results: Dict, folder_name: str):
        change_rates = [step.get('entropy_change_rate', 0) for step in generation_results['step_analyses']]

        plt.figure(figsize=(12, 6))
        plt.plot(change_rates)
        plt.title('Entropy Change Rate')
        plt.xlabel('Generation Step')
        plt.ylabel('Change Rate')
        plt.savefig(os.path.join(folder_name, 'entropy_change_rate.png'))
        plt.close()

    def sample_next_token(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                          top_p: float = 0.0) -> torch.Tensor:
        """
        Sample the next token from the logits.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size].
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.

        Returns:
            Tensor containing the next token ID.
        """
        logits = logits[:, -1, :] / temperature
        filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        return next_token_id

    def visualize_entropy_over_time(self, generation_results: Dict, folder_name: str):
        if not (self.config.logits_entropy.enabled and self.config.attention_entropy.enabled):
            logger.warning("Logits and attention entropy visualization is not enabled in the configuration.")
            return

        steps = range(1, len(generation_results['step_analyses']) + 1)
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        plt.figure(figsize=(12, 6))
        plt.plot(steps, logits_entropies, label='Logits Entropy', marker='o')
        plt.plot(steps, attention_entropies, label='Attention Entropy', marker='s')
        plt.xlabel('Generation Step')
        plt.ylabel('Entropy')
        plt.title('Entropy Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_name, 'entropy_over_time.png'))
        plt.close()

    def visualize_attention_flow(self, generation_results: Dict, folder_name: str):
        attention_flows = [step['attention_flow'] for step in generation_results['step_analyses']]

        # Initialize an empty list to store reshaped attention flows
        reshaped_flows = []

        for flow in attention_flows:
            if flow.shape[0] == 1:
                # If the flow is (1, X), reshape it to (X, X)
                seq_len = flow.shape[1]
                reshaped_flow = np.zeros((seq_len, seq_len))
                reshaped_flow[-1, :] = flow[0, :]  # Put the attention weights in the last row
            else:
                # If the flow is already (X, X), use it as is
                reshaped_flow = flow
            reshaped_flows.append(reshaped_flow)

        # Determine the maximum sequence length
        max_seq_len = max(flow.shape[0] for flow in reshaped_flows)

        # Pad or trim attention flows to a consistent shape
        padded_flows = []
        for flow in reshaped_flows:
            if flow.shape[0] < max_seq_len:
                # Pad with zeros
                padded_flow = np.pad(flow, ((0, max_seq_len - flow.shape[0]), (0, max_seq_len - flow.shape[1])),
                                     mode='constant')
            else:
                # Use only the last max_seq_len tokens
                padded_flow = flow[-max_seq_len:, -max_seq_len:]
            padded_flows.append(padded_flow)

        # Stack all padded flows
        stacked_flows = np.stack(padded_flows)

        # Aggregate attention flows
        aggregated_flow = np.mean(stacked_flows, axis=0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(aggregated_flow, cmap='viridis')
        plt.title('Aggregated Attention Flow')
        plt.xlabel('Source Token')
        plt.ylabel('Target Token')
        plt.savefig(os.path.join(folder_name, 'attention_flow.png'))
        plt.close()

    def visualize_entropy_distribution(self, generation_results: Dict, folder_name: str):
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(logits_entropies, kde=True)
        plt.title('Distribution of Logits Entropy')
        plt.xlabel('Entropy')

        plt.subplot(1, 2, 2)
        sns.histplot(attention_entropies, kde=True)
        plt.title('Distribution of Attention Entropy')
        plt.xlabel('Entropy')

        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'entropy_distribution.png'))
        plt.close()

    def rolling_entropy(self, entropies: List[float], window: int = 5):
        return pd.Series(entropies).rolling(window=window).mean().tolist()

    def entropy_gradient(self, entropies: List[float]):
        if len(entropies) < 2:
            return []  # Return an empty list if there's not enough data for gradient calculation
        return np.gradient(entropies)

    def visualize_rolling_entropy(self, generation_results: Dict, folder_name: str, window: int = 5):
        steps = range(1, len(generation_results['step_analyses']) + 1)
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        rolling_logits = self.rolling_entropy(logits_entropies, window)
        rolling_attention = self.rolling_entropy(attention_entropies, window)

        plt.figure(figsize=(12, 6))
        plt.plot(steps[window - 1:], rolling_logits[window - 1:], label='Rolling Logits Entropy', marker='o')
        plt.plot(steps[window - 1:], rolling_attention[window - 1:], label='Rolling Attention Entropy', marker='s')
        plt.xlabel('Generation Step')
        plt.ylabel('Rolling Entropy')
        plt.title(f'Rolling Entropy Over Time (Window = {window})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_name, 'rolling_entropy.png'))
        plt.close()

    def visualize_entropy_gradient(self, generation_results: Dict, folder_name: str):
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        logits_gradient = self.entropy_gradient(logits_entropies)
        attention_gradient = self.entropy_gradient(attention_entropies)

        steps = range(1, len(logits_gradient) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(steps, logits_gradient, label='Logits Entropy Gradient', marker='o')
        plt.plot(steps, attention_gradient, label='Attention Entropy Gradient', marker='s')
        plt.xlabel('Generation Step')
        plt.ylabel('Entropy Gradient')
        plt.title('Entropy Gradient Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_name, 'entropy_gradient.png'))
        plt.close()

    def visualize_entropy_correlation(self, generation_results: Dict, folder_name: str):
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        correlation = np.corrcoef(logits_entropies, attention_entropies)[0, 1]

        plt.figure(figsize=(8, 6))
        plt.scatter(logits_entropies, attention_entropies)
        plt.xlabel('Logits Entropy')
        plt.ylabel('Attention Entropy')
        plt.title(f'Logits vs Attention Entropy (Correlation: {correlation:.2f})')
        plt.grid(True)
        plt.savefig(os.path.join(folder_name, 'entropy_correlation.png'))
        plt.close()

        return correlation

    def visualize_surprisal(self, generation_results: Dict, folder_name: str):
        if not self.config.surprisal.enabled:
            logger.warning("Surprisal visualization is not enabled in the configuration.")
            return

        steps = range(1, len(generation_results['step_analyses']) + 1)
        surprisal_values = [step['surprisal'] for step in generation_results['step_analyses']]

        plt.figure(figsize=(12, 6))
        plt.plot(steps, surprisal_values, label='Surprisal', marker='o')
        plt.xlabel('Generation Step')
        plt.ylabel('Surprisal (Negative Log Probability)')
        plt.title('Surprisal Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_name, 'surprisal_over_time.png'))
        plt.close()

    def visualize_entropy_over_time_plotly(self, generation_results: Dict, folder_name: str):
        if not (self.config.logits_entropy.enabled and self.config.attention_entropy.enabled):
            logger.warning("Logits and attention entropy visualization is not enabled in the configuration.")
            return

        steps = list(range(1, len(generation_results['step_analyses']) + 1))
        logits_entropies = [step['logits_entropy'] for step in generation_results['step_analyses']]
        attention_entropies = [step['attention_entropy'] for step in generation_results['step_analyses']]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=logits_entropies, mode='lines+markers', name='Logits Entropy'))
        fig.add_trace(go.Scatter(x=steps, y=attention_entropies, mode='lines+markers', name='Attention Entropy'))

        fig.update_layout(
            title='Entropy Over Time',
            xaxis_title='Generation Step',
            yaxis_title='Entropy',
            legend_title='Entropy Type',
            hovermode='x unified'
        )

        fig.write_html(os.path.join(folder_name, 'entropy_over_time_interactive.html'))

    def visualize_multi_dim_state_plotly(self, generation_results: Dict, folder_name: str):
        states = np.array([step['multi_dim_state'] for step in generation_results['step_analyses']])

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Logits Entropy', 'Attention Entropy', 'Surprisal', 'Max Probability'))

        for i, metric in enumerate(['Logits Entropy', 'Attention Entropy', 'Surprisal', 'Max Probability']):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(
                go.Scatter(x=list(range(1, len(states) + 1)), y=states[:, i], mode='lines+markers', name=metric),
                row=row, col=col
            )

        fig.update_layout(height=800, width=1000, title_text='Multi-dimensional State Analysis')
        fig.update_xaxes(title_text='Generation Step')
        fig.update_yaxes(title_text='Value')

        fig.write_html(os.path.join(folder_name, 'multi_dim_state_interactive.html'))

    def visualize_attention_flow_plotly(self, generation_results: Dict, folder_name: str):
        attention_flows = [step['attention_flow'] for step in generation_results['step_analyses']]

        # Initialize an empty list to store reshaped attention flows
        reshaped_flows = []

        for flow in attention_flows:
            if flow.shape[0] == 1:
                # If the flow is (1, X), reshape it to (X, X)
                seq_len = flow.shape[1]
                reshaped_flow = np.zeros((seq_len, seq_len))
                reshaped_flow[-1, :] = flow[0, :]  # Put the attention weights in the last row
            else:
                # If the flow is already (X, X), use it as is
                reshaped_flow = flow
            reshaped_flows.append(reshaped_flow)

        # Determine the maximum sequence length
        max_seq_len = max(flow.shape[0] for flow in reshaped_flows)

        # Pad or trim attention flows to a consistent shape
        padded_flows = []
        for flow in reshaped_flows:
            if flow.shape[0] < max_seq_len:
                # Pad with zeros
                padded_flow = np.pad(flow, ((0, max_seq_len - flow.shape[0]), (0, max_seq_len - flow.shape[1])),
                                     mode='constant')
            else:
                # Use only the last max_seq_len tokens
                padded_flow = flow[-max_seq_len:, -max_seq_len:]
            padded_flows.append(padded_flow)

        # Create animated heatmap
        fig = go.Figure(
            data=[go.Heatmap(z=padded_flows[0], colorscale='Viridis')],
            layout=go.Layout(
                title='Attention Flow Over Time',
                xaxis=dict(title='Source Token'),
                yaxis=dict(title='Target Token'),
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        dict(label='Play',
                             method='animate',
                             args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]),
                        dict(label='Pause',
                             method='animate',
                             args=[[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
                                            'transition': {'duration': 0}}])
                    ]
                }]
            ),
            frames=[go.Frame(data=[go.Heatmap(z=flow)]) for flow in padded_flows]
        )

        fig.write_html(os.path.join(folder_name, 'attention_flow_animated.html'))

        # Create aggregated heatmap
        aggregated_flow = np.mean(padded_flows, axis=0)
        fig_aggregated = go.Figure(data=[go.Heatmap(z=aggregated_flow, colorscale='Viridis')])
        fig_aggregated.update_layout(
            title='Aggregated Attention Flow',
            xaxis_title='Source Token',
            yaxis_title='Target Token'
        )
        fig_aggregated.write_html(os.path.join(folder_name, 'attention_flow_aggregated.html'))

    def run_analysis(self, generation_results: Dict, folder_name: str):
        # Existing visualizations
        self.visualize_entropy_over_time(generation_results, folder_name)
        self.visualize_entropy_distribution(generation_results, folder_name)
        self.visualize_rolling_entropy(generation_results, folder_name)
        self.visualize_entropy_gradient(generation_results, folder_name)
        self.visualize_entropy_correlation(generation_results, folder_name)
        self.visualize_surprisal(generation_results, folder_name)

        self.visualize_multi_dim_state(generation_results, folder_name)
        self.visualize_entropy_change_rate(generation_results, folder_name)
        self.visualize_attention_flow(generation_results, folder_name)

        self.visualize_entropy_over_time_plotly(generation_results, folder_name)
        self.visualize_multi_dim_state_plotly(generation_results, folder_name)
        self.visualize_attention_flow_plotly(generation_results, folder_name)

        new_generate_html_report(
            generation_results,
            self.tokenizer,
            folder_name,
            'report.html'
        )


class BasicEntropyAnalysisWrapper(BaseEntropyAnalysisWrapper):
    def _load_model(self, model_name: str, device: str) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,  # Ensure this if using custom models
            attn_implementation="eager",
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        return model.to(device)


if __name__ == "__main__":
    config = EntropyAnalysisConfig()

    # Initialize the wrapper
    wrapper = BasicEntropyAnalysisWrapper("meta-llama/Llama-3.2-1B-Instruct", config=config)

    # Generate and analyze text
    generation_input = """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are a helpful Assistant.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Which number is bigger 9.11 or 9.9?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    experiment_folder = "005"

    os.makedirs(experiment_folder, exist_ok=True)
    g_r = []
    for _ in range(2):
        generation_results = wrapper.generate_and_analyze(
            generation_input,
            max_length=120,
            temperature=1.0,
            top_p=1.0,  # Added top_p parameter for stochastic sampling
        )

        # Create a timestamped folder for results
        results_folder = create_timestamped_folder(experiment_folder)

        # Save metadata
        save_metadata(wrapper.model_name, wrapper.device, 0, 0, results_folder)

        # Save generation results
        save_generation_results(generation_results, results_folder)
        g_r.append(copy(generation_results))
        wrapper.run_analysis(generation_results, results_folder)

        print(f"Analysis results, visualizations, and metadata saved in folder: {results_folder}")
    new_new_generate_html_report(g_r, wrapper.tokenizer, experiment_folder, 'report.html')