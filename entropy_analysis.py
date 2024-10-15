from collections import Counter
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import logging

from calibration_data import input_output_pairs
from utils import create_timestamped_folder, generate_html_report, save_generation_results, save_metadata

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
        self.perplexity = ConfigurableAnalysis()
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
        self.logits_entropy_thresholds = {}
        self.attn_entropy_thresholds = {}

    @abstractmethod
    def _load_model(self, model_name: str, device: str):
        pass

    @staticmethod
    def calculate_entropy(probs: np.ndarray) -> float:
        """Calculate the entropy of a probability distribution."""
        probs = probs[probs > 0]
        return entropy(probs, base=2)

    def calculate_sequence_entropy(self, probs: torch.Tensor) -> List[float]:
        """Calculate entropy for each token in a sequence."""
        return [self.calculate_entropy(token_probs.cpu().numpy()) for token_probs in probs]

    def collect_calibration_data(self, input_output_pairs: List[Tuple[str, str]]) -> None:
        """
        Collect entropy data from input-output pairs to calibrate thresholds.

        Args:
            input_output_pairs: List of tuples containing input and corresponding output text.
        """
        logits_entropy_list = []
        attention_entropy_list = []

        for input_text, output_text in input_output_pairs:
            try:
                # Concatenate input and output text for tokenization
                full_text = input_text + output_text
                inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)
                output_ids = self.tokenizer(output_text, return_tensors='pt')['input_ids'][0].to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)

                # Extract logits and attentions for output tokens
                logits = outputs.logits[0, -len(output_ids):, :]  # Shape: [output_len, vocab_size]
                attentions = outputs.attentions  # List of tensors per layer

                # Logits entropy
                probs = F.softmax(logits, dim=-1)
                logits_entropies = [self.calculate_entropy(token_probs.cpu().numpy()) for token_probs in probs]
                logits_entropy_list.extend(logits_entropies)

                # Attention entropy
                last_layer_attentions = attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
                output_attn_weights = last_layer_attentions[:, -len(output_ids):,
                                      :]  # Shape: [num_heads, output_len, seq_len]

                # Compute entropy per head and then average over heads for each token
                for token_attn_weights in output_attn_weights.permute(1, 0,
                                                                      2):  # Shape: [output_len, num_heads, seq_len]
                    head_entropies = [self.calculate_entropy(head_weights.cpu().numpy()) for head_weights in
                                      token_attn_weights]
                    token_entropy = np.mean(head_entropies)
                    attention_entropy_list.append(token_entropy)

            except Exception as e:
                logger.error(f"Error processing input-output pair: {e}", exc_info=True)
                continue

        # Use percentiles for threshold calculations
        self.logits_entropy_thresholds['high'] = np.percentile(logits_entropy_list, 75)
        self.logits_entropy_thresholds['low'] = np.percentile(logits_entropy_list, 25)
        self.attn_entropy_thresholds['high'] = np.percentile(attention_entropy_list, 75)
        self.attn_entropy_thresholds['low'] = np.percentile(attention_entropy_list, 25)

        logger.info(
            f"Logits Entropy Thresholds - Low (25th percentile): {self.logits_entropy_thresholds['low']:.4f}, High (75th percentile): {self.logits_entropy_thresholds['high']:.4f}")
        logger.info(
            f"Attention Entropy Thresholds - Low (25th percentile): {self.attn_entropy_thresholds['low']:.4f}, High (75th percentile): {self.attn_entropy_thresholds['high']:.4f}")

    def categorize_state(self, logits_entropy: float, attention_entropy: float, surprisal: float,
                         max_probability: float) -> str:
        high_logits_entropy = self.logits_entropy_thresholds['high']
        low_logits_entropy = self.logits_entropy_thresholds['low']
        high_attn_entropy = self.attn_entropy_thresholds['high']
        low_attn_entropy = self.attn_entropy_thresholds['low']

        # Define additional thresholds
        very_high_logits_entropy = high_logits_entropy * 1.5
        very_low_logits_entropy = low_logits_entropy * 0.5
        high_surprisal = 4.0  # log2(16), representing a 1/16 probability
        low_surprisal = 2.0  # log2(4), representing a 1/4 probability
        high_max_probability = 0.9
        low_max_probability = 0.3

        if logits_entropy > very_high_logits_entropy and attention_entropy > high_attn_entropy:
            return 'Very Uncertain'
        elif logits_entropy > high_logits_entropy and attention_entropy > high_attn_entropy:
            return 'Uncertain'
        elif logits_entropy < very_low_logits_entropy and attention_entropy < low_attn_entropy and max_probability > high_max_probability:
            return 'Very Overconfident'
        elif logits_entropy < low_logits_entropy and attention_entropy < low_attn_entropy:
            return 'Overconfident'
        elif low_logits_entropy <= logits_entropy <= high_logits_entropy and low_attn_entropy <= attention_entropy <= high_attn_entropy:
            if surprisal < low_surprisal and max_probability > high_max_probability:
                return 'Highly Confident'
            elif surprisal > high_surprisal and max_probability < low_max_probability:
                return 'Slightly Uncertain'
            else:
                return 'Confident'
        elif logits_entropy > high_logits_entropy and attention_entropy <= high_attn_entropy:
            return 'Exploring'
        elif logits_entropy <= high_logits_entropy and attention_entropy > high_attn_entropy:
            return 'Focusing'
        else:
            return 'Balanced'

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

        if all(k in step_analysis for k in ['logits_entropy', 'attention_entropy', 'surprisal', 'max_probability']):
            state = self.categorize_state(
                step_analysis['logits_entropy'],
                step_analysis['attention_entropy'],
                step_analysis['surprisal'],
                step_analysis['max_probability']
            )
            step_analysis['model_state'] = state

        return step_analysis

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

    def visualize_model_states(self, generation_results: Dict, folder_name: str):
        if not generation_results['step_analyses']:
            logger.warning("No step analyses available for visualization.")
            return

        if 'model_state' not in generation_results['step_analyses'][0]:
            logger.warning("Model state visualization is not available.")
            return

        steps = range(1, len(generation_results['step_analyses']) + 1)
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

        plt.figure(figsize=(15, 8))
        plt.plot(steps, numeric_states, marker='o')
        plt.yticks(range(len(state_to_num)), list(state_to_num.keys()))
        plt.xlabel('Generation Step')
        plt.ylabel('Model State')
        plt.title('Model State Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'model_states.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Additional visualization: State distribution
        state_counts = Counter(states)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(state_counts.keys(), state_counts.values())
        plt.xlabel('Model State')
        plt.ylabel('Frequency')
        plt.title('Distribution of Model States')
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'model_states_distribution.png'), dpi=300, bbox_inches='tight')
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

    def analyze_entropy_correlation(self, generation_results: Dict, folder_name: str):
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

    wrapper.collect_calibration_data(input_output_pairs)

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

    experiment_folder = "004"

    os.makedirs(experiment_folder, exist_ok=True)

    for _ in range(4):
        generation_results = wrapper.generate_and_analyze(
            generation_input,
            max_length=200,
            temperature=1.0,
            top_p=1.0,  # Added top_p parameter for stochastic sampling
        )

        # Create a timestamped folder for results
        results_folder = create_timestamped_folder(experiment_folder)

        # Save metadata
        save_metadata(wrapper.model_name, wrapper.device, wrapper.logits_entropy_thresholds,
                      wrapper.attn_entropy_thresholds, results_folder)

        # Save generation results
        save_generation_results(generation_results, results_folder)

        # Generate and save visualizations
        wrapper.visualize_entropy_over_time(generation_results, results_folder)
        wrapper.visualize_model_states(generation_results, results_folder)
        wrapper.visualize_entropy_distribution(generation_results, results_folder)
        wrapper.visualize_rolling_entropy(generation_results, results_folder)
        wrapper.visualize_entropy_gradient(generation_results, results_folder)
        correlation = wrapper.analyze_entropy_correlation(generation_results, results_folder)

        # New visualizations
        wrapper.visualize_surprisal(generation_results, results_folder)
        generate_html_report(
            generation_results,
            wrapper.tokenizer,
            results_folder,
            'report.html'
        )
        print(f"Analysis results, visualizations, and metadata saved in folder: {results_folder}")
