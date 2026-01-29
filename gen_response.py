"""
Refactored Model Evaluation Framework
Supports multiple safety evaluation datasets with a clean, extensible architecture.
"""

import os
import argparse
import json
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from huggingface_hub.errors import HfHubHTTPError


def retry_with_exponential_backoff(max_retries=5, base_delay=1.0, max_delay=60.0):
    """Decorator to retry function calls with exponential backoff for rate limiting."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (HfHubHTTPError, Exception) as e:
                    if attempt == max_retries:
                        print(f"Failed after {max_retries} retries. Last error: {e}")
                        raise

                    # Check if it's a rate limiting error
                    is_rate_limit = (
                        isinstance(e, HfHubHTTPError) and "429" in str(e)
                    ) or "Too Many Requests" in str(e)

                    if is_rate_limit:
                        # Calculate delay with exponential backoff + jitter
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        print(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                              f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        # For non-rate-limit errors, re-raise immediately
                        raise
            return None
        return wrapper
    return decorator


def setup_hf_cache():
    """Set up HuggingFace cache directory and environment variables."""
    # Set cache directory to avoid permission issues
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    print(f"Using HuggingFace cache directory: {cache_dir}")


@dataclass
class ModelConfig:
    """Configuration for model loading and generation."""
    model_path: str
    max_tokens: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 0
    use_lora: bool = False
    lora_rank: Optional[int] = None
    base_model_path: Optional[str] = None


@dataclass
class EvaluationResult:
    """Standard structure for evaluation results."""
    prompt: str
    response: str
    metadata: Dict[str, Any] = None


class ModelLoader:
    """Handles loading of models with or without LoRA adapters."""
    
    @staticmethod
    def load(model_path: str) -> Tuple[LLM, AutoTokenizer, bool]:
        """
        Load model and tokenizer, automatically detecting LoRA configuration.

        Args:
            model_path: Path to model or model name

        Returns:
            Tuple of (model, tokenizer, use_lora flag)
        """
        print(f"Loading tokenizer for model: {model_path}")
        tokenizer = ModelLoader._load_tokenizer_with_retry(model_path)

        lora_config = ModelLoader._load_lora_config(model_path)
        if lora_config:
            print("LoRA configuration found, loading LoRA model...")
            model = ModelLoader._load_lora_model_with_retry(model_path, lora_config)
            return model, tokenizer, True
        else:
            print("Loading standard model...")
            model = ModelLoader._load_standard_model_with_retry(model_path)
            return model, tokenizer, False
    
    @staticmethod
    @retry_with_exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
    def _load_tokenizer_with_retry(model_path: str) -> AutoTokenizer:
        """Load and configure tokenizer with retry logic."""
        return ModelLoader._load_tokenizer(model_path)

    @staticmethod
    @retry_with_exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
    def _load_standard_model_with_retry(model_path: str) -> LLM:
        """Load standard model with retry logic."""
        return LLM(model=model_path, tokenizer=model_path)

    @staticmethod
    @retry_with_exponential_backoff(max_retries=5, base_delay=2.0, max_delay=120.0)
    def _load_lora_model_with_retry(model_path: str, lora_config: Dict) -> LLM:
        """Load model with LoRA adapter with retry logic."""
        return ModelLoader._load_lora_model(model_path, lora_config)

    @staticmethod
    def _load_tokenizer(model_path: str) -> AutoTokenizer:
        """Load and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @staticmethod
    def _load_lora_config(model_path: str) -> Optional[Dict]:
        """Load LoRA configuration if exists."""
        lora_config_path = os.path.join(model_path, "adapter_config.json")
        
        if not os.path.exists(lora_config_path):
            return None
            
        try:
            with open(lora_config_path) as f:
                config = json.load(f)
            
            # Validate required fields
            if not config.get("base_model_name_or_path") or not config.get("r"):
                print(f"Warning: Invalid LoRA config in {model_path}")
                return None
                
            return config
            
        except Exception as e:
            print(f"Error loading LoRA config: {e}")
            return None
    
    @staticmethod
    def _load_lora_model(model_path: str, lora_config: Dict) -> LLM:
        """Load model with LoRA adapter."""
        return LLM(
            model=lora_config["base_model_name_or_path"],
            tokenizer=model_path,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=lora_config["r"],
        )


class BaseEvaluator(ABC):
    """Abstract base class for dataset evaluators."""
    
    def __init__(self, model: LLM, tokenizer: AutoTokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_name = self.__class__.__name__.replace("Evaluator", "").lower()
    
    @abstractmethod
    def load_data(self, eval_path: str) -> List[Dict]:
        """Load evaluation data from specified path or dataset."""
        pass
    
    @abstractmethod
    def prepare_prompts(self, data: List[Dict]) -> List[str]:
        """Prepare prompts for generation."""
        pass
    
    @abstractmethod
    def format_results(self, data: List[Dict], responses: List[str]) -> List[Dict]:
        """Format results for output."""
        pass
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses for given prompts."""
        # Use chat() method which handles tokenization properly
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k
        )

        lora_request = None
        if self.config.use_lora:
            lora_request = LoRARequest("lora", 1, self.config.model_path)
            print("Using LoRA adapter for generation")

        outputs = self.model.chat(
            messages_list,
            sampling_params=sampling_params,
            lora_request=lora_request
        )

        return [output.outputs[0].text for output in outputs]
    
    def evaluate(self, eval_path: str, output_dir: str):
        """Main evaluation pipeline."""
        print(f"Starting {self.dataset_name} evaluation...")
        
        # Load data
        data = self.load_data(eval_path)
        print(f"Loaded {len(data)} examples")
        
        # Prepare prompts
        prompts = self.prepare_prompts(data)
        
        # Generate responses
        responses = self.generate_responses(prompts)
        
        # Format results
        results = self.format_results(data, responses)
        
        # Save results
        output_file = os.path.join(output_dir, f"{self.dataset_name}.json")
        self.save_results(results, output_file)
        print(f"Results saved to {output_file}")
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)


class HarmbenchEvaluator(BaseEvaluator):
    """Evaluator for Harmbench dataset."""
    
    def load_data(self, eval_path: str) -> List[Dict]:
        with open(eval_path, 'r') as file:
            return json.load(file)
    
    def prepare_prompts(self, data: List[Dict]) -> List[str]:
        return [conv["prompt"] for conv in data]
    
    def format_results(self, data: List[Dict], responses: List[str]) -> List[Dict]:
        return [
            {
                "objective": data[i].get("objective", ""),
                "prompt": data[i]["prompt"],
                "response_generated": responses[i]
            }
            for i in range(len(responses))
        ]


class XSTestEvaluator(BaseEvaluator):
    """Evaluator for XSTest dataset."""
    
    def load_data(self, eval_path: str) -> List[Dict]:
        dataset = load_dataset("walledai/XSTest", split="test")
        return [{"prompt": x["prompt"], "label": x["label"]} for x in dataset]
    
    def prepare_prompts(self, data: List[Dict]) -> List[str]:
        return [x["prompt"] for x in data]
    
    def format_results(self, data: List[Dict], responses: List[str]) -> List[Dict]:
        return [
            {
                "label": data[i]["label"],
                "prompt": data[i]["prompt"],
                "response_generated": responses[i]
            }
            for i in range(len(responses))
        ]


class WildGuardEvaluator(BaseEvaluator):
    """Evaluator for WildGuard dataset."""
    
    def load_data(self, eval_path: str) -> List[Dict]:
        dataset = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
        # Filter for harmful prompts only
        harmful_data = []
        for x in dataset:
            if x["prompt_harm_label"] == "harmful":
                harmful_data.append({"prompt": x["prompt"]})
        return harmful_data
    
    def prepare_prompts(self, data: List[Dict]) -> List[str]:
        return [x["prompt"] for x in data]
    
    def format_results(self, data: List[Dict], responses: List[str]) -> List[Dict]:
        return [
            {
                "prompt": data[i]["prompt"],
                "response_generated": responses[i]
            }
            for i in range(len(responses))
        ]


class StrongRejectEvaluator(BaseEvaluator):
    """Evaluator for StrongREJECT dataset."""
    
    def load_data(self, eval_path: str) -> List[Dict]:
        dataset = load_dataset("walledai/StrongREJECT", split="train")
        return [{"prompt": x["prompt"]} for x in dataset]
    
    def prepare_prompts(self, data: List[Dict]) -> List[str]:
        return [x["prompt"] for x in data]
    
    def format_results(self, data: List[Dict], responses: List[str]) -> List[Dict]:
        return [
            {
                "prompt": data[i]["prompt"],
                "response_generated": responses[i]
            }
            for i in range(len(responses))
        ]


class WildJailbreakEvaluator(BaseEvaluator):
    """Evaluator for WildJailbreak dataset."""

    def load_data(self, eval_path: str) -> List[Dict]:
        dataset = load_dataset("allenai/wildjailbreak", "eval", split="train")
        # Filter for adversarial harmful prompts
        harmful_data = []
        for x in dataset:
            if x["data_type"] == "adversarial_harmful":
                harmful_data.append({
                    "prompt": x["adversarial"],
                    "data_type": x["data_type"],
                    "label": x.get("label", None)
                })
        return harmful_data

    def prepare_prompts(self, data: List[Dict]) -> List[str]:
        return [x["prompt"] for x in data]

    def format_results(self, data: List[Dict], responses: List[str]) -> List[Dict]:
        return [
            {
                "prompt": data[i]["prompt"],
                "data_type": data[i]["data_type"],
                "response_generated": responses[i]
            }
            for i in range(len(responses))
        ]


class EvaluatorFactory:
    """Factory class for creating appropriate evaluator instances."""
    
    EVALUATORS = {
        "harmbench": HarmbenchEvaluator,
        "xstest": XSTestEvaluator,
        "wildguard": WildGuardEvaluator,
        "strongreject": StrongRejectEvaluator,
        "wildjailbreak": WildJailbreakEvaluator,
    }
    
    @classmethod
    def create(cls, mode: str, model: LLM, tokenizer: AutoTokenizer, 
               config: ModelConfig) -> BaseEvaluator:
        """Create evaluator instance based on mode."""
        if mode not in cls.EVALUATORS:
            raise ValueError(f"Unknown evaluation mode: {mode}. "
                           f"Choose from: {list(cls.EVALUATORS.keys())}")
        
        evaluator_class = cls.EVALUATORS[mode]
        return evaluator_class(model, tokenizer, config)
    
    @classmethod
    def register(cls, mode: str, evaluator_class: type):
        """Register a new evaluator class."""
        cls.EVALUATORS[mode] = evaluator_class


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate language models on safety benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=list(EvaluatorFactory.EVALUATORS.keys()),
        required=True,
        help="Evaluation mode/dataset"
    )
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path or name of the pre-trained model"
    )
    
    parser.add_argument(
        "--eval_path",
        default="",
        help="Path to evaluation data (required for some modes)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="eval_results",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )

    parser.add_argument("--top_k",
        type=int,
        default=0,
        help="Top-k sampling parameter")
    
    return parser.parse_args()


def main():
    """Main execution function."""

    # Parse arguments
    args = parse_arguments()

    print(f"Configuration: {args}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, use_lora = ModelLoader.load(args.model_path)
    print(f"Model loaded successfully (LoRA: {use_lora})")

    # Create model configuration
    config = ModelConfig(
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_lora=use_lora
    )

    # Create and run evaluator
    evaluator = EvaluatorFactory.create(args.mode, model, tokenizer, config)
    evaluator.evaluate(args.eval_path, args.output_dir)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
