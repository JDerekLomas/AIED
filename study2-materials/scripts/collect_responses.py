"""
Data Collection Script for Study 2: Misconception Alignment

This script generates LLM responses for study items across multiple
models and prompting conditions.

Usage:
    python collect_responses.py --config config.yaml
    python collect_responses.py --items data/study_items.json --output results/

Requirements:
    pip install openai anthropic together pandas tqdm pyyaml
"""

import json
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import re

import pandas as pd
from tqdm import tqdm

# API clients - import conditionally
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from together import Together
except ImportError:
    Together = None

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

PROMPT_TEMPLATES = {
    "answer_only": """Solve this math problem. Select the best answer from the options.

Question: {question}

A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

Answer:""",

    "explain": """Solve this math problem. Show your reasoning step by step, then select the best answer from the options.

Question: {question}

A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

Reasoning:""",

    "persona": """You are a middle school student working on a math problem. Students at your level sometimes make mistakes - that's okay and normal.

Solve this problem the way a real student would. Show your thinking as you work through it, even if you're not sure. It's fine to make errors.

Question: {question}

A) {answer_a}
B) {answer_b}
C) {answer_c}
D) {answer_d}

Your thinking:"""
}


# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

MODEL_CONFIGS = {
    # Frontier tier (~90%+ GSM8K)
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    # Mid tier (~70-85% GSM8K)
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    "llama-70b": {
        "provider": "together",
        "model_id": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    # Weaker tier (~40-60% GSM8K)
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    "mistral-7b": {
        "provider": "mistral",
        "model_id": "open-mistral-7b",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    "llama-8b": {
        "provider": "together",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    # Gemini models
    "gemini-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.0-flash-exp",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    "gemini-pro": {
        "provider": "gemini",
        "model_id": "gemini-1.5-pro",
        "max_tokens": 500,
        "temperature": 0.7,
    },
}


# ==============================================================================
# API CLIENTS
# ==============================================================================

class LLMClient:
    """Unified interface for different LLM providers."""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.together_client = None
        self.mistral_client = None

    def _get_openai(self):
        if self.openai_client is None:
            if OpenAI is None:
                raise ImportError("openai package not installed")
            self.openai_client = OpenAI()
        return self.openai_client

    def _get_anthropic(self):
        if self.anthropic_client is None:
            if Anthropic is None:
                raise ImportError("anthropic package not installed")
            self.anthropic_client = Anthropic()
        return self.anthropic_client

    def _get_together(self):
        if self.together_client is None:
            if Together is None:
                raise ImportError("together package not installed")
            self.together_client = Together()
        return self.together_client

    def _get_mistral(self):
        if self.mistral_client is None:
            if Mistral is None:
                raise ImportError("mistralai package not installed")
            self.mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        return self.mistral_client

    def _get_gemini(self, model_id: str):
        if genai is None:
            raise ImportError("google-generativeai package not installed")
        genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY"))
        return genai.GenerativeModel(model_id)

    def generate(
        self,
        prompt: str,
        model_name: str,
        n: int = 1,
    ) -> list[dict]:
        """Generate n responses for a prompt."""
        config = MODEL_CONFIGS[model_name]
        provider = config["provider"]

        responses = []
        for _ in range(n):
            try:
                if provider == "openai":
                    response = self._generate_openai(prompt, config)
                elif provider == "anthropic":
                    response = self._generate_anthropic(prompt, config)
                elif provider == "together":
                    response = self._generate_together(prompt, config)
                elif provider == "mistral":
                    response = self._generate_mistral(prompt, config)
                elif provider == "gemini":
                    response = self._generate_gemini(prompt, config)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                responses.append({
                    "text": response,
                    "error": None,
                })
            except Exception as e:
                responses.append({
                    "text": None,
                    "error": str(e),
                })
                time.sleep(1)  # Back off on error

        return responses

    def _generate_openai(self, prompt: str, config: dict) -> str:
        client = self._get_openai()
        response = client.chat.completions.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, prompt: str, config: dict) -> str:
        client = self._get_anthropic()
        response = client.messages.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.content[0].text

    def _generate_together(self, prompt: str, config: dict) -> str:
        client = self._get_together()
        response = client.chat.completions.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.choices[0].message.content

    def _generate_mistral(self, prompt: str, config: dict) -> str:
        client = self._get_mistral()
        response = client.chat.complete(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.choices[0].message.content

    def _generate_gemini(self, prompt: str, config: dict) -> str:
        model = self._get_gemini(config["model_id"])
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
        )
        return response.text


# ==============================================================================
# RESPONSE PARSING
# ==============================================================================

def parse_answer(response: str) -> Optional[str]:
    """Extract answer letter from LLM response."""
    if response is None:
        return None

    response = response.strip()

    # Pattern 1: Explicit "Answer: X" format
    match = re.search(r'(?:answer|choice|select)[:\s]+([ABCD])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: "The answer is X" format
    match = re.search(r'(?:answer is|choose|pick)\s+([ABCD])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Letter at end of response
    match = re.search(r'\b([ABCD])\s*[.!]?\s*$', response)
    if match:
        return match.group(1).upper()

    # Pattern 4: Standalone letter
    if response.upper() in ['A', 'B', 'C', 'D']:
        return response.upper()

    # Pattern 5: Last letter mentioned in response
    letters = re.findall(r'\b([ABCD])\b', response)
    if letters:
        return letters[-1].upper()

    return None


def extract_reasoning(response: str) -> str:
    """Extract reasoning portion before answer."""
    if response is None:
        return ""

    lines = response.strip().split('\n')
    reasoning_lines = []
    for line in lines:
        if re.search(r'^(?:answer|my answer|final answer)[:\s]', line, re.IGNORECASE):
            break
        reasoning_lines.append(line)
    return '\n'.join(reasoning_lines).strip()


# ==============================================================================
# DATA COLLECTION
# ==============================================================================

def format_prompt(item: dict, condition: str) -> str:
    """Format a prompt for an item and condition."""
    template = PROMPT_TEMPLATES[condition]
    return template.format(
        question=item["question_text"],
        answer_a=item["answer_a"],
        answer_b=item["answer_b"],
        answer_c=item["answer_c"],
        answer_d=item["answer_d"],
    )


def collect_responses(
    items: list[dict],
    models: list[str],
    conditions: list[str],
    n_responses: int,
    output_dir: Path,
    client: LLMClient,
    resume: bool = True,
) -> pd.DataFrame:
    """Collect responses for all items across models and conditions."""

    output_file = output_dir / "responses.jsonl"
    collected = set()

    # Load existing responses if resuming
    if resume and output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                record = json.loads(line)
                key = (record["item_id"], record["model"], record["condition"], record["response_idx"])
                collected.add(key)
        print(f"Resuming: {len(collected)} responses already collected")

    # Calculate total work
    total = len(items) * len(models) * len(conditions) * n_responses
    remaining = total - len(collected)
    print(f"Total responses needed: {total}")
    print(f"Remaining: {remaining}")

    with open(output_file, "a") as f:
        with tqdm(total=remaining, desc="Collecting responses") as pbar:
            for item in items:
                for model in models:
                    for condition in conditions:
                        # Check which response indices we still need
                        needed_indices = [
                            i for i in range(n_responses)
                            if (item["item_id"], model, condition, i) not in collected
                        ]

                        if not needed_indices:
                            continue

                        # Format prompt
                        prompt = format_prompt(item, condition)

                        # Generate responses
                        responses = client.generate(prompt, model, n=len(needed_indices))

                        # Save responses
                        for idx, response in zip(needed_indices, responses):
                            parsed_answer = parse_answer(response["text"])
                            reasoning = extract_reasoning(response["text"]) if condition != "answer_only" else ""

                            record = {
                                "item_id": item["item_id"],
                                "model": model,
                                "condition": condition,
                                "response_idx": idx,
                                "prompt": prompt,
                                "raw_response": response["text"],
                                "parsed_answer": parsed_answer,
                                "reasoning": reasoning,
                                "correct_answer": item["correct_answer"],
                                "is_correct": parsed_answer == item["correct_answer"] if parsed_answer else None,
                                "error": response["error"],
                                "timestamp": datetime.now().isoformat(),
                            }

                            # Add misconception info if available
                            if "misconception_id" in item:
                                record["misconception_id"] = item["misconception_id"]
                                record["misconception_name"] = item.get("misconception_name", "")
                                record["misconception_category"] = item.get("misconception_category", "")

                            f.write(json.dumps(record) + "\n")
                            f.flush()
                            pbar.update(1)

                        # Rate limiting
                        time.sleep(0.1)

    # Load and return all responses
    records = []
    with open(output_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    return pd.DataFrame(records)


# ==============================================================================
# MAIN
# ==============================================================================

def load_items(items_path: Path) -> list[dict]:
    """Load items from JSON file."""
    with open(items_path, "r") as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        return data
    elif "items" in data:
        return data["items"]
    elif "misconceptions" in data:
        # Probe items format
        items = []
        for misc in data["misconceptions"]:
            for item in misc["items"]:
                items.append({
                    "item_id": item["item_id"],
                    "question_text": item["question"],
                    "answer_a": item["options"]["A"],
                    "answer_b": item["options"]["B"],
                    "answer_c": item["options"]["C"],
                    "answer_d": item["options"]["D"],
                    "correct_answer": item["correct"],
                    "misconception_id": misc["id"],
                    "misconception_name": misc["name"],
                    "misconception_category": misc["category"],
                    "target_distractor": item.get("target_distractor"),
                    "target_reasoning": item.get("target_reasoning"),
                })
        return items
    else:
        raise ValueError("Unknown items format")


def main():
    parser = argparse.ArgumentParser(description="Collect LLM responses for study items")
    parser.add_argument("--items", type=Path, required=True, help="Path to items JSON file")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"], help="Models to use")
    parser.add_argument("--conditions", nargs="+", default=["answer_only", "explain", "persona"], help="Prompting conditions")
    parser.add_argument("--n", type=int, default=5, help="Responses per item per condition")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume")
    args = parser.parse_args()

    # Setup
    args.output.mkdir(parents=True, exist_ok=True)

    # Load items
    print(f"Loading items from {args.items}")
    items = load_items(args.items)
    print(f"Loaded {len(items)} items")

    # Validate models
    for model in args.models:
        if model not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_CONFIGS.keys())}")

    # Initialize client
    client = LLMClient()

    # Collect responses
    print(f"Collecting responses...")
    print(f"  Models: {args.models}")
    print(f"  Conditions: {args.conditions}")
    print(f"  N per condition: {args.n}")

    df = collect_responses(
        items=items,
        models=args.models,
        conditions=args.conditions,
        n_responses=args.n,
        output_dir=args.output,
        client=client,
        resume=not args.no_resume,
    )

    # Summary
    print(f"\nCollection complete!")
    print(f"Total responses: {len(df)}")
    print(f"By model: {df['model'].value_counts().to_dict()}")
    print(f"By condition: {df['condition'].value_counts().to_dict()}")
    print(f"Parse success rate: {df['parsed_answer'].notna().mean():.1%}")
    print(f"Accuracy: {df['is_correct'].mean():.1%}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "items_file": str(args.items),
        "n_items": len(items),
        "n_responses": len(df),
        "models": args.models,
        "conditions": args.conditions,
        "n_per_condition": args.n,
        "parse_success_rate": float(df['parsed_answer'].notna().mean()),
        "accuracy": float(df['is_correct'].mean()),
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
