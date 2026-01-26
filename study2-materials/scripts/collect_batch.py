#!/usr/bin/env python3
"""
Batch Data Collection Script for Study 2: Misconception Alignment

This script presents items in batches (like a test) rather than one at a time.
This allows testing for sequencing effects and more realistic test-taking simulation.

Key differences from single-item approach:
- Multiple items presented in one prompt
- Position/sequence tracked for each item
- Tests fatigue, learning, and context effects
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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# API clients
from openai import OpenAI
from anthropic import Anthropic
from mistralai import Mistral
import google.generativeai as genai
import requests


# ==============================================================================
# MODEL CONFIGURATIONS (Validated)
# ==============================================================================

MODEL_CONFIGS = {
    # Frontier tier (~90% GSM8K)
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 92.0,
        "tier": "frontier",
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 96.0,
        "tier": "frontier",
    },
    # Mid tier (~75-85% GSM8K)
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "max_tokens": 4000,  # Higher for batch responses
        "temperature": 0.7,
        "gsm8k": 85.0,
        "tier": "mid",
    },
    "claude-3.5-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-5-haiku-20241022",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 85.0,
        "tier": "mid",
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 88.9,
        "tier": "mid",
    },
    "gemini-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.0-flash-exp",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 68.8,
        "tier": "mid",
    },
    # Weak tier
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 57.1,
        "tier": "weak",
    },
    "mistral-7b": {
        "provider": "mistral",
        "model_id": "open-mistral-7b",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 45.2,
        "tier": "weak",
    },
    # Very weak tier (via Replicate)
    "llama-2-7b": {
        "provider": "replicate",
        "model_id": "meta/llama-2-7b-chat",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 7.0,  # ~5-9% reported
        "tier": "very-weak",
    },
    "llama-2-13b": {
        "provider": "replicate",
        "model_id": "meta/llama-2-13b-chat",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 28.0,  # ~28% reported
        "tier": "very-weak",
    },
    # Groq (fast inference for Llama models)
    "llama-3.1-8b-groq": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 57.0,  # Llama 3.1 8B estimate
        "tier": "weak",
    },
    # Together.ai models
    "llama-3.1-8b-together": {
        "provider": "together",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 57.0,
        "tier": "weak",
    },
    "llama-3.2-3b-together": {
        "provider": "together",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 25.0,  # 3B model estimate - weaker
        "tier": "very-weak",
    },
    # OpenRouter models (access multiple providers)
    "gpt-3.5-turbo-openrouter": {
        "provider": "openrouter",
        "model_id": "openai/gpt-3.5-turbo",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 57.1,
        "tier": "weak",
    },
    "claude-3-haiku-openrouter": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-3-haiku",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 88.9,
        "tier": "mid",
    },
    "mistral-7b-openrouter": {
        "provider": "openrouter",
        "model_id": "mistralai/mistral-7b-instruct",
        "max_tokens": 4000,
        "temperature": 0.7,
        "gsm8k": 45.2,
        "tier": "weak",
    },
}


# ==============================================================================
# BATCH PROMPT TEMPLATES
# ==============================================================================

# Condition 1: Answer-Only (no reasoning required)
BATCH_ANSWER_ONLY_TEMPLATE = """You are taking a math test with {n_items} questions.
Select the best answer for each question (A, B, C, or D).

Format each answer as:
**Question N:** X

---

{questions}

---

Please answer all {n_items} questions above."""


# Condition 2: Explain-Then-Answer (show reasoning)
BATCH_EXPLAIN_TEMPLATE = """You are taking a math test with {n_items} questions.
Answer each question by showing your reasoning, then stating your final answer as a single letter (A, B, C, or D).

Format each answer as:
**Question N:**
[Your reasoning]
**Answer: X**

---

{questions}

---

Please answer all {n_items} questions above. Remember to show your work and clearly state each answer."""


BATCH_PERSONA_TEMPLATE = """You are a middle school student taking a math test with {n_items} questions.
Students at your level sometimes make mistakes - that's okay and normal. Work through each problem the way a real student would, showing your thinking even if you're not sure.

Format each answer as:
**Question N:**
[Your thinking/work]
**Answer: X**

---

{questions}

---

Please answer all {n_items} questions above. Show your thinking like a student would."""


BATCH_DIAGNOSE_SIMULATE_TEMPLATE = """You are simulating a struggling middle school student taking a math test with {n_items} questions, for educational research purposes.

For each question:
1. First, briefly identify what common mistake students typically make on this type of problem
2. Then solve the problem AS IF you were a student who holds that misconception
3. Show the flawed reasoning a real struggling student would use

Format each answer as:
**Question N:**
Common student error: [what mistake students make]
Student thinking: [show flawed reasoning]
**Answer: X**

---

{questions}

---

Please answer all {n_items} questions above. Remember to think like a struggling student who makes typical mistakes."""


def format_batch_prompt(items: list, condition: str = "explain") -> str:
    """Format multiple items into a single batch prompt."""
    questions = []
    for i, item in enumerate(items, 1):
        q = f"""**Question {i}:**
{item['question_text']}

A) {item['answer_a']}
B) {item['answer_b']}
C) {item['answer_c']}
D) {item['answer_d']}"""
        questions.append(q)

    questions_text = "\n\n".join(questions)

    if condition == "answer_only":
        template = BATCH_ANSWER_ONLY_TEMPLATE
    elif condition == "persona":
        template = BATCH_PERSONA_TEMPLATE
    elif condition == "diagnose_simulate":
        template = BATCH_DIAGNOSE_SIMULATE_TEMPLATE
    else:  # "explain" is default
        template = BATCH_EXPLAIN_TEMPLATE

    return template.format(n_items=len(items), questions=questions_text)


# ==============================================================================
# RESPONSE PARSING
# ==============================================================================

def parse_batch_response(response: str, n_items: int) -> list[dict]:
    """Parse a batch response into individual answers."""
    results = []

    # Try to find Question N patterns
    for i in range(1, n_items + 1):
        result = {
            "position": i,
            "parsed_answer": None,
            "reasoning": "",
        }

        # Look for Question N section
        pattern = rf'\*?\*?Question\s*{i}\*?\*?:?(.*?)(?=\*?\*?Question\s*{i+1}|\Z)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

        if match:
            section = match.group(1).strip()
            result["reasoning"] = section

            # Look for Answer: X pattern
            answer_match = re.search(r'\*?\*?Answer:?\s*\*?\*?\s*([ABCD])\b', section, re.IGNORECASE)
            if answer_match:
                result["parsed_answer"] = answer_match.group(1).upper()
            else:
                # Fallback: last letter mentioned
                letters = re.findall(r'\b([ABCD])\b', section)
                if letters:
                    result["parsed_answer"] = letters[-1].upper()

        results.append(result)

    return results


# ==============================================================================
# API CLIENTS
# ==============================================================================

class BatchLLMClient:
    """Client for batch API calls."""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.mistral_client = None
        self.openrouter_client = None

    def generate(self, prompt: str, model_name: str) -> dict:
        """Generate a single batch response."""
        config = MODEL_CONFIGS[model_name]
        provider = config["provider"]

        try:
            if provider == "openai":
                response = self._generate_openai(prompt, config)
            elif provider == "anthropic":
                response = self._generate_anthropic(prompt, config)
            elif provider == "mistral":
                response = self._generate_mistral(prompt, config)
            elif provider == "gemini":
                response = self._generate_gemini(prompt, config)
            elif provider == "openrouter":
                response = self._generate_openrouter(prompt, config)
            elif provider == "replicate":
                response = self._generate_replicate(prompt, config)
            elif provider == "groq":
                response = self._generate_groq(prompt, config)
            elif provider == "together":
                response = self._generate_together(prompt, config)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            return {"text": response, "error": None}
        except Exception as e:
            return {"text": None, "error": str(e)}

    def _generate_openai(self, prompt: str, config: dict) -> str:
        if self.openai_client is None:
            self.openai_client = OpenAI()
        response = self.openai_client.chat.completions.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, prompt: str, config: dict) -> str:
        if self.anthropic_client is None:
            self.anthropic_client = Anthropic()
        response = self.anthropic_client.messages.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.content[0].text

    def _generate_mistral(self, prompt: str, config: dict) -> str:
        if self.mistral_client is None:
            self.mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        response = self.mistral_client.chat.complete(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.choices[0].message.content

    def _generate_gemini(self, prompt: str, config: dict) -> str:
        genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY"))
        model = genai.GenerativeModel(config["model_id"])
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
        )
        return response.text

    def _generate_openrouter(self, prompt: str, config: dict) -> str:
        """Generate via OpenRouter API (OpenAI-compatible)."""
        if self.openrouter_client is None:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        response = self.openrouter_client.chat.completions.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.choices[0].message.content

    def _generate_replicate(self, prompt: str, config: dict) -> str:
        """Generate via Replicate API using direct HTTP requests."""
        api_token = os.environ.get("REPLICATE_API_TOKEN")

        # Start prediction
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {api_token}",
                "Content-Type": "application/json",
            },
            json={
                "version": "meta/llama-2-7b-chat",  # Use model version
                "input": {
                    "prompt": prompt,
                    "max_new_tokens": config["max_tokens"],
                    "temperature": config["temperature"],
                }
            }
        )
        prediction = response.json()

        # Poll for completion
        prediction_url = prediction.get("urls", {}).get("get")
        if not prediction_url:
            # Try newer API format
            prediction_id = prediction.get("id")
            prediction_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"

        while True:
            response = requests.get(
                prediction_url,
                headers={"Authorization": f"Token {api_token}"}
            )
            result = response.json()
            status = result.get("status")

            if status == "succeeded":
                output = result.get("output", "")
                if isinstance(output, list):
                    return "".join(output)
                return output
            elif status in ("failed", "canceled"):
                raise Exception(f"Replicate prediction {status}: {result.get('error')}")

            time.sleep(1)  # Poll every second

    def _generate_groq(self, prompt: str, config: dict) -> str:
        """Generate via Groq API (OpenAI-compatible, fast Llama inference)."""
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        response = client.chat.completions.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.choices[0].message.content

    def _generate_together(self, prompt: str, config: dict) -> str:
        """Generate via Together.ai API (OpenAI-compatible)."""
        client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )
        response = client.chat.completions.create(
            model=config["model_id"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        return response.choices[0].message.content


# ==============================================================================
# DATA COLLECTION
# ==============================================================================

def collect_batch_responses(
    items: list[dict],
    models: list[str],
    conditions: list[str],
    batch_size: int,
    n_batches: int,
    output_dir: Path,
    client: BatchLLMClient,
) -> pd.DataFrame:
    """Collect batch responses."""

    output_file = output_dir / "batch_responses.jsonl"

    # Create batches
    import random

    all_records = []
    total_runs = len(models) * len(conditions) * n_batches

    with open(output_file, "w") as f:
        with tqdm(total=total_runs, desc="Collecting batches") as pbar:
            for model in models:
                for condition in conditions:
                    for batch_idx in range(n_batches):
                        # Randomly sample items for this batch
                        batch_items = random.sample(items, min(batch_size, len(items)))

                        # Generate prompt
                        prompt = format_batch_prompt(batch_items, condition)

                        # Call API
                        response = client.generate(prompt, model)

                        if response["error"]:
                            print(f"Error for {model}/{condition}/batch{batch_idx}: {response['error'][:50]}")
                            pbar.update(1)
                            continue

                        # Parse response
                        parsed = parse_batch_response(response["text"], len(batch_items))

                        # Create records for each item in batch
                        for i, (item, parse_result) in enumerate(zip(batch_items, parsed)):
                            record = {
                                "batch_id": f"{model}_{condition}_{batch_idx}",
                                "batch_idx": batch_idx,
                                "position_in_batch": i + 1,
                                "batch_size": len(batch_items),
                                "item_id": item["item_id"],
                                "model": model,
                                "model_tier": MODEL_CONFIGS[model]["tier"],
                                "gsm8k_score": MODEL_CONFIGS[model]["gsm8k"],
                                "condition": condition,
                                "parsed_answer": parse_result["parsed_answer"],
                                "reasoning": parse_result["reasoning"],
                                "correct_answer": item["correct_answer"],
                                "is_correct": parse_result["parsed_answer"] == item["correct_answer"] if parse_result["parsed_answer"] else None,
                                "raw_response": response["text"] if i == 0 else "[see first item in batch]",
                                "timestamp": datetime.now().isoformat(),
                            }

                            # Add misconception info
                            if "misconception_id" in item:
                                record["misconception_id"] = item["misconception_id"]
                                record["misconception_name"] = item.get("misconception_name", "")
                                record["misconception_category"] = item.get("misconception_category", "")
                                record["target_distractor"] = item.get("target_distractor", "")

                            f.write(json.dumps(record) + "\n")
                            all_records.append(record)

                        f.flush()
                        pbar.update(1)
                        time.sleep(0.5)  # Rate limiting

    return pd.DataFrame(all_records)


# ==============================================================================
# MAIN
# ==============================================================================

def load_items(items_path: Path) -> list[dict]:
    """Load items from JSON file."""
    with open(items_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif "items" in data:
        return data["items"]
    elif "misconceptions" in data:
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
                })
        return items
    else:
        raise ValueError("Unknown items format")


def main():
    parser = argparse.ArgumentParser(description="Collect batch LLM responses")
    parser.add_argument("--items", type=Path, required=True, help="Path to items JSON")
    parser.add_argument("--output", type=Path, default=Path("batch_results"), help="Output directory")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"], help="Models to use")
    parser.add_argument("--conditions", nargs="+", default=["answer_only", "explain", "persona"], help="Conditions: answer_only, explain, persona, diagnose_simulate")
    parser.add_argument("--batch-size", type=int, default=10, help="Items per batch")
    parser.add_argument("--n-batches", type=int, default=5, help="Number of batches per condition")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Validate models
    for model in args.models:
        if model not in MODEL_CONFIGS:
            available = list(MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model: {model}. Available: {available}")

    # Load items
    print(f"Loading items from {args.items}")
    items = load_items(args.items)
    print(f"Loaded {len(items)} items")

    # Initialize client
    client = BatchLLMClient()

    # Collect
    print(f"Collecting batch responses...")
    print(f"  Models: {args.models}")
    print(f"  Conditions: {args.conditions}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per condition: {args.n_batches}")
    print(f"  Total batches: {len(args.models) * len(args.conditions) * args.n_batches}")

    df = collect_batch_responses(
        items=items,
        models=args.models,
        conditions=args.conditions,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        output_dir=args.output,
        client=client,
    )

    # Summary
    print(f"\nCollection complete!")
    print(f"Total item-responses: {len(df)}")

    if len(df) == 0:
        print("No responses collected - check API keys and model availability")
        return

    print(f"Parse success: {df['parsed_answer'].notna().mean():.1%}")
    print(f"Accuracy: {df['is_correct'].mean():.1%}")

    # Position analysis
    print(f"\n=== Position Effect (preliminary) ===")
    pos_acc = df.groupby('position_in_batch')['is_correct'].mean()
    print(pos_acc)


if __name__ == "__main__":
    main()
