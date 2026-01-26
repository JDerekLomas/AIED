#!/usr/bin/env python3
"""
Replication 4: Feature Extraction + ML Baseline
Paper: Razavi & Powers (arXiv:2504.08804)

Research Question: Can LLM-extracted features + tree model beat direct estimation?

Method:
1. Extract 7 features per item via LLM:
   - Vocabulary complexity (1-5)
   - Syntax complexity (1-5)
   - Conceptual complexity (1-5)
   - Cognitive load (1-5)
   - DOK level (1-4)
   - Skill difficulty (1-5)
   - Distractor quality (1-5)
2. Train GBM on 80% of items
3. Evaluate correlation on 20% holdout

Key Metric: r = 0.87 (benchmark for reading; math varies 0.62-0.82)
"""

import json
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from dotenv import load_dotenv
import re

load_dotenv()

# =============================================================================
# API CLIENT
# =============================================================================

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI not installed")


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_PROMPT = """Analyze this math question and rate it on the following dimensions.
Return ONLY a JSON object with these exact keys and integer values.

Question: {question}

Options:
{options}

Correct Answer: {correct}

Rate each dimension (use ONLY integers, no decimals):

1. vocabulary_complexity (1-5): How difficult is the mathematical vocabulary used?
   1 = basic words (add, subtract), 5 = advanced terms (coefficient, quadratic)

2. syntax_complexity (1-5): How complex is the sentence structure and notation?
   1 = simple direct question, 5 = nested clauses, complex notation

3. conceptual_complexity (1-5): How many mathematical concepts are required?
   1 = single concept, 5 = multiple integrated concepts

4. cognitive_load (1-5): How much working memory is needed?
   1 = single step, obvious answer, 5 = many steps, tracking multiple values

5. dok_level (1-4): Depth of Knowledge (Webb's DOK)
   1 = recall, 2 = skill/concept, 3 = strategic thinking, 4 = extended thinking

6. skill_difficulty (1-5): How difficult is the underlying mathematical skill?
   1 = early elementary, 5 = advanced middle school or beyond

7. distractor_quality (1-5): How plausible/tricky are the wrong answers?
   1 = obviously wrong, 5 = very plausible common errors

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{"vocabulary_complexity": N, "syntax_complexity": N, "conceptual_complexity": N, "cognitive_load": N, "dok_level": N, "skill_difficulty": N, "distractor_quality": N}}"""


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(question: str, options: str, correct: str) -> dict:
    """Extract 7 features using GPT-4o."""
    client = OpenAI()

    prompt = FEATURE_PROMPT.format(
        question=question,
        options=options,
        correct=correct
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON
    try:
        # Remove markdown code blocks if present
        if raw.startswith("```"):
            raw = re.sub(r'^```json?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)

        features = json.loads(raw)

        # Validate all features present and in range
        required = ['vocabulary_complexity', 'syntax_complexity', 'conceptual_complexity',
                    'cognitive_load', 'dok_level', 'skill_difficulty', 'distractor_quality']

        for key in required:
            if key not in features:
                features[key] = 3  # Default middle value
            else:
                # Ensure integer in valid range
                val = int(features[key])
                if key == 'dok_level':
                    features[key] = max(1, min(4, val))
                else:
                    features[key] = max(1, min(5, val))

        return features

    except json.JSONDecodeError:
        print(f"  Failed to parse JSON: {raw[:100]}")
        return {
            'vocabulary_complexity': 3,
            'syntax_complexity': 3,
            'conceptual_complexity': 3,
            'cognitive_load': 3,
            'dok_level': 2,
            'skill_difficulty': 3,
            'distractor_quality': 3
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_eedi_items(csv_path: str, limit: int = None) -> list:
    """Load Eedi items from CSV."""
    df = pd.read_csv(csv_path)

    items = []
    for _, row in df.iterrows():
        if 'AnswerAText' in row:
            options = f"A) {row['AnswerAText']}\nB) {row['AnswerBText']}\nC) {row['AnswerCText']}\nD) {row['AnswerDText']}"

            # Calculate difficulty as 1 - accuracy
            correct = row['CorrectAnswer']
            correct_pct = row[f'pct_{correct}'] / 100
            difficulty = 1 - correct_pct

            item = {
                "id": row['QuestionId'],
                "question": row['QuestionText'],
                "options": options,
                "correct": correct,
                "difficulty": difficulty,
                "total_responses": row['total_responses']
            }
            items.append(item)

        if limit and len(items) >= limit:
            break

    return items


# =============================================================================
# ML PIPELINE
# =============================================================================

def train_difficulty_model(features_df: pd.DataFrame, test_size: float = 0.2):
    """Train GBM model to predict difficulty from features."""

    feature_cols = ['vocabulary_complexity', 'syntax_complexity', 'conceptual_complexity',
                    'cognitive_load', 'dok_level', 'skill_difficulty', 'distractor_quality']

    X = features_df[feature_cols]
    y = features_df['difficulty']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train GBM
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics - Train
    train_r, _ = stats.pearsonr(y_pred_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)

    # Metrics - Test
    test_r, test_p = stats.pearsonr(y_pred_test, y_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))

    results = {
        "train": {
            "n": len(X_train),
            "pearson_r": train_r,
            "rmse": train_rmse,
            "mae": train_mae,
        },
        "test": {
            "n": len(X_test),
            "pearson_r": test_r,
            "pearson_p": test_p,
            "rmse": test_rmse,
            "mae": test_mae,
        },
        "cv_rmse": cv_rmse,
        "feature_importance": importance,
    }

    return model, results


# =============================================================================
# DIRECT LLM BASELINE
# =============================================================================

def direct_difficulty_estimate(question: str, options: str, correct: str) -> float:
    """Get direct difficulty estimate from LLM."""
    client = OpenAI()

    prompt = f"""Estimate the difficulty of this math question for middle school students.
Return ONLY a number from 0 to 1, where:
- 0 = very easy (most students get it right)
- 1 = very hard (most students get it wrong)

Question: {question}

Options:
{options}

Correct Answer: {correct}

Difficulty (0-1):"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )

    raw = response.choices[0].message.content.strip()

    try:
        # Extract number
        match = re.search(r'[\d.]+', raw)
        if match:
            return float(match.group())
        return 0.5
    except:
        return 0.5


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_feature_extraction_experiment(
    items: list,
    output_dir: Path,
    skip_direct: bool = False
):
    """Run full feature extraction experiment."""

    print(f"\n{'='*60}")
    print("PHASE 1: Feature Extraction")
    print('='*60)

    results = []

    for i, item in enumerate(items):
        print(f"\nItem {i+1}/{len(items)}: {item['id']}")

        # Extract features
        features = extract_features(
            item['question'],
            item['options'],
            item['correct']
        )

        # Direct estimate (optional)
        if not skip_direct:
            direct_est = direct_difficulty_estimate(
                item['question'],
                item['options'],
                item['correct']
            )
            time.sleep(0.1)  # Rate limiting
        else:
            direct_est = None

        result = {
            "item_id": item['id'],
            "difficulty": item['difficulty'],
            "features": features,
            "direct_estimate": direct_est,
        }
        results.append(result)

        print(f"  Actual: {item['difficulty']:.2f}, Direct: {direct_est:.2f if direct_est else 'N/A'}")
        print(f"  Features: DOK={features['dok_level']}, Cog={features['cognitive_load']}, Skill={features['skill_difficulty']}")

        time.sleep(0.1)  # Rate limiting

        # Save intermediate
        if (i + 1) % 10 == 0:
            with open(output_dir / "features_intermediate.json", 'w') as f:
                json.dump(results, f, indent=2)

    # Save features
    features_file = output_dir / "features.json"
    with open(features_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def analyze_feature_extraction(results: list) -> dict:
    """Train model and analyze results."""

    print(f"\n{'='*60}")
    print("PHASE 2: ML Training & Analysis")
    print('='*60)

    # Build DataFrame
    rows = []
    for r in results:
        row = {
            "item_id": r['item_id'],
            "difficulty": r['difficulty'],
            **r['features']
        }
        if r['direct_estimate'] is not None:
            row['direct_estimate'] = r['direct_estimate']
        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\nDataset: {len(df)} items")
    print(f"Difficulty range: {df['difficulty'].min():.2f} - {df['difficulty'].max():.2f}")
    print(f"Difficulty mean: {df['difficulty'].mean():.2f}")

    # Feature correlations with difficulty
    feature_cols = ['vocabulary_complexity', 'syntax_complexity', 'conceptual_complexity',
                    'cognitive_load', 'dok_level', 'skill_difficulty', 'distractor_quality']

    print("\nFeature Correlations with Difficulty:")
    feature_corrs = {}
    for col in feature_cols:
        r, p = stats.pearsonr(df[col], df['difficulty'])
        feature_corrs[col] = {"r": r, "p": p}
        print(f"  {col}: r={r:.3f} (p={p:.4f})")

    # Train ML model
    print("\n" + "-"*40)
    print("GBM MODEL RESULTS")
    print("-"*40)

    model, ml_results = train_difficulty_model(df)

    print(f"\nTrain (n={ml_results['train']['n']}):")
    print(f"  Pearson r: {ml_results['train']['pearson_r']:.3f}")
    print(f"  RMSE: {ml_results['train']['rmse']:.3f}")

    print(f"\nTest (n={ml_results['test']['n']}):")
    print(f"  Pearson r: {ml_results['test']['pearson_r']:.3f} (p={ml_results['test']['pearson_p']:.4f})")
    print(f"  RMSE: {ml_results['test']['rmse']:.3f}")

    print(f"\n5-Fold CV RMSE: {ml_results['cv_rmse']:.3f}")

    print("\nFeature Importance:")
    for feat, imp in sorted(ml_results['feature_importance'].items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    # Direct estimate baseline (if available)
    if 'direct_estimate' in df.columns:
        direct_r, direct_p = stats.pearsonr(df['direct_estimate'], df['difficulty'])
        direct_rmse = np.sqrt(mean_squared_error(df['difficulty'], df['direct_estimate']))

        print(f"\nDirect LLM Estimate Baseline:")
        print(f"  Pearson r: {direct_r:.3f} (p={direct_p:.4f})")
        print(f"  RMSE: {direct_rmse:.3f}")

        ml_results['direct_baseline'] = {
            "pearson_r": direct_r,
            "pearson_p": direct_p,
            "rmse": direct_rmse,
        }

    print(f"\nBenchmark from Razavi & Powers: r=0.62-0.87 (domain dependent)")

    analysis = {
        "n_items": len(df),
        "feature_correlations": feature_corrs,
        "ml_results": ml_results,
    }

    return analysis


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Replicate Feature Extraction Study")
    parser.add_argument("--data", type=str,
                       default="/Users/dereklomas/AIED/study2-materials/data/eedi/eedi_with_student_data.csv",
                       help="Path to Eedi data CSV")
    parser.add_argument("--output", type=str, default="pilot/replications/feature_extraction",
                       help="Output directory")
    parser.add_argument("--items", type=int, default=None,
                       help="Limit number of items")
    parser.add_argument("--skip-direct", action="store_true",
                       help="Skip direct difficulty estimation baseline")
    parser.add_argument("--analyze-only", type=str, default=None,
                       help="Just analyze existing features file")

    args = parser.parse_args()

    output_dir = Path("/Users/dereklomas/AIED/study2-materials") / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        with open(args.analyze_only) as f:
            results = json.load(f)
        analysis = analyze_feature_extraction(results)
        with open(output_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=float)
        return

    print("="*60)
    print("FEATURE EXTRACTION + ML REPLICATION")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Item limit: {args.items or 'all'}")
    print(f"Skip direct baseline: {args.skip_direct}")
    print()

    # Load data
    items = load_eedi_items(args.data, limit=args.items)
    print(f"Loaded {len(items)} items")

    if len(items) == 0:
        print("No items loaded!")
        return

    # Run feature extraction
    results = run_feature_extraction_experiment(
        items=items,
        output_dir=output_dir,
        skip_direct=args.skip_direct
    )

    # Analyze
    analysis = analyze_feature_extraction(results)

    # Save analysis
    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=float)
    print(f"\nAnalysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
