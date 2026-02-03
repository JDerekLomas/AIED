#!/usr/bin/env python3
"""
ML pipeline: structured LLM features → IRT parameter prediction.

Trains ridge/lasso/elasticnet/RF models on LLM-extracted features
to predict IRT parameters, with hold-out and LOO-CV evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

DEFAULT_OUTPUT_DIR = "pilot/structured_estimation"

RATING_TAGS = [
    "concept_complexity",
    "procedural_steps",
    "abstraction_level",
    "prerequisite_knowledge",
    "distractor_plausibility",
    "misconception_strength",
    "question_clarity",
    "cognitive_load",
    "age_appropriateness",
    "transfer_distance",
    "visual_complexity",
    "solution_uniqueness",
]

PREDICTION_TAGS = [
    "classical_difficulty",
    "irt_difficulty",
    "irt_discrimination",
    "guessing_probability",
]

TARGETS = {
    "b_2pl": "IRT Difficulty (b)",
    "a_2pl": "IRT Discrimination (a)",
    "c_3pl_guess": "Guessing (c)",
    "difficulty_classical": "Classical Difficulty",
}

# Maps direct LLM prediction tag to IRT target
DIRECT_PREDICTION_MAP = {
    "b_2pl": "irt_difficulty_mean",
    "a_2pl": "irt_discrimination_mean",
    "c_3pl_guess": "guessing_probability_mean",
    "difficulty_classical": "classical_difficulty_mean",
}

FEATURE_SETS = {
    "A_ratings_mean": [f"{t}_mean" for t in RATING_TAGS],
    "B_ratings_mean_std": [f"{t}_mean" for t in RATING_TAGS] + [f"{t}_std" for t in RATING_TAGS],
    "C_ratings_plus_direct": [f"{t}_mean" for t in RATING_TAGS] + [f"{t}_mean" for t in PREDICTION_TAGS],
    "D_all_28": (
        [f"{t}_mean" for t in RATING_TAGS]
        + [f"{t}_std" for t in RATING_TAGS]
        + [f"{t}_mean" for t in PREDICTION_TAGS]
    ),
}

MODELS = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=10000),
    "ElasticNet": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    "RF": lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
}


def load_data(output_dir):
    """Load features and IRT ground truth, split into train/holdout."""
    features = pd.read_csv(output_dir / "features.csv")
    split_info = json.loads((output_dir / "split_info.json").read_text())
    irt = json.loads(Path("results/irt_proper_statistics.json").read_text())

    # Build IRT dataframe
    irt_rows = []
    for qid_str, params in irt["items"].items():
        row = {"QuestionId": int(qid_str)}
        row.update(params)
        irt_rows.append(row)
    irt_df = pd.DataFrame(irt_rows)

    # Merge
    df = features.merge(irt_df, on="QuestionId", how="inner")

    train_qids = set(split_info["train_qids"])
    holdout_qids = set(split_info["holdout_qids"])

    train = df[df["QuestionId"].isin(train_qids)].copy()
    holdout = df[df["QuestionId"].isin(holdout_qids)].copy()

    return df, train, holdout


def evaluate(y_true, y_pred):
    """Compute evaluation metrics."""
    if len(y_true) < 3:
        return {"pearson_r": np.nan, "spearman_rho": np.nan, "rmse": np.nan, "mae": np.nan}
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"pearson_r": pearson_r, "spearman_rho": spearman_rho, "rmse": rmse, "mae": mae}


def run_holdout_evaluation(train, holdout, feature_cols, target_col, model_factory):
    """Train on train set, evaluate on holdout."""
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = holdout[feature_cols].values
    y_test = holdout[target_col].values

    # Handle NaN
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = model_factory()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    return evaluate(y_test, y_pred)


def run_loo_evaluation(df_all, feature_cols, target_col, model_factory):
    """Leave-one-out CV on all items."""
    X = df_all[feature_cols].values
    y = df_all[target_col].values

    if np.any(np.isnan(X)):
        return None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = model_factory()
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X_s, y, cv=loo)

    return evaluate(y, y_pred)


def baseline_direct_prediction(df, target_col):
    """Baseline: direct LLM prediction with no ML."""
    pred_col = DIRECT_PREDICTION_MAP.get(target_col)
    if pred_col is None or pred_col not in df.columns:
        return None
    valid = df[[target_col, pred_col]].dropna()
    if len(valid) < 3:
        return None
    return evaluate(valid[target_col].values, valid[pred_col].values)


def main(output_dir=None):
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
    else:
        output_dir = Path(output_dir)
    df_all, train, holdout = load_data(output_dir)
    print(f"All items: {len(df_all)}, Train: {len(train)}, Holdout: {len(holdout)}")

    results = []

    for target_col, target_label in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"Target: {target_label} ({target_col})")

        # Baseline
        for split_name, split_df in [("holdout", holdout), ("all_LOO", df_all)]:
            baseline = baseline_direct_prediction(split_df, target_col)
            if baseline:
                results.append({
                    "target": target_col,
                    "target_label": target_label,
                    "model": "Direct_LLM",
                    "features": "direct_prediction",
                    "eval_method": split_name,
                    **baseline,
                })
                print(f"  Baseline ({split_name}): r={baseline['pearson_r']:.3f} rho={baseline['spearman_rho']:.3f}")

        # ML models
        for feat_name, feat_cols in FEATURE_SETS.items():
            for model_name, model_factory in MODELS.items():
                # Holdout
                ho_result = run_holdout_evaluation(train, holdout, feat_cols, target_col, model_factory)
                if ho_result:
                    results.append({
                        "target": target_col,
                        "target_label": target_label,
                        "model": model_name,
                        "features": feat_name,
                        "eval_method": "holdout",
                        **ho_result,
                    })

                # LOO
                loo_result = run_loo_evaluation(df_all, feat_cols, target_col, model_factory)
                if loo_result:
                    results.append({
                        "target": target_col,
                        "target_label": target_label,
                        "model": model_name,
                        "features": feat_name,
                        "eval_method": "all_LOO",
                        **loo_result,
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

    # Generate report
    report = generate_report(results_df)
    (output_dir / "evaluation_report.md").write_text(report)
    print(f"\nReport saved to {output_dir / 'evaluation_report.md'}")


def generate_report(results_df):
    """Generate markdown evaluation report."""
    lines = [
        "# Structured IRT Estimation: Evaluation Report",
        f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for target_col, target_label in TARGETS.items():
        lines.append(f"\n## {target_label} (`{target_col}`)")

        for eval_method in ["holdout", "all_LOO"]:
            method_label = "Hold-out (20 items)" if eval_method == "holdout" else "LOO-CV (all items)"
            lines.append(f"\n### {method_label}")
            lines.append("")
            lines.append("| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |")
            lines.append("|-------|----------|-----------|------------|------|-----|")

            subset = results_df[
                (results_df["target"] == target_col) & (results_df["eval_method"] == eval_method)
            ].sort_values("pearson_r", ascending=False)

            for _, row in subset.iterrows():
                lines.append(
                    f"| {row['model']} | {row['features']} | "
                    f"{row['pearson_r']:.3f} | {row['spearman_rho']:.3f} | "
                    f"{row['rmse']:.3f} | {row['mae']:.3f} |"
                )

        # Best model summary
        best = results_df[
            (results_df["target"] == target_col) & (results_df["eval_method"] == "holdout")
        ].sort_values("pearson_r", ascending=False)
        if len(best) > 0:
            b = best.iloc[0]
            lines.append(f"\n**Best hold-out**: {b['model']} + {b['features']} (r={b['pearson_r']:.3f})")

            # Compare to baseline
            baseline = results_df[
                (results_df["target"] == target_col)
                & (results_df["eval_method"] == "holdout")
                & (results_df["model"] == "Direct_LLM")
            ]
            if len(baseline) > 0:
                bl = baseline.iloc[0]
                delta = b["pearson_r"] - bl["pearson_r"]
                lines.append(f"**vs Direct LLM**: Δr = {delta:+.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=DEFAULT_OUTPUT_DIR, help="Directory with features.csv and split_info.json")
    args = parser.parse_args()
    main(args.dir)
