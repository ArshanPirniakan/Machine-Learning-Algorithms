#!/usr/bin/env python3
"""Train and use a decision-tree-based learning-to-rank model.

Input CSV format:
    query_id, relevance, feature_1, feature_2, ...
Each row represents one candidate item for a query. Higher relevance values
indicate more relevant items. Feature columns must be numeric.

Examples:
    python decision_tree_ranker.py train --data ranking_data.csv --model ranker.joblib
    python decision_tree_ranker.py rank --data candidates.csv --model ranker.joblib --output ranked.csv

The training CSV requires query_id and relevance columns. The ranking CSV
requires query_id and the same feature columns used during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor


TARGET_COLUMN = "relevance"
GROUP_COLUMN = "query_id"
MODEL_VERSION = 1


def dcg(relevances: Iterable[float], k: int | None = None) -> float:
    values = np.asarray(list(relevances), dtype=float)
    if k is not None:
        values = values[:k]
    if values.size == 0:
        return 0.0
    gains = np.power(2.0, values) - 1.0
    discounts = np.log2(np.arange(2, values.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(actual: Iterable[float], predicted: Iterable[float], k: int = 10) -> float:
    actual_values = np.asarray(list(actual), dtype=float)
    predicted_values = np.asarray(list(predicted), dtype=float)
    order = np.argsort(-predicted_values, kind="stable")
    ideal = np.sort(actual_values)[::-1]
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg <= 0:
        return 0.0
    return dcg(actual_values[order], k) / ideal_dcg


def load_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    frame = pd.read_csv(file_path)
    if frame.empty:
        raise ValueError(f"CSV file is empty: {file_path}")
    return frame


def validate_training_data(frame: pd.DataFrame) -> list[str]:
    required = {GROUP_COLUMN, TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {sorted(missing)}")

    feature_columns = [
        column for column in frame.columns
        if column not in {GROUP_COLUMN, TARGET_COLUMN}
    ]
    if not feature_columns:
        raise ValueError("At least one numeric feature column is required.")

    non_numeric = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            "All feature columns must be numeric. Non-numeric columns: "
            + ", ".join(non_numeric)
        )

    if frame[feature_columns].isna().any().any():
        raise ValueError("Feature columns contain missing values. Clean or impute them first.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("The relevance column contains missing values.")
    if not np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain infinite values.")
    if not np.isfinite(frame[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("The relevance column contains infinite values.")
    if frame[GROUP_COLUMN].isna().any():
        raise ValueError("The query_id column contains missing values.")
    if (frame[TARGET_COLUMN] < 0).any():
        raise ValueError("Relevance values must be non-negative for NDCG evaluation.")
    return feature_columns


def evaluate(model: DecisionTreeRegressor, frame: pd.DataFrame, features: list[str]) -> dict:
    predictions = model.predict(frame[features])
    actual = frame[TARGET_COLUMN].to_numpy(dtype=float)
    grouped = frame[[GROUP_COLUMN, TARGET_COLUMN]].copy()
    grouped["prediction"] = predictions

    ndcg_scores = [
        ndcg_at_k(group[TARGET_COLUMN], group["prediction"], k=10)
        for _, group in grouped.groupby(GROUP_COLUMN, sort=False)
    ]

    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "mean_ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "evaluated_queries": int(len(ndcg_scores)),
        "evaluated_rows": int(len(frame)),
    }


def train(args: argparse.Namespace) -> None:
    frame = load_csv(args.data)
    features = validate_training_data(frame)

    unique_groups = frame[GROUP_COLUMN].nunique()
    if unique_groups < 2:
        raise ValueError("At least two distinct query_id groups are needed for a group-based split.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_indices, test_indices = next(
        splitter.split(frame[features], frame[TARGET_COLUMN], groups=frame[GROUP_COLUMN])
    )
    train_frame = frame.iloc[train_indices]
    test_frame = frame.iloc[test_indices]

    model = DecisionTreeRegressor(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    model.fit(train_frame[features], train_frame[TARGET_COLUMN])

    metrics = evaluate(model, test_frame, features)
    artifact = {
        "version": MODEL_VERSION,
        "model": model,
        "feature_columns": features,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "metrics": metrics,
        "parameters": {
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
        },
    }

    output_path = Path(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    print(json.dumps({
        "model_saved_to": str(output_path),
        "features": features,
        "metrics": metrics,
    }, indent=2))


def rank(args: argparse.Namespace) -> None:
    artifact_path = Path(args.model)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model file not found: {artifact_path}")

    artifact = joblib.load(artifact_path)
    if artifact.get("version") != MODEL_VERSION:
        raise ValueError("Unsupported or incompatible model artifact version.")

    frame = load_csv(args.data)
    if GROUP_COLUMN not in frame.columns:
        raise ValueError(f"Ranking CSV must contain a {GROUP_COLUMN} column.")

    features = artifact["feature_columns"]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Ranking CSV is missing trained feature columns: {missing}")

    non_numeric = [
        column for column in features
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"Ranking features must be numeric: {non_numeric}")
    if frame[features].isna().any().any():
        raise ValueError("Ranking feature columns contain missing values.")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Ranking feature columns contain infinite values.")

    result = frame.copy()
    result["predicted_relevance"] = artifact["model"].predict(result[features])
    result["rank"] = (
        result.groupby(GROUP_COLUMN)["predicted_relevance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    result = result.sort_values(
        [GROUP_COLUMN, "rank"],
        ascending=[True, True],
        kind="stable",
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Ranked results saved to: {output_path}")
    else:
        print(result.to_csv(index=False), end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and apply a decision-tree-based ranking model."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train and evaluate a ranker.")
    train_parser.add_argument("--data", required=True, help="Path to training CSV.")
    train_parser.add_argument("--model", default="decision_tree_ranker.joblib")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--max-depth", type=int, default=6)
    train_parser.add_argument("--min-samples-split", type=int, default=4)
    train_parser.add_argument("--min-samples-leaf", type=int, default=2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.set_defaults(func=train)

    rank_parser = commands.add_parser("rank", help="Score and rank candidate items.")
    rank_parser.add_argument("--data", required=True, help="Path to candidate CSV.")
    rank_parser.add_argument("--model", required=True, help="Trained .joblib model.")
    rank_parser.add_argument("--output", help="Optional path for ranked output CSV.")
    rank_parser.set_defaults(func=rank)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train" and not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    if args.command == "train" and args.max_depth < 1:
        parser.error("--max-depth must be at least 1.")
    if args.command == "train" and args.min_samples_split < 2:
        parser.error("--min-samples-split must be at least 2.")
    if args.command == "train" and args.min_samples_leaf < 1:
        parser.error("--min-samples-leaf must be at least 1.")

    try:
        args.func(args)
        return 0
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
