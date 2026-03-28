#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "runs_so_far",
    "wickets_so_far",
    "balls_so_far",
    "overs_so_far",
    "current_run_rate",
    "balls_remaining",
    "wickets_remaining",
    "venue",
    "batter",
    "bowler",
    "non_striker",
]


def build_snapshot_from_txt(input_path: Path, max_overs: int = 20) -> tuple[int, int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Input file is empty.")
    if len(lines) % 2 != 0:
        raise ValueError("Input must be runs/wickets pairs per over (even number of non-empty lines).")

    current_over = len(lines) // 2
    if current_over > max_overs:
        raise ValueError(f"Input has {current_over} overs, but max_overs is {max_overs}.")

    total_runs = 0
    total_wickets = 0

    for over_idx in range(current_over):
        runs_line = lines[2 * over_idx]
        wickets_line = lines[2 * over_idx + 1]

        try:
            runs = int(runs_line)
            wickets = int(wickets_line)
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer at over {over_idx + 1}: runs='{runs_line}', wickets='{wickets_line}'"
            ) from exc

        if not (0 <= runs <= 60):
            raise ValueError(f"Runs out of range at over {over_idx + 1}: {runs} (allowed 0-60)")
        if not (0 <= wickets <= 10):
            raise ValueError(f"Wickets out of range at over {over_idx + 1}: {wickets} (allowed 0-10)")

        total_runs += runs
        total_wickets += wickets
        if total_wickets > 10:
            raise ValueError("Total wickets cannot exceed 10.")

    balls_so_far = current_over * 6
    return total_runs, total_wickets, balls_so_far


def fit_label_encoders(dataset_path: Path, categorical_cols: list[str]) -> dict[str, LabelEncoder]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, usecols=categorical_cols)
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def encode_category(encoders: dict[str, LabelEncoder], col: str, value: str) -> int:
    le = encoders[col]
    if value not in le.classes_:
        raise ValueError(
            f"Unknown {col}: '{value}'. Pick a value from the training dataset."
        )
    return int(le.transform([value])[0])


def build_feature_row(
    runs_so_far: int,
    wickets_so_far: int,
    balls_so_far: int,
    venue_code: int,
    batter_code: int,
    bowler_code: int,
    non_striker_code: int,
) -> pd.DataFrame:
    overs_so_far = balls_so_far / 6.0
    current_run_rate = (runs_so_far / overs_so_far) if overs_so_far > 0 else 0.0
    balls_remaining = 120 - balls_so_far
    wickets_remaining = 10 - wickets_so_far

    row = {
        "runs_so_far": runs_so_far,
        "wickets_so_far": wickets_so_far,
        "balls_so_far": balls_so_far,
        "overs_so_far": overs_so_far,
        "current_run_rate": current_run_rate,
        "balls_remaining": balls_remaining,
        "wickets_remaining": wickets_remaining,
        "venue": venue_code,
        "batter": batter_code,
        "bowler": bowler_code,
        "non_striker": non_striker_code,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


def heuristic_prediction(runs_so_far: int, balls_so_far: int) -> float:
    overs_so_far = balls_so_far / 6.0
    run_rate = (runs_so_far / overs_so_far) if overs_so_far > 0 else 0.0
    overs_left = 20.0 - overs_so_far
    return runs_so_far + run_rate * overs_left


def choose_with_index(name: str, classes: np.ndarray, default: str | None = None) -> str:
    print(f"\nSelect {name} (type number):")
    for i, item in enumerate(classes, start=1):
        print(f"  {i}. {item}")

    if default is not None and default in classes:
        default_idx = int(np.where(classes == default)[0][0]) + 1
    else:
        default_idx = 1

    while True:
        raw = input(f"{name} index [{default_idx}]: ").strip()
        if raw == "":
            idx = default_idx
        else:
            try:
                idx = int(raw)
            except ValueError:
                print("Please enter a valid integer index.")
                continue

        if 1 <= idx <= len(classes):
            return str(classes[idx - 1])

        print(f"Please enter a value between 1 and {len(classes)}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict final score using your XGBoost model.")
    parser.add_argument(
        "--input",
        type=str,
        default="innings_input.txt",
        help="Path to txt input (runs/wickets per over).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cricket_score_model.json",
        help="Path to saved XGBoost model JSON.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="final_ball_by_ball_first_innings.csv",
        help="Dataset used to fit categorical encoders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    dataset_path = Path(args.data)

    print("=== Final Score Predictor (XGBoost) ===")
    print("Input format: runs and wickets lines per over (same as your old CLI).")

    try:
        runs_so_far, wickets_so_far, balls_so_far = build_snapshot_from_txt(input_path)
    except Exception as exc:
        print(f"Failed to parse input file: {exc}")
        return

    print(f"\nCurrent score snapshot:")
    print(f"Runs so far      : {runs_so_far}")
    print(f"Wickets so far   : {wickets_so_far}")
    print(f"Balls so far     : {balls_so_far}")
    print(f"Overs completed  : {balls_so_far / 6:.1f}")

    categorical_cols = ["venue", "batter", "bowler", "non_striker"]
    try:
        encoders = fit_label_encoders(dataset_path, categorical_cols)
    except Exception as exc:
        print(f"Failed to load categorical values from dataset: {exc}")
        print("Falling back to heuristic prediction only.")
        pred = heuristic_prediction(runs_so_far, balls_so_far)
        print(f"\nPredicted Final Score (heuristic): {int(round(pred))}")
        return

    venue = choose_with_index("venue", encoders["venue"].classes_)
    batter = choose_with_index("batter", encoders["batter"].classes_)
    bowler = choose_with_index("bowler", encoders["bowler"].classes_)
    non_striker = choose_with_index("non_striker", encoders["non_striker"].classes_)

    try:
        venue_code = encode_category(encoders, "venue", venue)
        batter_code = encode_category(encoders, "batter", batter)
        bowler_code = encode_category(encoders, "bowler", bowler)
        non_striker_code = encode_category(encoders, "non_striker", non_striker)
    except Exception as exc:
        print(f"Failed to encode categorical input: {exc}")
        return

    x_row = build_feature_row(
        runs_so_far=runs_so_far,
        wickets_so_far=wickets_so_far,
        balls_so_far=balls_so_far,
        venue_code=venue_code,
        batter_code=batter_code,
        bowler_code=bowler_code,
        non_striker_code=non_striker_code,
    )

    if model_path.exists():
        try:
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(model_path)
            pred = float(model.predict(x_row)[0])
            print(f"\nPredicted Final Score (XGBoost): {int(round(pred))}")
            return
        except Exception as exc:
            print(f"Model prediction failed: {exc}")
            print("Falling back to heuristic prediction.")
    else:
        print(f"Model not found at {model_path}. Falling back to heuristic prediction.")

    pred = heuristic_prediction(runs_so_far, balls_so_far)
    print(f"\nPredicted Final Score (heuristic): {int(round(pred))}")


if __name__ == "__main__":
    main()
