# %% [markdown]
# # XGBoost Cricket Score Prediction (Corrected)
# This notebook implements a tuned XGBoost regressor for ball-by-ball score prediction.

# %% [code]
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# 1. SETUP & CONSTANTS
FILE_PATH = "final_ball_by_ball_first_innings.csv"
OVERS_TO_PREDICT = [5, 10, 15, 20]

# 2. DATA LOADING & FEATURE ENGINEERING
df = pd.read_csv(FILE_PATH)
df = df.sort_values(by=["match_id", "over", "ball"]).reset_index(drop=True)

# Cumulative Progress
df["runs_so_far"] = df.groupby("match_id")["runs_total"].cumsum()
df["wickets_so_far"] = df.groupby("match_id")["wicket_fallen"].cumsum()
df["balls_so_far"] = df.groupby("match_id").cumcount() + 1
df["overs_so_far"] = df["balls_so_far"] / 6.0

# Critical Fix: Handle Division by Zero/Infinity for the first ball
df["current_run_rate"] = (df["runs_so_far"] / df["overs_so_far"]).replace([np.inf, -np.inf], 0)
df["balls_remaining"] = 120 - df["balls_so_far"]
df["wickets_remaining"] = 10 - df["wickets_so_far"]

# Target Variable: Final Score
final_score_map = df.groupby("match_id")["runs_total"].sum().reset_index()
final_score_map = final_score_map.rename(columns={"runs_total": "final_score"})
df = df.merge(final_score_map, on="match_id")

# 3. CATEGORICAL ENCODING
categorical_cols = ["venue", "batter", "bowler", "non_striker"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 4. TRAIN / TEST SPLIT (BY MATCH ID)
feature_cols = [
    "runs_so_far", "wickets_so_far", "balls_so_far", "overs_so_far",
    "current_run_rate", "balls_remaining", "wickets_remaining",
    "venue", "batter", "bowler", "non_striker"
]

np.random.seed(42)
match_ids = df["match_id"].unique()
np.random.shuffle(match_ids)
train_split = int(len(match_ids) * 0.8)

train_matches = match_ids[:train_split]
test_matches = match_ids[train_split:]

# Correcting the NameError by defining these before training
X_train = df[df["match_id"].isin(train_matches)][feature_cols]
y_train = df[df["match_id"].isin(train_matches)]["final_score"]
X_test = df[df["match_id"].isin(test_matches)][feature_cols]
y_test = df[df["match_id"].isin(test_matches)]["final_score"]

# 5. HYPERPARAMETER TUNING
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [300]
}

print("Starting Grid Search Tuning...")
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best Params: {grid_search.best_params_}")

# 6. EVALUATION AT SPECIFIC MILESTONES
all_results = []
for match in test_matches:
    match_data = df[df["match_id"] == match]
    for over in OVERS_TO_PREDICT:
        ball_idx = over * 6
        snapshot = match_data[match_data["balls_so_far"] >= ball_idx].head(1)
        
        if not snapshot.empty:
            pred = best_model.predict(snapshot[feature_cols])[0]
            actual = snapshot["final_score"].values[0]
            all_results.append({
                "match_id": match, "over": over, 
                "pred": round(pred), "actual": actual, 
                "mae": abs(actual - pred)
            })

res_df = pd.DataFrame(all_results)
print(f"\nFinal Tuned Test MAE: {res_df['mae'].mean():.2f}")   

# Save the model to a file
best_model.save_model("cricket_score_model.json")
print("Model saved successfully as cricket_score_model.json")