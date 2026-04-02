import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = Path(__file__).resolve().parent / "cricket_score_model_v2.json"

# 1. LOAD DATA & PLAYER SKILLS
df = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")

# 2. MERGE SKILLS (Mapping talent to every ball)
df = df.merge(bat_stats[['batter', 'strike_rate', 'average']], on='batter', how='left')
df = df.merge(bowl_stats[['bowler', 'economy']], on='bowler', how='left')

# Fill missing players with median stats
df['strike_rate'] = df['strike_rate'].fillna(df['strike_rate'].median())
df['average'] = df['average'].fillna(df['average'].median())
df['economy'] = df['economy'].fillna(df['economy'].median())

# 3. ADVANCED FEATURE ENGINEERING
df = df.sort_values(by=["match_id", "over", "ball"]).reset_index(drop=True)

# Basic Cumulative Progress
df["runs_so_far"] = df.groupby("match_id")["runs_total"].cumsum()
df["wickets_so_far"] = df.groupby("match_id")["wicket_fallen"].cumsum()
df["balls_so_far"] = df.groupby("match_id").cumcount() + 1
df["current_run_rate"] = (df["runs_so_far"] / (df["balls_so_far"] / 6.0)).replace([np.inf, -np.inf], 0)
df["balls_remaining"] = 120 - df["balls_so_far"]
df["wickets_remaining"] = 10 - df["wickets_so_far"]

# --- MAE IMPROVEMENT 1: MOMENTUM (Last 18 balls / 3 overs) ---
df["runs_last_3_overs"] = df.groupby("match_id")["runs_total"].transform(lambda x: x.rolling(18, 1).sum())
df["wickets_last_3_overs"] = df.groupby("match_id")["wicket_fallen"].transform(lambda x: x.rolling(18, 1).sum())

# --- MAE IMPROVEMENT 2: VENUE BASELINE (Target Encoding) ---
# Instead of a random ID, we use the average historical score of the ground
venue_means = df.groupby("venue")["runs_total"].sum() / df.groupby("venue")["match_id"].nunique()
df["venue_avg_score"] = df["venue"].map(venue_means)

# --- MAE IMPROVEMENT 3: PRESSURE INDEX (Matchup Interaction) ---
df["pressure_index"] = df["strike_rate"] * df["economy"]

# Get Final Target
final_scores = df.groupby("match_id")["runs_total"].sum().reset_index().rename(columns={"runs_total": "final_score"})
df = df.merge(final_scores, on="match_id")

# 4. SELECT FEATURES FOR TRAINING
FEATURE_COLS = [
    "runs_so_far", "wickets_so_far", "balls_so_far", "current_run_rate",
    "balls_remaining", "wickets_remaining", "venue_avg_score", 
    "strike_rate", "average", "economy",
    "runs_last_3_overs", "wickets_last_3_overs", "pressure_index"
]

X = df[FEATURE_COLS]
y = df["final_score"]

# 5. SPLIT & TUNE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Professional Hyperparameters to prevent Overfitting
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [300],
    'subsample': [0.8], # Use only 80% of data per tree to increase robustness
    'colsample_bytree': [0.8] # Use only 80% of features per tree
}

print("Running GridSearch for optimal MAE...")
grid = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid, cv=3, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)

# 6. RESULTS & SAVE
best_model = grid.best_estimator_
predictions = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"\n--- SUCCESS ---")
print(f"Final Model MAE: {mae:.2f} runs")
print(f"Best Parameters: {grid.best_params_}")

# Save the brain for your simulator
best_model.save_model(str(MODEL_PATH))

# Feature Importance Plot (To see what actually drives the score)
xgb.plot_importance(best_model)
plt.show()