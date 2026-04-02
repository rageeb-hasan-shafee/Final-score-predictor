# %% [markdown]
# # XGBoost Cricket Score Prediction (Corrected)
# This notebook implements a tuned XGBoost regressor for ball-by-ball score prediction.

# %% [code]
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# 1. SETUP & CONSTANTS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "dataset" / "final_ball_by_ball_first_innings.csv"
MODEL_PATH = Path(__file__).resolve().parent / "cricket_score_model.json"
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
best_model.save_model(str(MODEL_PATH))
print(f"Model saved successfully as {MODEL_PATH}")

# 7. VISUALIZATION
# Convert results list to DataFrame
res_df = pd.DataFrame(all_results)

# i. MAE vs Overs (at 5, 10, 15, and 20 overs)
# We expect the error to decrease as the match nears the end
mae_per_over = res_df.groupby('over')['mae'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=mae_per_over, x='over', y='mae', palette='viridis')
plt.title('Prediction Error (MAE) vs. Match Progress')
plt.xlabel('Overs Completed')
plt.ylabel('Mean Absolute Error (Runs)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('mae_vs_overs.png')

# ii. MAE vs Matches (Error Distribution Bins)
# We calculate the average MAE per match to see how many matches fall in each error range
match_errors = res_df.groupby('match_id')['mae'].mean()

# Define bins and labels (0-5, 6-10, 11-15, etc.)
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50', '50+']
match_errors_binned = pd.cut(match_errors, bins=bins, labels=labels, right=False).value_counts().sort_index()

plt.figure(figsize=(12, 6))
match_errors_binned.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Prediction Errors across Matches')
plt.xlabel('MAE Range (Runs)')
plt.ylabel('Number of Matches')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('mae_distribution.png')

print("Plots generated: mae_vs_overs.png and mae_distribution.png")