import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = Path(__file__).resolve().parent / "cricket_score_model_v2.json"

# 1. Load Data and Player Stats
df = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
bat_stats = pd.read_csv(DATASET_DIR / "batter_stats.csv")
bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats.csv")

# 2. Merge Stats into Main Data
# This maps the batter's skill to every single ball they faced
df = df.merge(bat_stats[['batter', 'strike_rate', 'average']], on='batter', how='left')
# This maps the bowler's skill to every ball they bowled
df = df.merge(bowl_stats[['bowler', 'economy']], on='bowler', how='left')

# Fill missing values (for players with no previous data) with medians
df['strike_rate'] = df['strike_rate'].fillna(df['strike_rate'].median())
df['average'] = df['average'].fillna(df['average'].median())
df['economy'] = df['economy'].fillna(df['economy'].median())

# 3. Feature Engineering (The standard logic)
df = df.sort_values(by=["match_id", "over", "ball"]).reset_index(drop=True)
df["runs_so_far"] = df.groupby("match_id")["runs_total"].cumsum()
df["wickets_so_far"] = df.groupby("match_id")["wicket_fallen"].cumsum()
df["balls_so_far"] = df.groupby("match_id").cumcount() + 1
df["overs_so_far"] = df["balls_so_far"] / 6.0
df["current_run_rate"] = (df["runs_so_far"] / df["overs_so_far"]).replace([np.inf, -np.inf], 0)
df["balls_remaining"] = 120 - df["balls_so_far"]
df["wickets_remaining"] = 10 - df["wickets_so_far"]

# Target: Final Score
final_scores = df.groupby("match_id")["runs_total"].sum().reset_index().rename(columns={"runs_total": "final_score"})
df = df.merge(final_scores, on="match_id")

# 4. Encoding Venue (Still needed as venues are fixed locations)
le_venue = LabelEncoder()
df["venue"] = le_venue.fit_transform(df["venue"].astype(str))

# 5. Define New Feature List
# Note: We REMOVED 'batter' and 'bowler' names and ADDED their stats
FEATURE_COLS = [
    "runs_so_far", "wickets_so_far", "balls_so_far", "current_run_rate",
    "balls_remaining", "wickets_remaining", "venue", 
    "strike_rate", "average", "economy"
]

X = df[FEATURE_COLS]
y = df["final_score"]

# 6. Train the New Brain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_v2 = xgb.XGBRegressor(
    n_estimators=300, 
    max_depth=6, 
    learning_rate=0.1, 
    objective="reg:squarederror"
)

print("Training Skill-Based Model...")
model_v2.fit(X_train, y_train)

# 7. Evaluate
preds = model_v2.predict(X_test)
new_mae = mean_absolute_error(y_test, preds)
print(f"New Model MAE: {new_mae:.2f}")

# Save the new version
model_v2.save_model(str(MODEL_PATH))