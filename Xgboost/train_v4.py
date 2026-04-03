# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DATASET_DIR = PROJECT_ROOT / "dataset"
# OUTPUT_DIR = Path(__file__).resolve().parent
# MODEL_PATH = OUTPUT_DIR / "cricket_score_model_v4.json"
# MAE_PHASES_PLOT_PATH = OUTPUT_DIR / "mae_phases.png"
# MAE_DISTRIBUTION_PLOT_PATH = OUTPUT_DIR / "mae_distribution.png"

# # 1. LOAD & PREPARE
# df = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
# bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
# bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")

# # Merge Stats
# df = df.merge(bat_stats[['batter', 'strike_rate', 'average']], on='batter', how='left')
# df = df.merge(bowl_stats[['bowler', 'economy']], on='bowler', how='left')
# df[['strike_rate', 'average', 'economy']] = df[['strike_rate', 'average', 'economy']].fillna(df[['strike_rate', 'average', 'economy']].median())

# # 2. FEATURE ENGINEERING (INCLUDING PHASES)
# df = df.sort_values(by=["match_id", "over", "ball"]).reset_index(drop=True)

# # Define Match Phases
# # Phase 1: Powerplay (Overs 0-6)
# # Phase 2: Middle (Overs 6-15)
# # Phase 3: Death (Overs 15-20)
# df['match_phase'] = pd.cut(df['over'], bins=[-1, 6, 15, 20], labels=[1, 2, 3]).astype(int)

# # Standard Cumulative Features
# df["runs_so_far"] = df.groupby("match_id")["runs_total"].cumsum()
# df["wickets_so_far"] = df.groupby("match_id")["wicket_fallen"].cumsum()
# df["balls_so_far"] = df.groupby("match_id").cumcount() + 1
# df["current_run_rate"] = (df["runs_so_far"] / (df["balls_so_far"] / 6.0)).replace([np.inf, -np.inf], 0)
# df["balls_remaining"] = 120 - df["balls_so_far"]
# df["wickets_remaining"] = 10 - df["wickets_so_far"]

# # Momentum & Baseline
# df["runs_last_3_overs"] = df.groupby("match_id")["runs_total"].transform(lambda x: x.rolling(18, 1).sum())
# venue_means = df.groupby("venue")["runs_total"].sum() / df.groupby("venue")["match_id"].nunique()
# df["venue_avg_score"] = df["venue"].map(venue_means)

# # Target Variable
# final_scores = df.groupby("match_id")["runs_total"].sum().reset_index().rename(columns={"runs_total": "final_score"})
# df = df.merge(final_scores, on="match_id")

# # 3. TRAINING
# FEATURE_COLS = [
#     "runs_so_far", "wickets_so_far", "balls_so_far", "current_run_rate",
#     "balls_remaining", "wickets_remaining", "venue_avg_score", 
#     "strike_rate", "economy", "match_phase", "runs_last_3_overs"
# ]

# X = df[FEATURE_COLS]
# y = df["final_score"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1)
# model.fit(X_train, y_train)
# model.save_model(str(MODEL_PATH))

# # 4. EVALUATION & PLOTTING
# test_df = X_test.copy()
# test_df['actual'] = y_test
# test_df['pred'] = model.predict(X_test)
# test_df['mae'] = abs(test_df['actual'] - test_df['pred'])

# # i. MAE vs Overs (Phases)
# # Mapping numbers back to names for the plot
# phase_map = {1: 'Powerplay (1-6)', 2: 'Middle (7-15)', 3: 'Death (16-20)'}
# phase_mae = test_df.groupby('match_phase')['mae'].mean()
# phase_mae.index = phase_mae.index.map(phase_map)

# plt.figure(figsize=(10, 5))
# phase_mae.plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title('MAE across Match Phases')
# plt.ylabel('Mean Absolute Error (Runs)')
# plt.xticks(rotation=0)
# plt.grid(axis='y', alpha=0.3)
# plt.savefig(MAE_PHASES_PLOT_PATH)

# # ii. MAE vs Matches (Distribution)
# bins = [0, 5, 10, 15, 20, 30, 50, 100]
# labels = ['0-5', '6-10', '11-15', '16-20', '21-30', '31-50', '50+']
# # We look at the MAE at the end of each match prediction
# match_mae = test_df.groupby(test_df.index)['mae'].mean() 
# mae_bins = pd.cut(match_mae, bins=bins, labels=labels).value_counts().sort_index()

# plt.figure(figsize=(10, 5))
# mae_bins.plot(kind='bar', color='salmon', edgecolor='black')
# plt.title('Match Count by MAE Range')
# plt.xlabel('MAE (Runs)')
# plt.ylabel('Number of Predictions')
# plt.xticks(rotation=45)
# plt.grid(axis='y', alpha=0.3)
# plt.savefig(MAE_DISTRIBUTION_PLOT_PATH)

# print(f"Overall MAE: {test_df['mae'].mean():.2f}")
# print(f"Model saved successfully as {MODEL_PATH}")

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION & PATHS ---
# .parent.parent moves from /Xgboost/ to /Final-score-predictor/
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
DATASET_DIR = PROJECT_ROOT / "dataset"
# Save model inside the Xgboost folder
MODEL_PATH = PROJECT_ROOT / "Xgboost" / "cricket_team_model_v4.json"

print(f"Looking for data in: {DATASET_DIR}")

# --- 2. LOAD DATA ---
try:
    df = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
    bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
    bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
except FileNotFoundError as e:
    print(f"ERROR: Could not find files. Check your paths: {e}")
    exit()

# --- 3. MERGE PLAYER STATS ---
df = df.merge(bat_stats[['batter', 'strike_rate']], on='batter', how='left')
df = df.merge(bowl_stats[['bowler', 'economy']], on='bowler', how='left')

# Fill NaNs for players not in the stats CSV with medians
df['strike_rate'] = df['strike_rate'].fillna(df['strike_rate'].median())
df['economy'] = df['economy'].fillna(df['economy'].median())

# --- 4. FEATURE ENGINEERING ---
df = df.sort_values(by=["match_id", "over", "ball"]).reset_index(drop=True)

# Basic Match State
df["runs_so_far"] = df.groupby("match_id")["runs_total"].cumsum()
df["wickets_so_far"] = df.groupby("match_id")["wicket_fallen"].cumsum()
df["balls_so_far"] = df.groupby("match_id").cumcount() + 1
df["current_run_rate"] = (df["runs_so_far"] / (df["balls_so_far"] / 6.0)).replace([np.inf, -np.inf], 0)
df["balls_remaining"] = 120 - df["balls_so_far"]
df["wickets_remaining"] = 10 - df["wickets_so_far"]

# Momentum & Phase
df["runs_last_3_overs"] = df.groupby("match_id")["runs_total"].transform(lambda x: x.rolling(18, 1).sum())
df['match_phase'] = pd.cut(df['over'], bins=[-1, 6, 15, 20], labels=[1, 2, 3]).astype(int)

# Venue Average Score (Target Encoding)
venue_map = df.groupby("venue")["runs_total"].sum() / df.groupby("venue")["match_id"].nunique()
df["venue_avg_score"] = df["venue"].map(venue_map)

# Target Variable (Final Score of the innings)
final_scores = df.groupby("match_id")["runs_total"].sum().reset_index().rename(columns={"runs_total": "final_score"})
df = df.merge(final_scores, on="match_id")

# --- 5. TRAINING & SPLITTING ---
FEATURE_COLS = [
    'runs_so_far', 'wickets_so_far', 'balls_so_far', 'current_run_rate', 
    'balls_remaining', 'wickets_remaining', 'venue_avg_score', 
    'strike_rate', 'economy', 'match_phase', 'runs_last_3_overs'
]

X = df[FEATURE_COLS]
y = df["final_score"]

# The split ensures X_test exists for evaluation graphs later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
model.fit(X_train, y_train)

# Save the trained model
model.save_model(str(MODEL_PATH))
print(f"Model v4 saved successfully with {len(FEATURE_COLS)} features!")

# --- 6. EVALUATION & PLOTTING ---
test_df = X_test.copy()
test_df["actual"] = y_test
test_df["pred"] = model.predict(X_test)
test_df["mae"] = abs(test_df["actual"] - test_df["pred"])

# Prepare bins for Phase Graph
over_bins = [0, 6, 10, 15, 20]
over_labels = ["1-6 (PP)", "6-10", "11-15", "16-20 (Death)"]
test_df['over_category'] = pd.cut(test_df['balls_so_far']/6.0, bins=over_bins, labels=over_labels, include_lowest=True)

# Plot 1: MAE vs Match Phases
plt.figure(figsize=(10, 6))
phase_mae = test_df.groupby('over_category', observed=True)['mae'].mean()
phase_mae.plot(kind='bar', color='skyblue', edgecolor='black', width=0.7)
plt.title("v4 Model: MAE vs Match Phases", fontsize=14)
plt.ylabel("Mean Absolute Error (Runs)")
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Xgboost" / "mae_v4_phases.png")

# Plot 2: Error Distribution
plt.figure(figsize=(10, 6))
dist_bins = [0, 5, 10, 15, 20, 30, 50, 100]
dist_labels = ["0-5", "6-10", "11-15", "16-20", "21-30", "31-50", "50+"]
mae_dist = pd.cut(test_df['mae'], bins=dist_bins, labels=dist_labels).value_counts().sort_index()
mae_dist.plot(kind='bar', color='salmon', edgecolor='black')
plt.title("v4 Model: Match Distribution by MAE", fontsize=14)
plt.xlabel("MAE Range (Runs)")
plt.ylabel("Number of Predictions")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Xgboost" / "mae_v4_distribution.png")

# --- 7. CONSOLE OUTPUT ---
print("\n" + "="*50)
print("             V4 TRAINING SUMMARY")
print("="*50)
print(f"Final Model Test MAE : {test_df['mae'].mean():.2f} runs")
print("-"*50)
print("MAE by Phase:")
for phase, value in phase_mae.items():
    print(f"  {phase:<15} : {value:.2f}")
print("="*50)
print(f"SUCCESS: Phased plot saved as mae_v4_phases.png")
print(f"SUCCESS: Distribution plot saved as mae_v4_distribution.png")
print("="*50 + "\n")