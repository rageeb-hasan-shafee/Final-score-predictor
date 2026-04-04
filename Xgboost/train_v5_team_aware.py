import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_PATH = SCRIPT_DIR / "cricket_team_model_v5.json"
MAE_OVERS_PLOT_PATH = OUTPUT_DIR / "mae_v5_phases.png"
MAE_MATCHES_PLOT_PATH = OUTPUT_DIR / "mae_v5_distribution.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. LOAD DATA
try:
    df = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
    bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
    bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure dataset files are in {DATASET_DIR}")
    exit()

# 2. CALCULATE TEAM-LEVEL AVERAGES (No Future Bias Logic)
def get_match_team_stats(df, bat_stats, bowl_stats):
    match_batters = df.groupby('match_id')['batter'].unique().reset_index()
    match_bowlers = df.groupby('match_id')['bowler'].unique().reset_index()

    def get_avg_bat(player_list):
        stats = bat_stats[bat_stats['batter'].isin(player_list)]
        return stats['strike_rate'].mean(), stats['average'].mean()

    def get_avg_bowl(player_list):
        stats = bowl_stats[bowl_stats['bowler'].isin(player_list)]
        return stats['economy'].mean()

    match_batters[['team_sr', 'team_avg']] = pd.DataFrame(match_batters['batter'].apply(get_avg_bat).tolist(), index=match_batters.index)
    match_bowlers['team_eco'] = match_bowlers['bowler'].apply(get_avg_bowl)

    df = df.merge(match_batters[['match_id', 'team_sr', 'team_avg']], on='match_id', how='left')
    df = df.merge(match_bowlers[['match_id', 'team_eco']], on='match_id', how='left')
    return df

df = get_match_team_stats(df, bat_stats, bowl_stats)

# 3. FEATURE ENGINEERING
df = df.sort_values(by=["match_id", "over", "ball"]).reset_index(drop=True)
df["runs_so_far"] = df.groupby("match_id")["runs_total"].cumsum()
df["wickets_so_far"] = df.groupby("match_id")["wicket_fallen"].cumsum()
df["balls_so_far"] = df.groupby("match_id").cumcount() + 1
df["current_run_rate"] = (df["runs_so_far"] / (df["balls_so_far"] / 6.0)).replace([np.inf, -np.inf], 0)
df["balls_remaining"] = 120 - df["balls_so_far"]
df["wickets_remaining"] = 10 - df["wickets_so_far"]
df["runs_last_3_overs"] = df.groupby("match_id")["runs_total"].transform(lambda x: x.rolling(18, 1).sum())

# Phase logic: 1: Powerplay, 2: Middle, 3: Death
df['match_phase'] = pd.cut(df['over'], bins=[-1, 6, 15, 20], labels=[1, 2, 3]).astype(int)

# Venue Stats
venue_means = df.groupby("venue")["runs_total"].sum() / df.groupby("venue")["match_id"].nunique()
df["venue_avg_score"] = df["venue"].map(venue_means)

# Target
final_scores = df.groupby("match_id")["runs_total"].sum().reset_index().rename(columns={"runs_total": "final_score"})
df = df.merge(final_scores, on="match_id")

# 4. TRAINING
FEATURE_COLS = [
    "runs_so_far", "wickets_so_far", "balls_so_far", "current_run_rate",
    "balls_remaining", "wickets_remaining", "venue_avg_score", 
    "team_sr", "team_avg", "team_eco", "match_phase", "runs_last_3_overs"
]

X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, objective='reg:squarederror')
model.fit(X_train, y_train)
model.save_model(str(MODEL_PATH))

# 5. EVALUATION & PHASED PLOTTING
test_df = X_test.copy()
test_df["actual"] = y_test
test_df["pred"] = model.predict(X_test)
test_df["mae"] = abs(test_df["actual"] - test_df["pred"])

# Define Over-Specific Bins (1-6, 6-10, 11-15, 16-20)
over_bins = [0, 6, 10, 15, 20]
over_labels = ["1-6 (Powerplay)", "6-10", "11-15", "16-20 (Death)"]
test_df['over_category'] = pd.cut(test_df['balls_so_far']/6.0, bins=over_bins, labels=over_labels, include_lowest=True)

# Plot 1: MAE vs Phased Overs
plt.figure(figsize=(10, 6))
phase_mae = test_df.groupby('over_category', observed=True)['mae'].mean()
phase_mae.plot(kind='bar', color='skyblue', edgecolor='black', width=0.7)
plt.title("MAE vs Match Phases (Team-Aware Model)", fontsize=14)
plt.ylabel("Mean Absolute Error (Runs)")
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(MAE_OVERS_PLOT_PATH)

# Plot 2: MAE Distribution
plt.figure(figsize=(10, 6))
dist_bins = [0, 5, 10, 15, 20, 30, 50, 100]
dist_labels = ["0-5", "6-10", "11-15", "16-20", "21-30", "31-50", "50+"]
mae_dist = pd.cut(test_df['mae'], bins=dist_bins, labels=dist_labels).value_counts().sort_index()
mae_dist.plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Match Distribution by MAE Range", fontsize=14)
plt.xlabel("MAE Range (Runs)")
plt.ylabel("Number of Predictions")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(MAE_MATCHES_PLOT_PATH)

# --- CONSOLE OUTPUT ---
print("\n" + "="*50)
print("             TRAINING SUMMARY")
print("="*50)
print(f"Final Model Test MAE : {test_df['mae'].mean():.2f} runs")
print("-"*50)
print("MAE by Phase:")
for phase, value in phase_mae.items():
    print(f"  {phase:<15} : {value:.2f}")
print("="*50)
print(f"SUCCESS: Model saved as {MODEL_PATH.name}")
print(f"SUCCESS: Phased plot saved as {MAE_OVERS_PLOT_PATH.name}")
print(f"SUCCESS: Distribution plot saved as {MAE_MATCHES_PLOT_PATH.name}")
print("="*50 + "\n")