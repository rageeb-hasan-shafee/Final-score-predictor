import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# ===========================================================
# 1. LOAD DATASET
# ===========================================================
file_path = "final_ball_by_ball_first_innings.csv"  
df = pd.read_csv(file_path)

# ===========================================================
# 2. SORT BALL-BY-BALL
# ===========================================================
df = df.sort_values(by=["match_id", "over", "ball"]).reset_index(drop=True)

# ===========================================================
# 3. FEATURE ENGINEERING
# ===========================================================
df["runs_so_far"] = df.groupby("match_id")["runs_total"].cumsum()
df["wickets_so_far"] = df.groupby("match_id")["wicket_fallen"].cumsum()

df["balls_so_far"] = df.groupby("match_id").cumcount() + 1
df["overs_so_far"] = df["balls_so_far"] / 6.0

df["current_run_rate"] = df["runs_so_far"] / df["overs_so_far"].replace(0, np.nan)

df["balls_remaining"] = 120 - df["balls_so_far"]
df["wickets_remaining"] = 10 - df["wickets_so_far"]

# ===========================================================
# 4. FINAL SCORE PER MATCH
# ===========================================================
final_score = df.groupby("match_id")["runs_total"].sum().reset_index()
final_score = final_score.rename(columns={"runs_total": "final_score"})
df = df.merge(final_score, on="match_id")

# ===========================================================
# 5. ENCODE CATEGORICAL FEATURES
# ===========================================================
categorical_cols = ["venue", "batter", "bowler", "non_striker"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ===========================================================
# 6. FEATURES FOR TRAINING
# ===========================================================
feature_cols = [
    "runs_so_far", "wickets_so_far", "balls_so_far", "overs_so_far",
    "current_run_rate", "balls_remaining", "wickets_remaining",
    "venue", "batter", "bowler", "non_striker"
]

# ===========================================================
# 7. TRAIN / TEST SPLIT
# ===========================================================
np.random.seed(42)
match_ids = df["match_id"].unique()
np.random.shuffle(match_ids)

train_ratio = 0.8
train_matches = match_ids[:int(len(match_ids) * train_ratio)]
test_matches = match_ids[int(len(match_ids) * train_ratio):]

train_df = df[df["match_id"].isin(train_matches)]
test_df = df[df["match_id"].isin(test_matches)]

X_train = train_df[feature_cols]
y_train = train_df["final_score"]

X_test = test_df[feature_cols]
y_test = test_df["final_score"]

# ===========================================================
# 8. TRAIN MODEL (XGBoost)
# ===========================================================
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cpu",
    random_state=42
)

model.fit(X_train, y_train)

# ===========================================================
# 9. PREDICT AT 5, 10, 15, 20 OVERS
# ===========================================================
overs_to_predict = [5, 10, 15, 20]
sample_matches = np.random.choice(test_matches, size=5, replace=False)

results = []

for match in sample_matches:
    match_df = df[df["match_id"] == match]

    for over in overs_to_predict:
        ball_number = over * 6

        snapshot = match_df[match_df["balls_so_far"] >= ball_number].head(1)
        if snapshot.empty:
            continue

        pred = model.predict(snapshot[feature_cols])[0]
        actual = snapshot["final_score"].values[0]

        results.append({
            "match_id": match,
            "overs_completed": over,
            "predicted_final_score": round(pred),
            "actual_final_score": actual,
            "difference": round(actual - pred),
        })

results_df = pd.DataFrame(results)
results_df["MAE"] = results_df.apply(
    lambda row: abs(row["difference"]), axis=1
)

print(results_df)


import matplotlib.pyplot as plt

# -----------------------------------------
# CREATE MAE BUCKETS
# -----------------------------------------
bins = [0, 10, 20, 30, 40, 50, 100, 200]  # adjust as needed
labels = ["0–10", "10–20", "20–30", "30–40", "40–50", "50–100", "100–200"]

results_df["MAE_bucket"] = pd.cut(results_df["MAE"], bins=bins, labels=labels, right=False)

# Frequency count
freq = results_df["MAE_bucket"].value_counts().sort_index()

# -----------------------------------------
# PLOT FREQUENCY GRAPH
# -----------------------------------------
plt.figure(figsize=(10,5))
plt.bar(freq.index.astype(str), freq.values)

plt.title("MAE Bucket Frequency Distribution")
plt.xlabel("MAE Range")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()
# ===========================================================
# 10. CALCULATE MAE FOR ALL TEST MATCHES
# ===========================================================
all_results = []

for match in test_matches:
    match_df = df[df["match_id"] == match]

    for over in overs_to_predict:
        ball_number = over * 6

        snapshot = match_df[match_df["balls_so_far"] >= ball_number].head(1)
        if snapshot.empty:
            continue

        pred = model.predict(snapshot[feature_cols])[0]
        actual = snapshot["final_score"].values[0]

        all_results.append({
            "match_id": match,
            "overs_completed": over,
            "predicted_final_score": round(pred),
            "actual_final_score": actual,
            "difference": round(actual - pred),
        })

all_results_df = pd.DataFrame(all_results)
all_results_df["MAE"] = all_results_df.apply(
    lambda row: abs(row["difference"]), axis=1
)

# -----------------------------------------
# CREATE MAE BUCKETS FOR ALL MATCHES
# -----------------------------------------
all_results_df["MAE_bucket"] = pd.cut(all_results_df["MAE"], bins=bins, labels=labels, right=False)

# Frequency count
all_freq = all_results_df["MAE_bucket"].value_counts().sort_index()

# -----------------------------------------
# PLOT FREQUENCY GRAPH FOR ALL MATCHES
# -----------------------------------------
plt.figure(figsize=(10,5))
plt.bar(all_freq.index.astype(str), all_freq.values, color='steelblue')

plt.title("MAE Bucket Frequency Distribution (All Test Matches)")
plt.xlabel("MAE Range")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\nTotal predictions: {len(all_results_df)}")
print(f"Mean MAE: {all_results_df['MAE'].mean():.2f}")
print(f"Median MAE: {all_results_df['MAE'].median():.2f}")