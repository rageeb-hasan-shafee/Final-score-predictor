import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BASE_DIR = Path(__file__).resolve().parent / "Xgboost"
MODEL_PATH = BASE_DIR / "cricket_team_model_v5.json"

def get_team_averages(player_list, stats_df, stat_cols, player_col="batter"):
    # Filter stats for only the players in the provided list
    # Clean whitespace from player names in case of user typing errors
    player_list = [p.strip() for p in player_list]
    team_stats = stats_df[stats_df[player_col].isin(player_list)]
    
    # Return averages, or default values if players aren't found
    averages = team_stats[stat_cols].mean().to_dict()
    return averages

def predict_match_score():
    # 1. Load Resources
    if not MODEL_PATH.exists():
        console.print(f"[bold red]Error: Model file not found at {MODEL_PATH}[/bold red]")
        console.print("[yellow]Please run your training script (train_v5_team.py) first.[/yellow]")
        return

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    
    try:
        bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
        bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
        df_raw = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
    except FileNotFoundError as e:
        console.print(f"[bold red]Data Error: {e}[/bold red]")
        return
    
    # Train LabelEncoder for Venue mapping
    le_v = LabelEncoder().fit(df_raw["venue"].astype(str))
    
    console.print(Panel("[bold cyan]T20 Match Real-Time Predictor (Team-Aware)[/bold cyan]"))

    # 2. Manual Inputs (Current Match State)
    venue = input("Enter Venue: ")
    venue_id = le_v.transform([venue])[0] if venue in le_v.classes_ else 0
    
    curr_runs = int(input("Current Runs Scored: "))
    curr_wickets = int(input("Current Wickets Lost: "))
    curr_overs = float(input("Current Overs Completed (e.g. 10.2): "))
    runs_3_overs = int(input("Runs scored in last 3 overs (Momentum): "))
    
    # Convert overs to balls
    whole_overs = int(curr_overs)
    extra_balls = int(round((curr_overs - whole_overs) * 10))
    total_balls_done = (whole_overs * 6) + extra_balls
    
    # 3. Playing XI Inputs
    print("\n--- Team Info ---")
    batting_team_xi = input("Enter Batting Playing 11 (comma separated): ").split(",")
    bowling_team_xi = input("Enter Bowling Playing 11 (comma separated): ").split(",")

    # 4. Calculate "Future Potential" Features
    # Use remaining batters for batting strength
    remaining_batters = batting_team_xi[curr_wickets:] 
    bat_skills = get_team_averages(remaining_batters, bat_stats, ['strike_rate', 'average'], "batter")
    
    # Use full bowling unit for bowling strength
    bowl_skills = get_team_averages(bowling_team_xi, bowl_stats, ['economy'], "bowler")

    # 5. Build Feature Vector (COLUMN NAMES MUST MATCH TRAINING)
    over_num = whole_overs
    # 1: Powerplay, 2: Middle, 3: Death
    phase = 1 if over_num < 6 else (2 if over_num < 15 else 3)
    
    # Create input for XGBoost with the EXACT names used in train_v5
    input_row = pd.DataFrame([{
        "runs_so_far": curr_runs,
        "wickets_so_far": curr_wickets,
        "balls_so_far": total_balls_done,
        "current_run_rate": curr_runs / (total_balls_done / 6) if total_balls_done > 0 else 0,
        "balls_remaining": 120 - total_balls_done,
        "wickets_remaining": 10 - curr_wickets,
        "venue_avg_score": 165.0, 
        "team_sr": bat_skills.get('strike_rate', 125.0), # Updated name
        "team_avg": bat_skills.get('average', 25.0),    # Updated name
        "team_eco": bowl_skills.get('economy', 8.5),    # Updated name
        "match_phase": phase,
        "runs_last_3_overs": runs_3_overs
    }])

    # Ensure the columns are in the exact same order as the training features
    feature_names = model.get_booster().feature_names
    input_row = input_row[feature_names]

    # 6. Predict
    prediction = model.predict(input_row)[0]
    
    console.print(Panel(f"[bold green]Predicted Final Total: {int(round(prediction))}[/bold green]"))

if __name__ == "__main__":
    predict_match_score()

# def get_team_averages(player_list, stats_df, stat_cols, type="batter"):
#     # Filter stats for only the players in the Playing 11
#     team_stats = stats_df[stats_df[type].isin(player_list)]
#     return team_stats[stat_cols].mean().to_dict()

# def predict_match_score():
#     # 1. Load Resources
#     model = xgb.XGBRegressor()
#     model.load_model(str(MODEL_PATH))
    
#     bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
#     bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
#     df_raw = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
    
#     le_v = LabelEncoder().fit(df_raw["venue"].astype(str))
    
#     console.print(Panel("[bold cyan]T20 Match Real-Time Predictor[/bold cyan]"))

#     # 2. Manual Inputs (Current Match State)
#     venue = input("Enter Venue: ")
#     venue_id = le_v.transform([venue])[0] if venue in le_v.classes_ else 0
    
#     curr_runs = int(input("Current Runs Scored: "))
#     curr_wickets = int(input("Current Wickets Lost: "))
#     curr_overs = float(input("Current Overs Completed (e.g. 10.2): "))
    
#     # Convert overs to balls
#     whole_overs = int(curr_overs)
#     extra_balls = int(round((curr_overs - whole_overs) * 10))
#     total_balls_done = (whole_overs * 6) + extra_balls
    
#     # 3. Playing XI Inputs
#     # Note: In a real app, you'd load these from a list, but here we take names
#     print("\n--- Batting Team Info ---")
#     batting_team_xi = input("Enter Playing 11 Batters (comma separated): ").split(",")
#     # Identify who is out based on wickets lost (assuming top order falls first for simplicity)
#     remaining_batters = batting_team_xi[curr_wickets:] 
    
#     print("\n--- Bowling Team Info ---")
#     bowling_team_xi = input("Enter Playing 11 Bowlers (comma separated): ").split(",")

#     # 4. Calculate "Future Potential" Features
#     # We use the average stats of the remaining batting lineup
#     bat_skills = get_team_averages(remaining_batters, bat_stats, ['strike_rate', 'average'], "batter")
    
#     # We use the average economy of the whole bowling unit
#     bowl_skills = get_team_averages(bowling_team_xi, bowl_stats, ['economy'], "bowler")

#     # 5. Build Feature Vector
#     # Calculate match phase
#     over_num = whole_overs + 1
#     phase = 1 if over_num <= 6 else (2 if over_num <= 15 else 3)
    
#     # Create input for XGBoost
#     input_row = pd.DataFrame([{
#         "runs_so_far": curr_runs,
#         "wickets_so_far": curr_wickets,
#         "balls_so_far": total_balls_done,
#         "current_run_rate": curr_runs / (total_balls_done / 6) if total_balls_done > 0 else 0,
#         "balls_remaining": 120 - total_balls_done,
#         "wickets_remaining": 10 - curr_wickets,
#         "venue_avg_score": 165.0, # You can map this to venue_id from your training data
#         "strike_rate": bat_skills.get('strike_rate', 120),
#         "average": bat_skills.get('average', 25),
#         "economy": bowl_skills.get('economy', 8.0),
#         "match_phase": phase,
#         "runs_last_3_overs": curr_runs * 0.15 # Estimate or ask user for actual last 3 overs runs
#     }])

#     # 6. Predict
#     prediction = model.predict(input_row[model.get_booster().feature_names])[0]
    
#     console.print(f"\n[bold green]Predicted Final Total: {int(round(prediction))}[/bold green]")

# if __name__ == "__main__":
#     predict_match_score()