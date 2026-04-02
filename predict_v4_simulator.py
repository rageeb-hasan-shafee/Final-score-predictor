#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import questionary
from rich.console import Console
from rich.table import Table

console = Console()
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "Xgboost"
INPUT_PATH = BASE_DIR / "input" / "innings_input.txt"

def run_debug_simulator():
    print("--- DEBUG: Script Started ---")
    
    # 1. Check Files
    files = [
        DATASET_DIR / "batter_stats_v2.csv",
        DATASET_DIR / "bowler_stats_v2.csv",
        DATASET_DIR / "final_ball_by_ball_first_innings.csv",
        INPUT_PATH,
    ]
    for f in files:
        if not f.exists():
            print(f"--- DEBUG ERROR: {f} NOT FOUND ---")
            return
    print("--- DEBUG: All files found ---")

    # 2. Load Data
    try:
        bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
        bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
        df_raw = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
        print("--- DEBUG: CSVs Loaded ---")
        
        le_v = LabelEncoder().fit(df_raw["venue"].astype(str))
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_DIR / "cricket_score_model_v2.json"))
        print("--- DEBUG: Model & Encoders Loaded ---")
    except Exception as e:
        print(f"--- DEBUG LOAD ERROR: {e} ---")
        return

    # 3. Read Input
    lines = [int(l.strip()) for l in INPUT_PATH.read_text().splitlines() if l.strip()]
    if not lines:
        print("--- DEBUG ERROR: innings_input.txt is EMPTY ---")
        return
        
    current_runs = sum(lines[0::2])
    current_wickets = sum(lines[1::2])
    current_balls = (len(lines)//2) * 6
    print(f"--- DEBUG: Current State: {current_runs}/{current_wickets} ({current_balls} balls) ---")

    # 4. Interactive Part
    try:
        venue = questionary.select("Select Venue:", choices=sorted(le_v.classes_)).ask()
        if not venue: return # Handle Ctrl+C
        
        bat_team = questionary.select("Select Batting Team:", choices=sorted(bat_stats['team'].unique().astype(str))).ask()
        team_batters = sorted(bat_stats[bat_stats['team'] == bat_team]['batter'].unique().tolist())
        
        striker = questionary.select("Select Striker:", choices=team_batters).ask()
        non_striker = questionary.select("Select Non-Striker:", choices=[b for b in team_batters if b != striker]).ask()
        
        bowl_team = questionary.select("Select Bowling Team:", choices=sorted(bowl_stats['team'].unique().astype(str))).ask()
        team_bowlers = sorted(bowl_stats[bowl_stats['team'] == bowl_team]['bowler'].unique().tolist())
    except Exception as e:
        print(f"--- DEBUG UI ERROR: {e} ---")
        return

    last_bowler = None
    
    # 5. Over Loop
    while current_balls < 120 and current_wickets < 10:
        over_num = (current_balls // 6) + 1
        console.print(f"\n[bold cyan]Over {over_num} | Score: {current_runs}/{current_wickets}[/bold cyan]")
        
        available_bowlers = [b for b in team_bowlers if b != last_bowler]
        current_bowler = questionary.select(f"Bowler for Over {over_num}:", choices=available_bowlers).ask()
        
        # Simple prediction logic for the demo
        current_balls += 6
        
        # Fetch stats safely
        s_sr = bat_stats[bat_stats['batter']==striker]['strike_rate'].values[0]
        b_eco = bowl_stats[bowl_stats['bowler']==current_bowler]['economy'].values[0]
        
        # Prepare feature vector
        input_data = pd.DataFrame([{
            "runs_so_far": current_runs, "wickets_so_far": current_wickets, "balls_so_far": current_balls,
            "current_run_rate": current_runs/(current_balls/6), "balls_remaining": 120-current_balls,
            "wickets_remaining": 10-current_wickets, "venue": le_v.transform([venue])[0],
            "strike_rate": s_sr, "average": 25.0, "economy": b_eco
        }])
        
        pred = int(model.predict(input_data)[0])
        console.print(f"[bold yellow]Projected Final Score: {pred}[/bold yellow]")
        
        action = questionary.select("Result of over?", choices=["Normal (Swap Strike)", "Wicket", "No Swap"]).ask()
        if action == "Wicket":
            current_wickets += 1
            if current_wickets < 10:
                striker = questionary.select("New Batter:", choices=[b for b in team_batters if b not in [striker, non_striker]]).ask()
        elif action == "Normal (Swap Strike)":
            striker, non_striker = non_striker, striker
            
        last_bowler = current_bowler

if __name__ == "__main__":
    run_debug_simulator()