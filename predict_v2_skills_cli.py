# #!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from pathlib import Path
# import questionary
# from rich.console import Console
# from rich.table import Table
# from rich.panel import Panel

# console = Console()

# def load_resources():
#     """Loads the model, venue list, and player statistics."""
#     try:
#         bat_stats = pd.read_csv("batter_stats.csv").set_index("batter")
#         bowl_stats = pd.read_csv("bowler_stats.csv").set_index("bowler")
#         venue_info = pd.read_csv("venue_nature.csv")
        
#         # Load the Venue Encoder (Same as used in training)
#         df_raw = pd.read_csv("final_ball_by_ball_first_innings.csv")
#         from sklearn.preprocessing import LabelEncoder
#         le_venue = LabelEncoder()
#         le_venue.fit(df_raw["venue"].astype(str))
        
#         return bat_stats, bowl_stats, venue_info, le_venue
#     except Exception as e:
#         console.print(f"[red]Error loading CSV resources:[/red] {e}")
#         return None, None, None, None

# def build_match_status(input_path: Path):
#     """Parses the innings_input.txt file."""
#     raw_lines = input_path.read_text(encoding="utf-8").splitlines()
#     lines = [int(line.strip()) for line in raw_lines if line.strip() != ""]
#     runs = sum(lines[0::2])
#     wickets = sum(lines[1::2])
#     overs = len(lines) // 2
#     balls = overs * 6
#     return {
#         "runs_so_far": runs,
#         "wickets_so_far": wickets,
#         "balls_so_far": balls,
#         "current_run_rate": runs / overs if overs > 0 else 0,
#         "balls_remaining": 120 - balls,
#         "wickets_remaining": 10 - wickets
#     }, overs

# def run_v2_cli():

    
#     console.print(Panel("[bold green]Cricket Predictor V2: Skill-Based Engine[/bold green]", expand=False))

#     bat_stats, bowl_stats, venue_info, le_venue = load_resources()
#     if bat_stats is None: return

#     # 1. Parse TXT
#     status, over_count = build_match_status(Path("innings_input.txt"))

#     # 2. User Inputs (Venue & Players)
#     venue = questionary.select("Select Venue:", choices=sorted(le_venue.classes_), use_shortcuts=False).ask()
#     batter = questionary.select("Select Striker:", choices=sorted(bat_stats.index), use_shortcuts=False).ask()
#     bowler = questionary.select("Select Bowler:", choices=sorted(bowl_stats.index), use_shortcuts=False).ask()

#     # 3. Lookup Skills
#     b_sr = bat_stats.loc[batter, "strike_rate"]
#     b_avg = bat_stats.loc[batter, "average"]
#     bowl_eco = bowl_stats.loc[bowler, "economy"]
#     venue_id = le_venue.transform([venue])[0]

#     # 4. Prepare Features (Order must match training FEATURE_COLS)
#     input_data = pd.DataFrame([{
#         "runs_so_far": status["runs_so_far"],
#         "wickets_so_far": status["wickets_so_far"],
#         "balls_so_far": status["balls_so_far"],
#         "current_run_rate": status["current_run_rate"],
#         "balls_remaining": status["balls_remaining"],
#         "wickets_remaining": status["wickets_remaining"],
#         "venue": venue_id,
#         "strike_rate": b_sr,
#         "average": b_avg,
#         "economy": bowl_eco
#     }])

#     # 5. Predict
#     model_path = "cricket_score_model_v2.json"
#     if Path(model_path).exists():
#         model = xgb.XGBRegressor()
#         model.load_model(model_path)
#         prediction = model.predict(input_data)[0]
        
#         # Display Results
#         table = Table(title="Match Intelligence Result")
#         table.add_column("Factor", style="cyan")
#         table.add_column("Value", style="magenta")
#         table.add_row("Venue Profile", venue)
#         table.add_row("Batter Skill (SR)", f"{b_sr:.2f}")
#         table.add_row("Bowler Skill (Eco)", f"{bowl_eco:.2f}")
#         table.add_row("Current State", f"{status['runs_so_far']}/{status['wickets_so_far']} ({over_count} ov)")
#         console.print(table)
        
#         console.print(f"\n[bold yellow]PREDICTED FINAL SCORE: {int(round(prediction))}[/bold yellow]\n")
#     else:
#         console.print("[red]Model v2 file not found! Run training first.[/red]")

# if __name__ == "__main__":
#     run_v2_cli()

#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

try:
    import questionary
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("Missing dependencies. Run: pip install questionary rich pandas xgboost scikit-learn")
    raise

console = Console()
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "Xgboost"
INPUT_PATH = BASE_DIR / "input" / "innings_input.txt"

def load_all_resources():
    """Loads all CSV stats, the venue encoder, and the XGBoost model."""
    try:
        # 1. Load DataFrames
        bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
        bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
        venue_info = pd.read_csv(DATASET_DIR / "venue_nature.csv")
        
        # 2. Re-create Venue Encoder for consistency
        df_raw = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
        le_venue = LabelEncoder()
        le_venue.fit(df_raw["venue"].astype(str))
        
        # 3. Load Model
        model = xgb.XGBRegressor()
        model_path = MODEL_DIR / "cricket_score_model_v2.json"
        if model_path.exists():
            model.load_model(str(model_path))
        else:
            return None, None, None, None, None
            
        return bat_stats, bowl_stats, venue_info, le_venue, model
    except Exception as e:
        console.print(f"[red]Resource Loading Error:[/red] {e}")
        return None, None, None, None, None

def get_match_status_from_txt(path: Path):
    """Parses runs/wickets from innings_input.txt."""
    lines = [int(l.strip()) for l in path.read_text().splitlines() if l.strip()]
    runs = sum(lines[0::2])
    wickets = sum(lines[1::2])
    overs = len(lines) // 2
    balls = overs * 6
    return {
        "runs_so_far": runs,
        "wickets_so_far": wickets,
        "balls_so_far": balls,
        "current_run_rate": runs / overs if overs > 0 else 0,
        "balls_remaining": 120 - balls,
        "wickets_remaining": 10 - wickets
    }, overs

def run_final_cli():
    console.print(Panel("[bold cyan]Cricket Predictor Pro: Team-Filtered Edition[/bold cyan]", expand=False))
    
    # Load resources
    bat_stats, bowl_stats, venue_info, le_v, model = load_all_resources()
    if model is None:
        console.print("[red]Error: Model V2 or CSVs not found. Please train v2 first.[/red]")
        return

    # 1. Parse current match state
    status, over_count = get_match_status_from_txt(INPUT_PATH)

    # 2. Venue Selection
    venue = questionary.select("Select Match Venue:", choices=sorted(le_v.classes_), use_shortcuts=False).ask()
    v_nature = venue_info[venue_info['venue'] == venue]['nature'].values[0]

    # 3. Batter Selection (Team Filtered)
    bat_team = questionary.select("Select Batting Team:", choices=sorted(bat_stats['team'].unique())).ask()
    team_batters = bat_stats[bat_stats['team'] == bat_team]['batter'].unique().tolist()
    batter_name = questionary.select(f"Select Batter from {bat_team}:", choices=sorted(team_batters)).ask()
    
    # Get Batter Stats
    b_row = bat_stats[bat_stats['batter'] == batter_name].iloc[0]
    
    # 4. Bowler Selection (Team Filtered)
    bowl_team = questionary.select("Select Bowling Team:", choices=sorted(bowl_stats['team'].unique())).ask()
    team_bowlers = bowl_stats[bowl_stats['team'] == bowl_team]['bowler'].unique().tolist()
    bowler_name = questionary.select(f"Select Bowler from {bowl_team}:", choices=sorted(team_bowlers)).ask()
    
    # Get Bowler Stats
    bw_row = bowl_stats[bowl_stats['bowler'] == bowler_name].iloc[0]

    # 5. Build Feature Row (Matches FEATURE_COLS in training)
    # Order: runs_so_far, wickets_so_far, balls_so_far, current_run_rate, 
    #        balls_remaining, wickets_remaining, venue, strike_rate, average, economy
    input_row = pd.DataFrame([{
        "runs_so_far": status["runs_so_far"],
        "wickets_so_far": status["wickets_so_far"],
        "balls_so_far": status["balls_so_far"],
        "current_run_rate": status["current_run_rate"],
        "balls_remaining": status["balls_remaining"],
        "wickets_remaining": status["wickets_remaining"],
        "venue": le_v.transform([venue])[0],
        "strike_rate": b_row["strike_rate"],
        "average": b_row["average"],
        "economy": bw_row["economy"]
    }])

    # 6. Prediction
    pred = model.predict(input_row)[0]
    final_score = int(round(pred))

    # Output Display
    console.print("\n[bold magenta]Prediction Summary[/bold magenta]")
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("Category", style="dim")
    table.add_column("Selection")
    table.add_row("Venue", f"{venue} ({v_nature})")
    table.add_row("Striker", f"{batter_name} (SR: {b_row['strike_rate']:.1f})")
    table.add_row("Bowler", f"{bowler_name} (Eco: {bw_row['economy']:.1f})")
    table.add_row("Current State", f"{status['runs_so_far']}/{status['wickets_so_far']} in {over_count} ov")
    console.print(table)

    console.print(f"\n[bold green]>>> PREDICTED FINAL SCORE: {final_score} <<<[/bold green]\n")

if __name__ == "__main__":
    run_final_cli()