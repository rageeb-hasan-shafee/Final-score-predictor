
# #!/usr/bin/env python3
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from pathlib import Path

# try:
#     import questionary
#     from rich.console import Console
#     from rich.table import Table
#     from rich.panel import Panel
# except ImportError:
#     print("Missing dependencies. Install with: pip install questionary rich numpy pandas xgboost")
#     raise

# console = Console()

# # The MAE you achieved during training
# MODEL_MAE = 15.23 

# def build_xgb_features_from_txt(input_path: Path):
#     if not input_path.exists():
#         raise FileNotFoundError(f"Input file not found: {input_path}")

#     raw_lines = input_path.read_text(encoding="utf-8").splitlines()
#     lines = [int(line.strip()) for line in raw_lines if line.strip() != ""]

#     if len(lines) % 2 != 0:
#         raise ValueError("Input must have pairs of (runs, wickets).")

#     runs_per_over = lines[0::2]
#     wickets_per_over = lines[1::2]
    
#     current_over = len(runs_per_over)
#     total_runs = sum(runs_per_over)
#     total_wickets = sum(wickets_per_over)
#     total_balls = current_over * 6
    
#     # Feature dictionary aligned with your XGBoost training
#     features = {
#         "runs_so_far": total_runs,
#         "wickets_so_far": total_wickets,
#         "balls_so_far": total_balls,
#         "overs_so_far": current_over,
#         "current_run_rate": total_runs / current_over if current_over > 0 else 0,
#         "balls_remaining": 120 - total_balls,
#         "wickets_remaining": 10 - total_wickets,
#         "venue": 0, "batter": 0, "bowler": 0, "non_striker": 0 
#     }
    
#     return pd.DataFrame([features]), current_over, total_runs, total_wickets

# def run_cli():
#     console.print(Panel("[bold cyan]XGBoost Cricket Score Predictor[/bold cyan]", expand=False))

#     input_path = Path("innings_input.txt")
#     model_path = Path("cricket_score_model.json")

#     try:
#         X_input, over, runs, wickets = build_xgb_features_from_txt(input_path)
        
#         # Status Table
#         table = Table(title=f"Match Summary: {over} Overs Completed")
#         table.add_column("Metric", style="cyan")
#         table.add_column("Value", style="bold yellow")
#         table.add_row("Current Score", f"{runs}/{wickets}")
#         table.add_row("Run Rate", f"{runs/over:.2f}")
#         table.add_row("Balls Left", str(120 - (over*6)))
#         console.print(table)

#         if model_path.exists():
#             model = xgb.XGBRegressor()
#             model.load_model(str(model_path))
#             prediction = round(model.predict(X_input)[0])
            
#             # Display Prediction + MAE
#             console.print("\n[bold green]📊 PREDICTION RESULTS:[/bold green]")
#             console.print(f"[bold white]Predicted Final Score:[/bold white] [bold yellow]{prediction}[/bold yellow]")
#             console.print(f"[bold white]Expected Range:[/bold white] [cyan]{prediction - round(MODEL_MAE)} - {prediction + round(MODEL_MAE)}[/cyan]")
#             console.print(f"[italic white](Based on Model MAE: {MODEL_MAE})[/italic white]\n")
#         else:
#             console.print("[red]Error: Model file 'cricket_score_model.json' not found.[/red]")

#     except Exception as e:
#         console.print(f"[red]Error:[/red] {e}")

# if __name__ == "__main__":
#     run_cli()

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
    print("Missing dependencies. Install with: pip install questionary rich pandas xgboost scikit-learn")
    raise

console = Console()
MODEL_MAE = 15.23  # Based on your high-performance training result

def load_data_and_encoders():
    """Recreates the LabelEncoders to match the training logic exactly."""
    df = pd.read_csv("final_ball_by_ball_first_innings.csv")
    encoders = {}
    for col in ["venue", "batter", "bowler", "non_striker"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders

def build_snapshot_from_txt(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    lines = [int(line.strip()) for line in raw_lines if line.strip() != ""]

    runs_per_over = lines[0::2]
    wickets_per_over = lines[1::2]
    
    current_over = len(runs_per_over)
    total_runs = sum(runs_per_over)
    total_wickets = sum(wickets_per_over)
    total_balls = current_over * 6
    
    return {
        "runs_so_far": total_runs,
        "wickets_so_far": total_wickets,
        "balls_so_far": total_balls,
        "overs_so_far": current_over,
        "current_run_rate": total_runs / current_over if current_over > 0 else 0,
        "balls_remaining": 120 - total_balls,
        "wickets_remaining": 10 - total_wickets
    }, current_over

def run_cli():
    console.print(Panel("[bold cyan]XGBoost Venue-Specific Predictor[/bold cyan]", expand=False))

    # 1. Load Data
    try:
        encoders = load_data_and_encoders()
        venue_info = pd.read_csv("venue_nature.csv")
    except Exception as e:
        console.print(f"[red]Error loading data files:[/red] {e}")
        return

    # 2. Parse Match Status from TXT
    try:
        match_stats, over = build_snapshot_from_txt(Path("innings_input.txt"))
    except Exception as e:
        console.print(f"[red]Error parsing innings_input.txt:[/red] {e}")
        return

    # 3. Ask for Venue
    venue_list = sorted(encoders["venue"].classes_)
    selected_venue = questionary.select(
        "Select the Match Venue:",
        choices=venue_list,
        use_shortcuts=False
    ).ask()

    if not selected_venue:
        return

    # Show Venue Nature
    nature = venue_info[venue_info['venue'] == selected_venue]['nature'].values[0]
    avg_score = venue_info[venue_info['venue'] == selected_venue]['avg_score'].values[0]
    console.print(f"\n[bold magenta]Venue Profile:[/bold magenta] {nature} (Avg First Innings: {avg_score:.1f})")

    # 4. Prepare Feature Row
    # We use the selected venue, and the first available player as placeholder for others
    feature_dict = match_stats.copy()
    feature_dict["venue"] = encoders["venue"].transform([selected_venue])[0]
    feature_dict["batter"] = 0
    feature_dict["bowler"] = 0
    feature_dict["non_striker"] = 0

    # Ensure correct column order
    cols = ["runs_so_far", "wickets_so_far", "balls_so_far", "overs_so_far", 
            "current_run_rate", "balls_remaining", "wickets_remaining", 
            "venue", "batter", "bowler", "non_striker"]
    X_input = pd.DataFrame([feature_dict])[cols]

    # 5. Predict
    model_path = "cricket_score_model.json"
    if Path(model_path).exists():
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        pred = round(model.predict(X_input)[0])

        # Status Table
        table = Table(title="Match Status & Prediction")
        table.add_column("Item", style="cyan")
        table.add_column("Details", style="yellow")
        table.add_row("Current Over", str(over))
        table.add_row("Current Score", f"{match_stats['runs_so_far']}/{match_stats['wickets_so_far']}")
        table.add_row("Venue", selected_venue)
        console.print(table)

        console.print(f"\n[bold green]Predicted Final Score: {pred}[/bold green]")
        console.print(f"[cyan]Confidence Interval: {pred - round(MODEL_MAE)} to {pred + round(MODEL_MAE)}[/cyan]\n")
    else:
        console.print("[red]Model file not found. Please train the model first.[/red]")

if __name__ == "__main__":
    run_cli()