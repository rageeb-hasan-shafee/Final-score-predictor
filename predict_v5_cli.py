# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from pathlib import Path
# from sklearn.preprocessing import LabelEncoder
# from rich.console import Console
# from rich.panel import Panel

# console = Console()

# # Configuration
# PROJECT_ROOT = Path(__file__).resolve().parent
# DATASET_DIR = PROJECT_ROOT / "dataset"
# BASE_DIR = Path(__file__).resolve().parent / "Xgboost"
# MODEL_PATH = BASE_DIR / "cricket_team_model_v5.json"

# def get_team_averages(player_list, stats_df, stat_cols, player_col="batter"):
#     # Filter stats for only the players in the provided list
#     # Clean whitespace from player names in case of user typing errors
#     player_list = [p.strip() for p in player_list]
#     team_stats = stats_df[stats_df[player_col].isin(player_list)]
    
#     # Return averages, or default values if players aren't found
#     averages = team_stats[stat_cols].mean().to_dict()
#     return averages

# def predict_match_score():
#     # 1. Load Resources
#     if not MODEL_PATH.exists():
#         console.print(f"[bold red]Error: Model file not found at {MODEL_PATH}[/bold red]")
#         console.print("[yellow]Please run your training script (train_v5_team.py) first.[/yellow]")
#         return

#     model = xgb.XGBRegressor()
#     model.load_model(str(MODEL_PATH))
    
#     try:
#         bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
#         bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
#         df_raw = pd.read_csv(DATASET_DIR / "final_ball_by_ball_first_innings.csv")
#     except FileNotFoundError as e:
#         console.print(f"[bold red]Data Error: {e}[/bold red]")
#         return
    
#     # Train LabelEncoder for Venue mapping
#     le_v = LabelEncoder().fit(df_raw["venue"].astype(str))
    
#     console.print(Panel("[bold cyan]T20 Match Real-Time Predictor (Team-Aware)[/bold cyan]"))

#     # 2. Manual Inputs (Current Match State)
#     venue = input("Enter Venue: ")
#     venue_id = le_v.transform([venue])[0] if venue in le_v.classes_ else 0
    
#     curr_runs = int(input("Current Runs Scored: "))
#     curr_wickets = int(input("Current Wickets Lost: "))
#     curr_overs = float(input("Current Overs Completed (e.g. 10.2): "))
#     runs_3_overs = int(input("Runs scored in last 3 overs (Momentum): "))
    
#     # Convert overs to balls
#     whole_overs = int(curr_overs)
#     extra_balls = int(round((curr_overs - whole_overs) * 10))
#     total_balls_done = (whole_overs * 6) + extra_balls
    
#     # 3. Playing XI Inputs
#     print("\n--- Team Info ---")
#     batting_team_xi = input("Enter Batting Playing 11 (comma separated): ").split(",")
#     bowling_team_xi = input("Enter Bowling Playing 11 (comma separated): ").split(",")

#     # 4. Calculate "Future Potential" Features
#     # Use remaining batters for batting strength
#     remaining_batters = batting_team_xi[curr_wickets:] 
#     bat_skills = get_team_averages(remaining_batters, bat_stats, ['strike_rate', 'average'], "batter")
    
#     # Use full bowling unit for bowling strength
#     bowl_skills = get_team_averages(bowling_team_xi, bowl_stats, ['economy'], "bowler")

#     # 5. Build Feature Vector (COLUMN NAMES MUST MATCH TRAINING)
#     over_num = whole_overs
#     # 1: Powerplay, 2: Middle, 3: Death
#     phase = 1 if over_num < 6 else (2 if over_num < 15 else 3)
    
#     # Create input for XGBoost with the EXACT names used in train_v5
#     input_row = pd.DataFrame([{
#         "runs_so_far": curr_runs,
#         "wickets_so_far": curr_wickets,
#         "balls_so_far": total_balls_done,
#         "current_run_rate": curr_runs / (total_balls_done / 6) if total_balls_done > 0 else 0,
#         "balls_remaining": 120 - total_balls_done,
#         "wickets_remaining": 10 - curr_wickets,
#         "venue_avg_score": 165.0, 
#         "team_sr": bat_skills.get('strike_rate', 125.0), # Updated name
#         "team_avg": bat_skills.get('average', 25.0),    # Updated name
#         "team_eco": bowl_skills.get('economy', 8.5),    # Updated name
#         "match_phase": phase,
#         "runs_last_3_overs": runs_3_overs
#     }])

#     # Ensure the columns are in the exact same order as the training features
#     feature_names = model.get_booster().feature_names
#     input_row = input_row[feature_names]

#     # 6. Predict
#     prediction = model.predict(input_row)[0]
    
#     console.print(Panel(f"[bold green]Predicted Final Total: {int(round(prediction))}[/bold green]"))

# if __name__ == "__main__":
#     predict_match_score()

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import re
import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# --- CONFIGURATION ---
# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR.parent
# DATASET_DIR = PROJECT_ROOT / "dataset"
# MODEL_PATH = SCRIPT_DIR / "cricket_team_model_v5.json"
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BASE_DIR = Path(__file__).resolve().parent / "Xgboost"
MODEL_PATH = BASE_DIR / "cricket_team_model_v5.json"
INPUT_DIR = PROJECT_ROOT / "input"
MAX_XI_PLAYERS = 11
MIN_XI_PLAYERS = 5


def normalize_player_name(player_name):
    return " ".join(player_name.strip().split())


def slugify_for_filename(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "unknown"


def save_lineup_to_input_file(role, team_name, player_list):
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    team_slug = slugify_for_filename(team_name)
    file_path = INPUT_DIR / f"input_{role}_{team_slug}.txt"
    content = [
        f"role: {role}",
        f"team: {team_name}",
        "players:",
        *player_list,
        "",
    ]
    file_path.write_text("\n".join(content), encoding="utf-8")
    return file_path


def parse_lineup_file(file_path):
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    players = []
    in_players_section = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("players:"):
            in_players_section = True
            continue
        if in_players_section:
            players.extend([
                normalize_player_name(player)
                for player in stripped.replace("|", ",").split(",")
                if player.strip()
            ])

    if not players:
        for line in raw_lines:
            stripped = line.strip()
            if not stripped or ":" in stripped:
                continue
            players.extend([
                normalize_player_name(player)
                for player in stripped.replace("|", ",").split(",")
                if player.strip()
            ])

    players = [player for player in players if player]
    unique_players = []
    for player in players:
        if player not in unique_players:
            unique_players.append(player)
    return unique_players


def load_lineup_from_file(role):
    lineup_files = sorted([file_path for file_path in INPUT_DIR.glob(f"input_{role}_*.txt") if file_path.is_file()], key=lambda path: path.name.lower())
    if not lineup_files:
        # Backward compatibility for older generic input files.
        lineup_files = sorted(
            [file_path for file_path in INPUT_DIR.glob("input*.txt") if file_path.is_file()],
            key=lambda path: path.name.lower(),
        )

    if not lineup_files:
        console.print(f"[bold red]No saved lineup files found in {INPUT_DIR}[/bold red]")
        return None

    selected_file = questionary.autocomplete(
        f"Select {role} lineup file:",
        choices=[file_path.name for file_path in lineup_files],
    ).ask()

    if not selected_file:
        return None

    file_path = INPUT_DIR / selected_file
    players = parse_lineup_file(file_path)
    if not (MIN_XI_PLAYERS <= len(players) <= MAX_XI_PLAYERS):
        console.print(
            f"[bold red]{selected_file} contains {len(players)} players. Expected between {MIN_XI_PLAYERS} and {MAX_XI_PLAYERS}.[/bold red]"
        )
        return None

    console.print(f"[green]Loaded {len(players)} players from {selected_file}[/green]")
    return players


def select_players_interactively(stats_df, player_type="batter"):
    teams = sorted(stats_df['team'].dropna().unique().tolist())
    selected_team = questionary.autocomplete(
        f"Select the {player_type.capitalize()} Team:",
        choices=teams,
    ).ask()

    team_players = stats_df[stats_df['team'] == selected_team].sort_values(by=player_type)
    player_choices = [normalize_player_name(player) for player in team_players[player_type].tolist()]

    selected_players = []
    available_players = player_choices.copy()

    while len(selected_players) < MAX_XI_PLAYERS and available_players:
        next_player = questionary.autocomplete(
            f"Select {player_type} #{len(selected_players) + 1} for {selected_team}:",
            choices=available_players,
        ).ask()

        if not next_player:
            break

        if next_player in selected_players:
            console.print(f"[yellow]{next_player} is already selected.[/yellow]")
            continue

        selected_players.append(next_player)
        available_players.remove(next_player)

        if len(selected_players) >= MIN_XI_PLAYERS:
            if len(selected_players) == MAX_XI_PLAYERS:
                break

            continue_prompt = questionary.confirm(
                f"Selected {len(selected_players)} players. Add another {player_type}?",
                default=True,
            ).ask()
            if not continue_prompt:
                break

    if len(selected_players) < MIN_XI_PLAYERS:
        console.print(
            f"[bold red]Select at least {MIN_XI_PLAYERS} players for {selected_team}.[/bold red]"
        )
        return None

    if len(selected_players) > MAX_XI_PLAYERS:
        console.print(
            f"[bold red]You can select at most {MAX_XI_PLAYERS} players.[/bold red]"
        )
        return None

    return selected_team, selected_players


def get_player_selections(stats_df, player_type="batter"):
    lineup_method = questionary.select(
        f"Choose how to enter the {player_type} lineup:",
        choices=[
            Choice("Search and select players", value="search"),
            Choice("Load from TXT file", value="file"),
        ],
    ).ask()

    if lineup_method == "file":
        loaded_players = load_lineup_from_file(player_type)
        if loaded_players:
            return loaded_players
        console.print("[yellow]Falling back to interactive selection.[/yellow]")

    selected = select_players_interactively(stats_df, player_type)
    if not selected:
        return None

    selected_team, selected_players = selected
    saved_file = save_lineup_to_input_file(player_type, selected_team, selected_players)
    console.print(f"[green]Saved {player_type} lineup to {saved_file.name}[/green]")
    return selected_players

def predict_match_score():
    console.print(Panel("[bold yellow]T20 Match Real-Time Predictor v5[/bold yellow]\n[cyan]Interactive Team & Venue Selection[/cyan]"))

    # 1. Load Resources
    if not MODEL_PATH.exists():
        console.print(f"[bold red]Error: Model not found at {MODEL_PATH}[/bold red]")
        return

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    
    bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
    bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
    venue_df = pd.read_csv(DATASET_DIR / "venue_nature.csv")

    # 2. Interactive Venue Selection
    venue_choices = [f"{row['venue']} ({row['nature']})" for _, row in venue_df.iterrows()]
    selected_venue_str = questionary.autocomplete(
        "Enter/Select Venue:",
        choices=venue_choices
    ).ask()
    
    # Extract raw venue name and avg score
    venue_name = selected_venue_str.split(" (")[0]
    venue_avg = venue_df[venue_df['venue'] == venue_name]['avg_score'].values[0]

    # 3. Interactive Player Selection
    batting_players = get_player_selections(bat_stats, "batter")
    bowling_players = get_player_selections(bowl_stats, "bowler")

    if not batting_players or not bowling_players:
        console.print("[bold red]Lineup selection was not completed.[/bold red]")
        return

    # Calculate Averages for selected players
    selected_bat_stats = bat_stats[bat_stats['batter'].isin(batting_players)]
    avg_sr = selected_bat_stats['strike_rate'].mean()
    avg_val = selected_bat_stats['average'].mean()

    selected_bowl_stats = bowl_stats[bowl_stats['bowler'].isin(bowling_players)]
    avg_eco = selected_bowl_stats['economy'].mean()

    # 4. Match State Inputs
    curr_runs = int(console.input("[bold green]Current Runs Scored: [/bold green]"))
    curr_wickets = int(console.input("[bold green]Current Wickets Lost: [/bold green]"))
    overs_done = float(console.input("[bold green]Current Overs Completed (e.g. 10.2): [/bold green]"))
    momentum = float(console.input("[bold green]Runs in last 3 overs (Momentum): [/bold green]"))

    # 5. Prepare Features
    total_balls = int(overs_done) * 6 + int((overs_done % 1) * 10)
    phase = 1 if overs_done <= 6 else (2 if overs_done <= 15 else 3)

    X_input = pd.DataFrame([{
        'runs_so_far': curr_runs,
        'wickets_so_far': curr_wickets,
        'balls_so_far': total_balls,
        'current_run_rate': curr_runs / (total_balls / 6) if total_balls > 0 else 0,
        'balls_remaining': 120 - total_balls,
        'wickets_remaining': 10 - curr_wickets,
        'venue_avg_score': venue_avg,
        'team_sr': avg_sr,
        'team_avg': avg_val,
        'team_eco': avg_eco,
        'match_phase': phase,
        'runs_last_3_overs': momentum
    }])

    # Align with model columns
    feature_order = model.get_booster().feature_names
    X_input = X_input[feature_order]

    # 6. Predict & Display
    pred = round(model.predict(X_input)[0])

    res_table = Table(title="Prediction Summary")
    res_table.add_column("Factor", style="cyan")
    res_table.add_column("Value", style="white")
    res_table.add_row("Venue", venue_name)
    res_table.add_row("Team SR", f"{avg_sr:.2f}")
    res_table.add_row("Opp. Economy", f"{avg_eco:.2f}")
    res_table.add_row("Current State", f"{curr_runs}/{curr_wickets} ({overs_done} ov)")
    
    console.print(res_table)
    console.print(Panel(f"[bold green]PREDICTED FINAL TOTAL: {pred}[/bold green]", expand=False))


def run_predictor_loop():
    while True:
        predict_match_score()
        next_action = questionary.select(
            "Choose next action:",
            choices=[
                Choice("Run another prediction", value="again"),
                Choice("Exit", value="exit"),
            ],
        ).ask()

        if next_action != "again":
            console.print("[cyan]Exited predictor.[/cyan]")
            break

if __name__ == "__main__":
    run_predictor_loop()
