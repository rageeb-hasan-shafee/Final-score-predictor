import pandas as pd
import xgboost as xgb
from pathlib import Path

# --- DEPENDENCY CHECK ---
try:
    import questionary
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("Missing dependencies. Install with: pip install questionary rich pandas xgboost")
    exit()

console = Console()

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
INPUT_PATH = PROJECT_ROOT / "input" / "innings_input.txt"
VENUE_STATS_PATH = PROJECT_ROOT / "dataset" / "venue_nature.csv"
MODEL_PATH = PROJECT_ROOT / "Xgboost" / "cricket_team_model_v4.json"

def analyze_over_by_over_file(file_path):
    """Processes file where: Line 1=Runs, Line 2=Wickets for each over."""
    if not file_path.exists():
        return None
    
    try:
        raw_lines = file_path.read_text(encoding="utf-8").splitlines()
        data = [int(line.strip()) for line in raw_lines if line.strip()]
        
        overs_data = []
        # Pair the lines: (Runs, Wickets)
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                overs_data.append({'runs': data[i], 'wickets': data[i+1]})
        
        total_overs = len(overs_data)
        curr_runs = sum(o['runs'] for o in overs_data)
        curr_wicks = sum(o['wickets'] for o in overs_data)
        
        # Momentum: Last 3 overs (18 balls)
        last_3 = overs_data[-3:]
        runs_last_3_overs = sum(o['runs'] for o in last_3)
        
        return {
            'runs': curr_runs,
            'wickets': curr_wicks,
            'overs': total_overs,
            'balls': total_overs * 6,
            'momentum': float(runs_last_3_overs)
        }
    except Exception as e:
        console.print(f"[bold red]Error parsing {file_path.name}: {e}[/bold red]")
        return None

def run_v4_predict():
    console.print(Panel("[bold yellow]XGBoost Cricket Predictor v4[/bold yellow]\n[cyan]Venue & Nature Integrated Mode[/cyan]"))

    # 1. Load Model
    if not MODEL_PATH.exists():
        console.print(f"[bold red]Error: Model not found at {MODEL_PATH}[/bold red]")
        return
    
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    
    # 2. Load Venue Stats
    if not VENUE_STATS_PATH.exists():
        console.print(f"[bold red]Error: {VENUE_STATS_PATH.name} not found in dataset folder![/bold red]")
        return
    
    venue_df = pd.read_csv(VENUE_STATS_PATH)
    # Create display string: "Venue Name (Nature)"
    venue_choices = []
    for _, row in venue_df.iterrows():
        display_name = f"{row['venue']} ({row['nature']})"
        venue_choices.append({
            "name": display_name,
            "value": {"avg": row['avg_score'], "name": row['venue'], "nature": row['nature']}
        })

    # 3. Analyze input file
    stats = analyze_over_by_over_file(INPUT_PATH)
    if not stats:
        console.print(f"[bold red]Error: Could not read or parse {INPUT_PATH.name}[/bold red]")
        return

    # 4. Interactive Venue Selection
    selected_venue_data = questionary.select(
        "Select the Match Venue:",
        choices=venue_choices
    ).ask()

    if not selected_venue_data:
        return

    # 5. Prepare Features (11 features for v4 model)
    # Phase logic: 1: Powerplay(1-6), 2: Middle(7-15), 3: Death(16-20)
    phase = 1 if stats['overs'] <= 6 else (2 if stats['overs'] <= 15 else 3)
    
    X_input = pd.DataFrame([{
        'runs_so_far': stats['runs'],
        'wickets_so_far': stats['wickets'],
        'balls_so_far': stats['balls'],
        'current_run_rate': stats['runs'] / stats['overs'] if stats['overs'] > 0 else 0,
        'balls_remaining': 120 - stats['balls'],
        'wickets_remaining': 10 - stats['wickets'],
        'venue_avg_score': float(selected_venue_data['avg']),
        'strike_rate': 128.5, # Median fallback for team strike rate
        'economy': 8.1,       # Median fallback for team economy
        'match_phase': int(phase),
        'runs_last_3_overs': stats['momentum']
    }])

    # Align columns to match model JSON feature order exactly
    feature_order = model.get_booster().feature_names
    X_input = X_input[feature_order]

    # 6. Predict
    prediction = round(model.predict(X_input)[0])

    # 7. Results Display
    table = Table(title="Match Status Snapshot")
    table.add_column("Category", style="cyan")
    table.add_column("Details", style="white")
    table.add_row("Venue", selected_venue_data['name'])
    table.add_row("Wicket Nature", selected_venue_data['nature'])
    table.add_row("Score", f"{stats['runs']}/{stats['wickets']}")
    table.add_row("Overs Bowled", str(stats['overs']))
    table.add_row("Momentum (Last 3 Overs)", f"{int(stats['momentum'])} runs")
    
    console.print(table)
    console.print(Panel(f"[bold green]PREDICTED FINAL SCORE: {prediction}[/bold green]", expand=False))

if __name__ == "__main__":
    run_v4_predict()