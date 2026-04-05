from pathlib import Path
import re

import pandas as pd
import streamlit as st
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BASE_DIR = PROJECT_ROOT / "Xgboost"
MODEL_PATH = BASE_DIR / "cricket_team_model_v5.json"
INPUT_DIR = PROJECT_ROOT / "input"
MAX_XI_PLAYERS = 11
MIN_XI_PLAYERS = 5


def normalize_player_name(player_name):
    return " ".join(str(player_name).strip().split())


def slugify_for_filename(value):
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "unknown"


def save_lineup_to_input_file(role, team_name, player_list):
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    team_slug = slugify_for_filename(team_name)
    file_path = INPUT_DIR / f"input_{role}_{team_slug}.txt"
    content = [f"role: {role}", f"team: {team_name}", "players:", *player_list, ""]
    file_path.write_text("\n".join(content), encoding="utf-8")
    return file_path


def parse_lineup_content(raw_text):
    raw_lines = raw_text.splitlines()
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
            players.extend(
                [
                    normalize_player_name(player)
                    for player in stripped.replace("|", ",").split(",")
                    if player.strip()
                ]
            )

    if not players:
        for line in raw_lines:
            stripped = line.strip()
            if not stripped or ":" in stripped:
                continue
            players.extend(
                [
                    normalize_player_name(player)
                    for player in stripped.replace("|", ",").split(",")
                    if player.strip()
                ]
            )

    unique_players = []
    for player in players:
        if player and player not in unique_players:
            unique_players.append(player)
    return unique_players


def parse_lineup_file(file_path):
    raw_text = file_path.read_text(encoding="utf-8")
    return parse_lineup_content(raw_text)


@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_dataframes():
    bat_stats = pd.read_csv(DATASET_DIR / "batter_stats_v2.csv")
    bowl_stats = pd.read_csv(DATASET_DIR / "bowler_stats_v2.csv")
    venue_df = pd.read_csv(DATASET_DIR / "venue_nature.csv")
    return bat_stats, bowl_stats, venue_df


def get_lineup_files(role):
    lineup_files = sorted(
        [file_path for file_path in INPUT_DIR.glob(f"input_{role}_*.txt") if file_path.is_file()],
        key=lambda path: path.name.lower(),
    )
    if not lineup_files:
        lineup_files = sorted(
            [file_path for file_path in INPUT_DIR.glob("input*.txt") if file_path.is_file()],
            key=lambda path: path.name.lower(),
        )
    return lineup_files


def get_lineup_via_file(role):
    files = get_lineup_files(role)
    if files:
        st.caption(
            f"Saved files available in {INPUT_DIR}: "
            + ", ".join(file_path.name for file_path in files[:5])
            + (" ..." if len(files) > 5 else "")
        )

    uploaded_file = st.file_uploader(
        f"Choose {role} lineup TXT file",
        type=["txt"],
        key=f"{role}_file_uploader",
        help="Click Browse files to open a file chooser dialog.",
    )

    if uploaded_file is None:
        st.info("Upload a TXT file to continue.")
        return None, None

    if role not in uploaded_file.name.lower():
        st.warning(
            f"The filename does not include '{role}'. Ensure this is the correct lineup file."
        )

    try:
        raw_text = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("File must be UTF-8 encoded text.")
        return None, None

    players = parse_lineup_content(raw_text)

    if not (MIN_XI_PLAYERS <= len(players) <= MAX_XI_PLAYERS):
        st.error(
            f"{uploaded_file.name} contains {len(players)} players. "
            f"Expected between {MIN_XI_PLAYERS} and {MAX_XI_PLAYERS}."
        )
        return None, None

    st.success(f"Loaded {len(players)} players from {uploaded_file.name}")
    return uploaded_file.name, players


def get_lineup_via_search(stats_df, player_type):
    teams = sorted(stats_df["team"].dropna().unique().tolist())
    team = st.selectbox(
        f"Select {player_type.capitalize()} Team",
        options=teams,
        key=f"{player_type}_team_select",
    )

    team_players = stats_df[stats_df["team"] == team].sort_values(by=player_type)
    player_options = [normalize_player_name(player) for player in team_players[player_type].tolist()]

    selected_players = st.multiselect(
        f"Select {player_type.capitalize()} Players ({MIN_XI_PLAYERS}-{MAX_XI_PLAYERS})",
        options=player_options,
        default=player_options[:MIN_XI_PLAYERS],
        key=f"{player_type}_players_multiselect",
    )

    if len(selected_players) < MIN_XI_PLAYERS:
        st.error(f"Select at least {MIN_XI_PLAYERS} players for {team}.")
        return None, None

    if len(selected_players) > MAX_XI_PLAYERS:
        st.error(f"You can select at most {MAX_XI_PLAYERS} players.")
        return None, None

    saved_file = save_lineup_to_input_file(role=player_type, team_name=team, player_list=selected_players)
    st.caption(f"Saved lineup to {saved_file.name}")
    return team, selected_players


def get_player_selections(stats_df, player_type):
    lineup_method = st.radio(
        f"Choose how to enter the {player_type} lineup",
        options=["Search and select players", "Load from TXT file"],
        horizontal=True,
        key=f"{player_type}_method_radio",
    )

    if lineup_method == "Load from TXT file":
        _, players = get_lineup_via_file(player_type)
        return players

    _, players = get_lineup_via_search(stats_df, player_type)
    return players


def compute_total_balls(overs_done):
    whole_overs = int(overs_done)
    extra_balls = int((overs_done % 1) * 10)
    return whole_overs * 6 + extra_balls


def main():
    st.set_page_config(page_title="T20 Match Real-Time Predictor v5", layout="wide")
    st.title("T20 Match Real-Time Predictor v5")
    st.caption("Interactive Team, Venue, and Match-State Selection")

    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()

    try:
        model = load_model()
        bat_stats, bowl_stats, venue_df = load_dataframes()
    except Exception as exc:
        st.error(f"Failed to load model/data: {exc}")
        st.stop()

    st.subheader("1) Venue Selection")
    venue_choices = [f"{row['venue']} ({row['nature']})" for _, row in venue_df.iterrows()]
    selected_venue_str = st.selectbox("Enter/Select Venue", options=venue_choices)
    venue_name = selected_venue_str.split(" (")[0]
    venue_avg = float(venue_df[venue_df["venue"] == venue_name]["avg_score"].values[0])

    st.subheader("2) Playing XI Selection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Batting Lineup")
        batting_players = get_player_selections(bat_stats, "batter")
    with col2:
        st.markdown("### Bowling Lineup")
        bowling_players = get_player_selections(bowl_stats, "bowler")

    st.subheader("3) Match State Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        curr_runs = st.number_input("Current Runs Scored", min_value=0, max_value=400, value=80, step=1)
    with c2:
        curr_wickets = st.number_input("Current Wickets Lost", min_value=0, max_value=10, value=2, step=1)
    with c3:
        overs_done = st.number_input("Current Overs Completed (e.g. 10.2)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    with c4:
        momentum = st.number_input("Runs in last 3 overs (Momentum)", min_value=0.0, max_value=100.0, value=24.0, step=1.0)

    predict_clicked = st.button("Predict Final Total", type="primary", use_container_width=True)

    if not predict_clicked:
        st.stop()

    if not batting_players or not bowling_players:
        st.error("Lineup selection was not completed.")
        st.stop()

    selected_bat_stats = bat_stats[bat_stats["batter"].isin(batting_players)]
    avg_sr = float(selected_bat_stats["strike_rate"].mean())
    avg_val = float(selected_bat_stats["average"].mean())

    selected_bowl_stats = bowl_stats[bowl_stats["bowler"].isin(bowling_players)]
    avg_eco = float(selected_bowl_stats["economy"].mean())

    total_balls = compute_total_balls(float(overs_done))
    phase = 1 if overs_done <= 6 else (2 if overs_done <= 15 else 3)

    X_input = pd.DataFrame(
        [
            {
                "runs_so_far": float(curr_runs),
                "wickets_so_far": float(curr_wickets),
                "balls_so_far": float(total_balls),
                "current_run_rate": float(curr_runs) / (total_balls / 6) if total_balls > 0 else 0.0,
                "balls_remaining": 120 - float(total_balls),
                "wickets_remaining": 10 - float(curr_wickets),
                "venue_avg_score": venue_avg,
                "team_sr": avg_sr,
                "team_avg": avg_val,
                "team_eco": avg_eco,
                "match_phase": float(phase),
                "runs_last_3_overs": float(momentum),
            }
        ]
    )

    feature_order = model.get_booster().feature_names
    X_input = X_input[feature_order]

    pred = int(round(model.predict(X_input)[0]))

    st.subheader("Prediction Summary")
    summary_df = pd.DataFrame(
        {
            "Factor": ["Venue", "Team SR", "Opp. Economy", "Current State"],
            "Value": [
                venue_name,
                f"{avg_sr:.2f}",
                f"{avg_eco:.2f}",
                f"{int(curr_runs)}/{int(curr_wickets)} ({float(overs_done):.1f} ov)",
            ],
        }
    )
    st.table(summary_df)

    st.success(f"PREDICTED FINAL TOTAL: {pred}")


if __name__ == "__main__":
    main()
