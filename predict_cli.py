#!/usr/bin/env python3

from pathlib import Path
import importlib
import numpy as np

try:
    import questionary
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("Missing dependencies. Install with:")
    print("pip install questionary rich numpy")
    raise

console = Console()


def build_snapshot_from_txt(input_path: Path, max_overs: int = 20, num_features: int = 6) -> tuple[np.ndarray, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip() != ""]

    if len(lines) == 0:
        raise ValueError("Input file is empty.")

    if len(lines) % 2 != 0:
        raise ValueError(
            "Input must contain an even number of non-empty lines: "
            "runs line, wickets line, repeated for each over."
        )

    current_over = len(lines) // 2
    if current_over > max_overs:
        raise ValueError(f"Input contains {current_over} overs, but max_overs is {max_overs}.")

    x = np.zeros((max_overs, num_features), dtype=np.float32)
    cumulative_runs = 0
    cumulative_wickets = 0

    for over_idx in range(current_over):
        runs_line = lines[2 * over_idx]
        wickets_line = lines[2 * over_idx + 1]

        try:
            runs = int(runs_line)
            wickets = int(wickets_line)
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer at over {over_idx + 1}: runs='{runs_line}', wickets='{wickets_line}'"
            ) from exc

        if not (0 <= runs <= 60):
            raise ValueError(f"Runs out of range at over {over_idx + 1}: {runs} (allowed 0-60)")
        if not (0 <= wickets <= 10):
            raise ValueError(f"Wickets out of range at over {over_idx + 1}: {wickets} (allowed 0-10)")

        over_no = over_idx + 1

        cumulative_runs += runs
        cumulative_wickets += wickets
        run_rate = cumulative_runs / over_no
        balls_remaining = (max_overs - over_no) * 6

        x[over_no - 1] = np.array(
            [
                runs,                 # runs_this_over
                wickets,              # wickets_this_over
                cumulative_runs,      # cumulative_runs
                cumulative_wickets,   # cumulative_wickets
                run_rate,             # run_rate
                balls_remaining,      # balls_remaining
            ],
            dtype=np.float32,
        )

    return x, current_over


def show_over_table(x: np.ndarray, current_over: int) -> None:
    table = Table(title="Innings Snapshot")
    table.add_column("Over", justify="right")
    table.add_column("Runs")
    table.add_column("Wkts")
    table.add_column("Cum Runs")
    table.add_column("Cum Wkts")
    table.add_column("RR")
    table.add_column("Balls Left")

    for i in range(current_over):
        r = x[i]
        table.add_row(
            str(i + 1),
            str(int(r[0])),
            str(int(r[1])),
            str(int(r[2])),
            str(int(r[3])),
            f"{r[4]:.2f}",
            str(int(r[5])),
        )
    console.print(table)


def heuristic_prediction(x: np.ndarray, current_over: int, max_overs: int = 20) -> float:
    last = x[current_over - 1]
    cumulative_runs = float(last[2])
    run_rate = float(last[4])
    overs_left = max_overs - current_over
    projected = cumulative_runs + run_rate * overs_left
    return projected


def predict_with_keras(model_path: Path, x: np.ndarray) -> float | None:
    try:
        keras_models = importlib.import_module("tensorflow.keras.models")
        load_model = getattr(keras_models, "load_model")
    except Exception:
        console.print("[yellow]TensorFlow not available. Using heuristic estimate.[/yellow]")
        return None

    if not model_path.exists():
        console.print(f"[yellow]Model not found at {model_path}. Using heuristic estimate.[/yellow]")
        return None

    model = load_model(model_path)
    pred = model.predict(np.expand_dims(x, axis=0), verbose=0)
    return float(pred.squeeze())


def main() -> None:
    console.print(Panel.fit("[bold cyan]Final Score Predictor (Questionary + Rich)[/bold cyan]"))

    max_overs = 20
    num_features = 6

    example_path = Path.cwd() / "innings_input.txt"
    console.print(
        "[cyan]Input format (txt): line1=runs over1, line2=wickets over1, line3=runs over2, line4=wickets over2...[/cyan]"
    )

    input_file = questionary.text(
        "Enter txt input path",
        default=str(example_path),
    ).ask()

    if not input_file:
        console.print("[red]No input file provided.[/red]")
        return

    try:
        x, current_over = build_snapshot_from_txt(Path(input_file), max_overs=max_overs, num_features=num_features)
    except Exception as exc:
        console.print(f"[red]Failed to parse input file:[/red] {exc}")
        return

    show_over_table(x, current_over)

    use_model = questionary.confirm("Use a trained Keras model (.h5/.keras) if available?", default=True).ask()

    prediction = None
    if use_model:
        model_input = questionary.text(
            "Enter model path",
            default=str(Path.cwd() / "ball_outcome_model.keras"),
        ).ask()
        prediction = predict_with_keras(Path(model_input), x)

    if prediction is None:
        prediction = heuristic_prediction(x, current_over, max_overs=max_overs)

    prediction = int(round(prediction))

    console.print(
        Panel.fit(
            f"[bold green]Predicted Final Score: {prediction}[/bold green]",
            title="Output",
        )
    )


if __name__ == "__main__":
    main()