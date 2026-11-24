"""CSV, JSON, and plot output for the search-depth and board-size experiments."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.heuristics import HEURISTIC_ABBREVIATIONS
from experiments.board_size_experiment import BoardSizeExperimentResult
from experiments.depth_experiment import DepthExperimentResult
from experiments.results import save_results
from experiments.tournament import GameRecord, TournamentResult


def _write_matrix(path: Path, row_names: Sequence[str], column_names: Sequence[str],
                  values: Sequence[Sequence[float]], corner: str = "agent") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([corner, *column_names])
        writer.writerows([name, *row] for name, row in zip(row_names, values))


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _save_depth_plot(path: Path, depths: Sequence[int], heuristics: Sequence[str],
                     stats: Dict[str, dict], metric: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for heuristic in heuristics:
        values = [stats[f"{heuristic}-D{depth}"][metric] for depth in depths]
        ax.plot(depths, values, marker="o",
                label=HEURISTIC_ABBREVIATIONS.get(heuristic, heuristic))
    ax.set_title(title)
    ax.set_xlabel("Search depth (plies)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Heuristic")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def update_combined_summary(results_root: Path) -> Path:
    """Write an index of experiment completion without changing experiment outputs."""
    root = Path(results_root)
    definitions = (
        (1, root / "experiment_1" / "tournament_summary.json"),
        (2, root / "experiment_2" / "summary.json"),
        (3, root / "experiment_3" / "summary.json"),
    )
    experiments = {}
    for number, summary_path in definitions:
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            experiments[str(number)] = {
                "completed": bool(summary.get("completed", False)),
                "games_completed": summary.get("games_completed", 0),
                "summary_file": str(summary_path.relative_to(root)),
                "configuration": summary.get("configuration", {}),
            }
        else:
            experiments[str(number)] = {
                "completed": False,
                "games_completed": 0,
                "summary_file": None,
                "configuration": None,
            }
    path = root / "combined_summary.json"
    _write_json(path, {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiments": experiments,
    })
    return path


def save_depth_results(result: DepthExperimentResult, results_root: Path) -> Path:
    """Persist completed depth results without modifying other experiment folders."""
    config = result.config
    heuristics = list(config.heuristics)
    depths = list(config.depths)
    expected_games = len(heuristics) * len(depths) * config.games_per_side * 2
    if not result.completed or len(result.records) != expected_games:
        raise ValueError("Cannot save Experiment 2 results until every scheduled game completes.")
    out = Path(results_root) / "experiment_2"
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("win_rate_matrix.csv", "win_rate"),
        ("draw_rate_matrix.csv", "draw_rate"),
        ("decision_time_matrix.csv", "average_decision_time"),
        ("nodes_matrix.csv", "average_nodes_per_decision"),
    )
    for filename, key in metrics:
        values = [[result.stats[f"{heuristic}-D{depth}"][key] for depth in depths]
                  for heuristic in heuristics]
        _write_matrix(out / filename, heuristics, [str(depth) for depth in depths], values)

    _save_depth_plot(
        plots / "win_rate_vs_depth.png", depths, heuristics, result.stats,
        "win_rate", "Win Rate vs Search Depth", "Win rate (wins / games)",
    )
    _save_depth_plot(
        plots / "decision_time_vs_depth.png", depths, heuristics, result.stats,
        "average_decision_time", "Average Decision Time vs Search Depth",
        "Seconds per decision",
    )
    _save_depth_plot(
        plots / "nodes_vs_depth.png", depths, heuristics, result.stats,
        "average_nodes_per_decision", "Average Evaluated Nodes vs Search Depth",
        "Evaluated terminal states and heuristic leaves per decision",
    )

    with (out / "game_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "game_number", "heuristic", "evaluated_depth", "baseline_depth",
            "evaluated_side", "winner", "moves", "duration_seconds",
            "evaluated_avg_decision_time_seconds", "baseline_avg_decision_time_seconds",
            "evaluated_nodes", "baseline_nodes",
        ])
        for record in result.records:
            writer.writerow([
                record.game_number, record.heuristic, record.evaluated_depth,
                record.baseline_depth, record.evaluated_side, record.winner,
                record.moves, round(record.duration, 6),
                round(record.evaluated_avg_decision_time, 6),
                round(record.baseline_avg_decision_time, 6),
                record.evaluated_nodes, record.baseline_nodes,
            ])

    _write_json(out / "summary.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": 2,
        "completed": result.completed and len(result.records) == expected_games,
        "games_completed": len(result.records),
        "configuration": {
            "board_rows": config.rows,
            "board_cols": config.cols,
            "connect_n": config.connect,
            "tested_depths": depths,
            "baseline_depth": config.baseline_depth,
            "heuristics": heuristics,
            "games_per_configuration": config.games_per_side * 2,
            "expected_total_games": expected_games,
            "workers": config.workers,
        },
        "baseline_note": (
            f"Every tested result was measured against a fixed depth-{config.baseline_depth} "
            "baseline using the same heuristic."
        ),
        "node_count_definition": (
            "Terminal positions plus heuristic evaluations at the search horizon, "
            "summed across the evaluated agent's decisions."
        ),
        "total_duration_seconds": result.duration,
        "statistics": result.stats,
    })
    return out


def _cross_heuristic_stats(records: Sequence[GameRecord],
                           names: Sequence[str]) -> Dict[str, dict]:
    stats = {
        name: {
            "games": 0, "wins": 0, "losses": 0, "draws": 0, "decisions": 0,
            "total_decision_time": 0.0, "nodes": 0, "duration": 0.0, "moves": 0,
        }
        for name in [*names, "overall"]
    }
    for record in records:
        if record.first_heuristic == record.second_heuristic:
            continue
        first_decisions = (record.moves + 1) // 2
        second_decisions = record.moves // 2
        sides = (
            (record.first_heuristic, "first", first_decisions,
             record.first_avg_decision_time, record.first_nodes),
            (record.second_heuristic, "second", second_decisions,
             record.second_avg_decision_time, record.second_nodes),
        )
        for heuristic, side, decisions, average_time, nodes in sides:
            outcome = "draws" if record.winner == "draw" else (
                "wins" if record.winner == side else "losses"
            )
            for key in (heuristic, "overall"):
                entry = stats[key]
                entry["games"] += 1
                entry[outcome] += 1
                entry["decisions"] += decisions
                entry["total_decision_time"] += average_time * decisions
                entry["nodes"] += nodes
                entry["duration"] += record.duration
                entry["moves"] += record.moves
    for entry in stats.values():
        games, decisions = entry["games"], entry["decisions"]
        entry["win_rate"] = entry["wins"] / games if games else 0.0
        entry["win_draw_rate"] = (
            (entry["wins"] + 0.5 * entry["draws"]) / games if games else 0.0
        )
        entry["average_decision_time"] = (
            entry["total_decision_time"] / decisions if decisions else 0.0
        )
        entry["average_nodes_per_decision"] = (
            entry["nodes"] / decisions if decisions else 0.0
        )
        entry["average_game_duration"] = entry["duration"] / games if games else 0.0
        entry["average_moves_per_game"] = entry["moves"] / games if games else 0.0
    return stats


def _save_board_comparison_plots(out: Path, results: Sequence[TournamentResult],
                                 comparison: dict, names: Sequence[str]) -> None:
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    legacy_plot = plots / "draw_rate_heatmaps.png"
    if legacy_plot.exists():
        legacy_plot.unlink()
    legacy_plot = plots / "win_rate_heatmaps.png"
    if legacy_plot.exists():
        legacy_plot.unlink()
    labels = [HEURISTIC_ABBREVIATIONS.get(name, name) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, result, label in zip(axes, results, ("Standard 6 × 7", "Expanded 10 × 11")):
        image = ax.imshow(result.matrices.win_draw_rates, cmap="YlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(names)), labels=labels)
        ax.set_yticks(range(len(names)), labels=labels)
        ax.set_xlabel("Opponent")
        ax.set_ylabel("Agent")
        ax.set_title(f"{label} Win-Draw Rate")
        for row in range(len(names)):
            for col in range(len(names)):
                ax.text(col, row,
                        f"{result.matrices.win_draw_rates[row][col] * 100:.1f}%",
                        ha="center", va="center")
    fig.colorbar(image, ax=axes, label="Win-draw rate")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.88, wspace=0.35)
    fig.savefig(plots / "win_draw_rate_heatmaps.png", dpi=150)
    plt.close(fig)

    for metric, title, ylabel, filename in (
        ("average_decision_time", "Decision Time by Board Size", "Seconds per decision",
         "decision_time_by_board_size.png"),
        ("average_nodes_per_decision", "Evaluated Nodes by Board Size",
         "Nodes per decision", "nodes_by_board_size.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(names))
        width = 0.36
        for board_index, board in enumerate(("standard", "expanded")):
            values = [comparison[board][name][metric] for name in names]
            offset = -width / 2 if board_index == 0 else width / 2
            ax.bar([i + offset for i in x], values, width,
                   label="6 × 7" if board == "standard" else "10 × 11")
        ax.set_xticks(list(x), labels=labels)
        ax.set_xlabel("Heuristic (cross-heuristic games only)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Board")
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=150)
        plt.close(fig)


def save_board_size_results(result: BoardSizeExperimentResult,
                            results_root: Path) -> Path:
    """Write both board results and their self-play-excluding comparison."""
    expected_per_board = (
        len(result.results[0].config.heuristics)
        * (len(result.results[0].config.heuristics) + 1) // 2
        * result.config.games_per_side * 2
    )
    if len(result.results) != 2 or any(
        len(tournament.records) != expected_per_board for tournament in result.results
    ):
        raise ValueError("Cannot save Experiment 3 results until both board tournaments complete.")
    out = Path(results_root) / "experiment_3"
    out.mkdir(parents=True, exist_ok=True)
    board_dirs = ("standard", "expanded")
    names = list(result.results[0].config.heuristics)
    comparison = {}
    for board_name, tournament_result in zip(board_dirs, result.results):
        board_dir = out / board_name
        save_results(tournament_result, board_dir)
        summary_path = board_dir / "tournament_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["experiment"] = 3
        summary["completed"] = len(tournament_result.records) == expected_per_board
        summary["games_completed"] = len(tournament_result.records)
        summary["standard_result_source"] = (
            result.standard_source if board_name == "standard" else None
        )
        summary["self_play_excluded_from_comparison"] = True
        _write_json(summary_path, summary)
        comparison[board_name] = _cross_heuristic_stats(tournament_result.records, names)

    comparison_path = out / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "board", "rows", "columns", "heuristic", "games", "wins", "losses", "draws",
            "win_draw_rate", "average_decision_time_seconds", "average_nodes_per_decision",
            "average_game_duration_seconds", "average_moves_per_game",
        ])
        for board_name, rows, cols in (
            ("standard", *result.config.board_sizes[0]),
            ("expanded", *result.config.board_sizes[1]),
        ):
            for heuristic in [*names, "overall"]:
                stats = comparison[board_name][heuristic]
                writer.writerow([
                    board_name, rows, cols, heuristic, stats["games"], stats["wins"],
                    stats["losses"], stats["draws"], stats["win_draw_rate"],
                    stats["average_decision_time"], stats["average_nodes_per_decision"],
                    stats["average_game_duration"], stats["average_moves_per_game"],
                ])
    _save_board_comparison_plots(out, result.results, comparison, names)
    total_games = sum(len(tournament.records) for tournament in result.results)
    expected_total_games = expected_per_board * 2
    _write_json(out / "summary.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": 3,
        "completed": total_games == expected_total_games,
        "games_completed": total_games,
        "configuration": {
            "board_sizes_rows_by_columns": [list(size) for size in result.config.board_sizes],
            "search_depth": result.config.depth,
            "connect_n": result.config.connect,
            "games_per_unique_pairing": result.config.games_per_side * 2,
            "expected_total_games": expected_total_games,
            "heuristics": names,
        },
        "standard_result_reused_from": result.standard_source,
        "comparison_excludes_self_play": True,
        "cross_heuristic_statistics": comparison,
    })
    return out
