"""Persist tournament outputs and render them for the terminal."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from agents.heuristics import HEURISTIC_ABBREVIATIONS
from experiments.tournament import GameRecord, Matrices, TournamentResult

GAME_COLUMNS = [
    "game_number", "first_heuristic", "second_heuristic", "first_instance", "winner",
    "winner_heuristic", "moves", "duration_seconds", "first_avg_decision_time_seconds",
    "second_avg_decision_time_seconds", "first_nodes_evaluated", "second_nodes_evaluated",
]

OUTPUT_FILES = (
    "win_rate_matrix.csv", "win_count_matrix.csv", "draw_count_matrix.csv",
    "game_results.csv", "tournament_summary.json",
)


def _write_matrix(path: Path, names: Sequence[str], matrix, corner: str = "agent") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([corner, *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row])


def _game_row(rec: GameRecord) -> list:
    return [
        rec.game_number, rec.first_heuristic, rec.second_heuristic, rec.first_instance,
        rec.winner, rec.winner_heuristic, rec.moves, round(rec.duration, 6),
        round(rec.first_avg_decision_time, 6), round(rec.second_avg_decision_time, 6),
        rec.first_nodes, rec.second_nodes,
    ]


def save_results(result: TournamentResult, results_dir: Path) -> List[Path]:
    """Write all outputs into ``results_dir``, replacing any previous tournament's files."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    m = result.matrices
    cfg = result.config

    rates = [[round(v, 6) for v in row] for row in m.win_rates]
    _write_matrix(results_dir / "win_rate_matrix.csv", m.names, rates)
    _write_matrix(results_dir / "win_count_matrix.csv", m.names, m.wins)
    _write_matrix(results_dir / "draw_count_matrix.csv", m.names, m.draws)

    with (results_dir / "game_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(GAME_COLUMNS)
        writer.writerows(_game_row(rec) for rec in result.records)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuration": {
            "board_rows": cfg.rows,
            "board_cols": cfg.cols,
            "connect_n": cfg.connect,
            "search_depth": cfg.depth,
            "heuristics": list(cfg.heuristics),
            "heuristic_abbreviations": {n: HEURISTIC_ABBREVIATIONS.get(n, n) for n in cfg.heuristics},
            "games_per_side": cfg.games_per_side,
            "total_games": len(result.records),
        },
        "conventions": {
            "win_rate": "wins / games played against that opponent",
            "diagonal": "self-play: each game counted once per instance, so games = 2 x self-play games",
        },
        "total_duration_seconds": round(result.duration, 3),
        "heuristic_statistics": result.stats,
        "matrices": {
            "heuristics": m.names,
            "win_rate": rates,
            "win_count": m.wins,
            "draw_count": m.draws,
            "games_played": m.games,
        },
    }
    with (results_dir / "tournament_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return [results_dir / name for name in OUTPUT_FILES]


def format_matrix(matrices: Matrices) -> str:
    names = matrices.names
    labels = [HEURISTIC_ABBREVIATIONS.get(n, n) for n in names]
    header = "Agent".ljust(8) + "".join(label.rjust(9) for label in labels)
    lines = [header]
    for label, row in zip(labels, matrices.win_rates):
        lines.append(label.ljust(8) + "".join(f"{v * 100:8.1f}%" for v in row))
    return "\n".join(lines)


def format_summary(result: TournamentResult) -> str:
    cfg = result.config
    lines = [
        f"Tournament: {len(result.records)} games, depth {cfg.depth}, "
        f"board {cfg.rows}x{cfg.cols}, {result.duration:.1f}s total",
        "",
        "Win rate matrix (row agent vs column opponent):",
        format_matrix(result.matrices),
        "",
        f"{'Heuristic':<26}{'W':>4}{'L':>4}{'D':>4}{'Win%':>8}{'AvgMove(s)':>12}{'Nodes/move':>12}",
    ]
    for name in cfg.heuristics:
        s = result.stats[name]
        lines.append(
            f"{name:<26}{s['wins']:>4}{s['losses']:>4}{s['draws']:>4}"
            f"{s['win_rate'] * 100:>7.1f}%{s['average_decision_time']:>12.3f}"
            f"{s['average_nodes_per_decision']:>12.0f}"
        )
    return "\n".join(lines)
