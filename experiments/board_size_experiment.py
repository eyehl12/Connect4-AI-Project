"""Compare heuristic tournaments on standard and expanded boards."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from config import BOARD_SIZES, CONNECT_N, EXPERIMENT_SETTINGS, GAMES_PER_SIDE
from agents.heuristics import HEURISTICS
from experiments.tournament import (
    GameRecord,
    TournamentConfig,
    TournamentResult,
    build_schedule,
    compute_matrices,
    heuristic_statistics,
    run_tournament,
)


@dataclass(frozen=True)
class BoardSizeExperimentConfig:
    board_sizes: Sequence[Tuple[int, int]] = BOARD_SIZES
    depth: int = EXPERIMENT_SETTINGS[3].search_depth
    connect: int = CONNECT_N
    games_per_side: int = GAMES_PER_SIDE
    workers: int = 1

    def __post_init__(self):
        if len(self.board_sizes) != 2 or any(rows < 1 or cols < 1
                                             for rows, cols in self.board_sizes):
            raise ValueError("Exactly two positive board dimensions are required.")
        if self.depth < 1 or self.connect < 1 or self.games_per_side < 1:
            raise ValueError("Depth, connect length, and games_per_side must be positive.")


@dataclass
class BoardSizeExperimentResult:
    config: BoardSizeExperimentConfig
    results: List[TournamentResult]
    standard_source: str


def _read_reusable_standard(path: Path) -> Optional[TournamentResult]:
    """Load historical Experiment 1 only when its saved configuration is exact."""
    summary_path = path / "tournament_summary.json"
    games_path = path / "game_results.csv"
    if not summary_path.is_file() or not games_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = summary["configuration"]
        names = cfg["heuristics"]
        if (
            summary.get("completed") is False
            or summary.get("conventions", {}).get("move_selection")
            != "A move is selected randomly among tied best-scoring moves."
            or (cfg["board_rows"], cfg["board_cols"], cfg["connect_n"], cfg["search_depth"])
            != (6, 7, CONNECT_N, 7)
            or cfg["total_games"] != 24
            or cfg["games_per_side"] != 2
            or set(names) != set(HEURISTICS)
        ):
            return None
        records = []
        with games_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                records.append(GameRecord(
                    game_number=int(row["game_number"]),
                    first_heuristic=row["first_heuristic"],
                    second_heuristic=row["second_heuristic"],
                    first_instance=row["first_instance"],
                    winner=row["winner"],
                    winner_heuristic=row["winner_heuristic"],
                    moves=int(row["moves"]),
                    duration=float(row["duration_seconds"]),
                    first_avg_decision_time=float(row["first_avg_decision_time_seconds"]),
                    second_avg_decision_time=float(row["second_avg_decision_time_seconds"]),
                    first_nodes=int(row["first_nodes_evaluated"]),
                    second_nodes=int(row["second_nodes_evaluated"]),
                ))
        expected_schedule = build_schedule(names, 2)
        if len(records) != len(expected_schedule) or any(
            record.game_number != spec.game_number
            or record.first_heuristic != spec.first_heuristic
            or record.second_heuristic != spec.second_heuristic
            or record.first_instance != spec.first_instance
            or record.winner not in ("first", "second", "draw")
            for record, spec in zip(records, expected_schedule)
        ):
            return None
        tournament_config = TournamentConfig(
            heuristics=tuple(names),
            depth=7,
            rows=6,
            cols=7,
            connect=CONNECT_N,
            games_per_side=2,
            workers=1,
        )
        return TournamentResult(
            config=tournament_config,
            records=records,
            matrices=compute_matrices(records, names),
            duration=float(summary.get("total_duration_seconds", 0.0)),
            stats=heuristic_statistics(records, names),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_board_size_experiment(
    results_root: Path,
    config: BoardSizeExperimentConfig = BoardSizeExperimentConfig(),
    progress: Optional[Callable[[Tuple[int, int], GameRecord, int], None]] = None,
) -> BoardSizeExperimentResult:
    """Reuse a configuration-verified standard result, then run the expanded board."""
    results_root = Path(results_root)
    standard_size = config.board_sizes[0]
    standard_candidates = (
        results_root / "experiment_1",
        results_root,
    )
    standard_result = None
    source = "regenerated"
    if (standard_size == (6, 7) and config.depth == 7
            and config.connect == CONNECT_N and config.games_per_side == 2):
        for candidate in standard_candidates:
            standard_result = _read_reusable_standard(candidate)
            if standard_result is not None:
                source = str(candidate)
                break
    results = []
    if standard_result is None:
        standard_config = TournamentConfig(
            depth=config.depth,
            rows=standard_size[0],
            cols=standard_size[1],
            connect=config.connect,
            games_per_side=config.games_per_side,
            workers=config.workers,
        )
        standard_result = run_tournament(
            standard_config,
            (lambda record, total: progress(standard_size, record, total))
            if progress else None,
        )
    results.append(standard_result)

    expanded_size = config.board_sizes[1]
    expanded_config = TournamentConfig(
        depth=config.depth,
        rows=expanded_size[0],
        cols=expanded_size[1],
        connect=config.connect,
        games_per_side=config.games_per_side,
        workers=config.workers,
    )
    expanded_result = run_tournament(
        expanded_config,
        (lambda record, total: progress(expanded_size, record, total))
        if progress else None,
    )
    results.append(expanded_result)
    return BoardSizeExperimentResult(config=config, results=results, standard_source=source)
