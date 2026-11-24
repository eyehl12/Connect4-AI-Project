"""Command-line entry point.

    python main.py                       run the full tournament and save results
    python main.py play H1 H3            play one verbose debugging game
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from agents.agent import Agent
from agents.heuristics import HEURISTIC_ABBREVIATIONS, HEURISTICS
from config import (
    BOARD_SIZES,
    COLS,
    DEFAULT_SEARCH_DEPTH,
    EXPERIMENT_SETTINGS,
    RESULTS_DIR,
    ROWS,
    SEARCH_DEPTHS,
)
from experiments.advanced_results import (
    save_board_size_results,
    save_depth_results,
    update_combined_summary,
)
from experiments.board_size_experiment import (
    BoardSizeExperimentConfig,
    run_board_size_experiment,
)
from experiments.depth_experiment import DepthExperimentConfig, run_depth_experiment
from experiments.results import format_summary, save_results
from experiments.tournament import TournamentConfig, run_tournament
from game.game import Game

_ALIASES = {abbr: name for name, abbr in HEURISTIC_ABBREVIATIONS.items()}


def _resolve(name: str) -> str:
    name = _ALIASES.get(name.upper(), name)
    if name not in HEURISTICS:
        raise argparse.ArgumentTypeError(
            f"unknown heuristic '{name}' (choose from {', '.join(HEURISTICS)} or H1-H3)")
    return name


def _positive_even_games(value: str) -> int:
    try:
        games = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("games must be a positive even integer") from None
    if games < 2 or games % 2:
        raise argparse.ArgumentTypeError("games must be a positive even integer")
    return games


def _run_experiment_1(args, runtime_start: float) -> None:
    settings = EXPERIMENT_SETTINGS[1]
    config = TournamentConfig(
        depth=settings.search_depth, rows=settings.rows, cols=settings.cols,
        games_per_side=args.games // 2, workers=args.workers,
    )

    def progress(record, total):
        print(f"[Experiment 1 {record.game_number:>2}/{total}] "
              f"{record.first_heuristic} (first) vs "
              f"{record.second_heuristic}: {record.winner_heuristic} "
              f"({record.moves} moves; elapsed {time.perf_counter() - runtime_start:.1f}s)",
              flush=True)

    print(f"Running Experiment 1: heuristic performance, depth {config.depth}, "
          f"board {config.rows}x{config.cols}.")
    result = run_tournament(config, progress)
    out = Path(args.results_dir) / "experiment_1"
    paths = save_results(result, out)
    update_combined_summary(Path(args.results_dir))
    print(format_summary(result))
    print(f"Experiment 1 complete: {len(result.records)} games in {result.duration:.1f}s; "
          f"results in {out} ({len(paths)} files).")


def _run_experiment_2(args, runtime_start: float) -> None:
    config = DepthExperimentConfig(
        depths=tuple(args.depths), games_per_side=args.games // 2, workers=args.workers
    )

    def progress(record, total):
        print(f"[Experiment 2 {record.game_number:>2}/{total}] "
              f"{record.heuristic} D{record.evaluated_depth} vs fixed "
              f"D{record.baseline_depth} (evaluated agent {record.evaluated_side}; "
              f"{record.winner}; elapsed {time.perf_counter() - runtime_start:.1f}s)",
              flush=True)

    print(f"Running Experiment 2: search depths {list(config.depths)} on "
          f"{config.rows}x{config.cols}; baseline D{config.baseline_depth}.")
    result = run_depth_experiment(config, progress)
    out = save_depth_results(result, Path(args.results_dir))
    update_combined_summary(Path(args.results_dir))
    for heuristic in config.heuristics:
        for depth in config.depths:
            stats = result.stats[f"{heuristic}-D{depth}"]
            print(f"  {heuristic}-D{depth}: {stats['wins']}W/"
                  f"{stats['losses']}L/{stats['draws']}D "
                  f"({stats['win_rate']:.1%} wins; "
                  f"{stats['average_decision_time']:.3f}s/decision)")
    print(f"Experiment 2 complete: {len(result.records)} games in {result.duration:.1f}s; "
          f"results in {out}.")


def _run_experiment_3(args, runtime_start: float) -> None:
    config = BoardSizeExperimentConfig(
        board_sizes=BOARD_SIZES, games_per_side=args.games // 2, workers=args.workers
    )

    def progress(board_size, record, total):
        print(f"[Experiment 3 {board_size[0]}x{board_size[1]} "
              f"{record.game_number:>2}/{total}] {record.first_heuristic} vs "
              f"{record.second_heuristic}: {record.winner_heuristic} "
              f"(elapsed {time.perf_counter() - runtime_start:.1f}s)", flush=True)

    print(f"Running Experiment 3: boards {config.board_sizes}, depth {config.depth}.")
    result = run_board_size_experiment(Path(args.results_dir), config, progress)
    out = save_board_size_results(result, Path(args.results_dir))
    update_combined_summary(Path(args.results_dir))
    print(f"Experiment 3 complete: {sum(len(item.records) for item in result.results)} "
          f"games; standard-board data {result.standard_source}; results in {out}.")


def _run_experiments(args) -> int:
    runtime_start = time.perf_counter()
    runners = {
        "1": _run_experiment_1,
        "2": _run_experiment_2,
        "3": _run_experiment_3,
    }
    selected = list(runners) if args.experiment == "all" else [args.experiment]
    for experiment in selected:
        runners[experiment](args, runtime_start)
    return 0


def _play_single(args) -> int:
    first = Agent(args.first, args.depth)
    second = Agent(args.second, args.depth)

    def show(board, col, player):
        print(f"Move {board.move_count}: player {player} -> column {col}\n{board}\n", flush=True)

    result = Game(first, second, ROWS, COLS).play(on_move=show)
    outcome = "draw" if result.winner == 0 else f"player {result.winner} wins"
    print(f"Result: {outcome} after {len(result.moves)} moves ({result.duration:.1f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Connect Four AI experiments.")
    parser.add_argument("--experiment", choices=("1", "2", "3", "all"), default="1",
                        help="experiment to run (default: 1)")
    parser.add_argument("--depths", nargs="+", type=int, default=list(SEARCH_DEPTHS),
                        help="depths tested in Experiment 2 (default: 4 7 10)")
    parser.add_argument("--games", type=_positive_even_games, default=4,
                        help="total games per matchup/configuration, evenly split by first player (default: 4)")
    parser.add_argument("--depth", type=int, default=DEFAULT_SEARCH_DEPTH, help="search depth in plies")
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)),
                        help="parallel game processes for the tournament")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("tournament", help="run the full tournament (default)")
    play = sub.add_parser("play", help="play one game and print the board each turn")
    play.add_argument("first", type=_resolve, help="player 1 heuristic (name or H1-H3)")
    play.add_argument("second", type=_resolve, help="player 2 heuristic (name or H1-H3)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "play":
        return _play_single(args)
    return _run_experiments(args)


if __name__ == "__main__":
    sys.exit(main())
