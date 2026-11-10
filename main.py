"""Command-line entry point.

    python main.py                       run the full tournament and save results
    python main.py play H1 H3            play one verbose debugging game
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agents.agent import Agent
from agents.heuristics import HEURISTIC_ABBREVIATIONS, HEURISTICS
from config import COLS, DEFAULT_SEARCH_DEPTH, RESULTS_DIR, ROWS
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


def _run_tournament(args) -> int:
    config = TournamentConfig(depth=args.depth, workers=args.workers)

    def progress(record, total):
        print(f"[{record.game_number:>2}/{total}] {record.first_heuristic} (first) vs "
              f"{record.second_heuristic}: {record.winner_heuristic} "
              f"({record.moves} moves, {record.duration:.1f}s)", flush=True)

    print(f"Running tournament at depth {config.depth} with {config.workers} worker(s)...")
    result = run_tournament(config, progress)
    paths = save_results(result, Path(args.results_dir))
    print()
    print(format_summary(result))
    print()
    print(f"Results written to {Path(args.results_dir)}: " + ", ".join(p.name for p in paths))
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
    parser = argparse.ArgumentParser(description="Connect Four heuristic comparison.")
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
    return _run_tournament(args)


if __name__ == "__main__":
    sys.exit(main())
