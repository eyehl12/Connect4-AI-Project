"""Round-robin tournament between heuristics, plus win-rate matrix computation."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Callable, Dict, List, Optional, Sequence

from agents.agent import Agent
from agents.heuristics import HEURISTICS
from config import COLS, CONNECT_N, DEFAULT_SEARCH_DEPTH, GAMES_PER_SIDE, ROWS
from game.game import Game


@dataclass(frozen=True)
class TournamentConfig:
    heuristics: Sequence[str] = tuple(HEURISTICS)
    depth: int = DEFAULT_SEARCH_DEPTH
    rows: int = ROWS
    cols: int = COLS
    connect: int = CONNECT_N
    games_per_side: int = GAMES_PER_SIDE
    workers: int = 1  # parallel game processes; results do not depend on this


@dataclass(frozen=True)
class GameSpec:
    game_number: int
    first_heuristic: str
    second_heuristic: str
    # "A"/"B" identify the two agent instances of a pairing (A is the first-listed
    # heuristic). This distinguishes the two instances in self-play.
    first_instance: str


@dataclass(frozen=True)
class GameRecord:
    game_number: int
    first_heuristic: str
    second_heuristic: str
    first_instance: str
    winner: str  # "first", "second" or "draw"
    winner_heuristic: str  # heuristic name or "draw"
    moves: int
    duration: float
    first_avg_decision_time: float
    second_avg_decision_time: float
    first_nodes: int
    second_nodes: int


@dataclass
class Matrices:
    """Square matrices indexed [row heuristic][opponent heuristic] in ``names`` order.

    Off-diagonal cells count games between the two distinct heuristics. Diagonal
    (self-play) cells count each game once from each instance's perspective, so
    ``games`` there is twice the number of self-play games.
    """
    names: List[str]
    wins: List[List[int]]
    draws: List[List[int]]
    games: List[List[int]]
    win_rates: List[List[float]]


@dataclass
class TournamentResult:
    config: TournamentConfig
    records: List[GameRecord]
    matrices: Matrices
    duration: float = 0.0
    stats: Dict[str, dict] = field(default_factory=dict)


# ------------------------------------------------------------------ scheduling
def build_schedule(names: Sequence[str], games_per_side: int = GAMES_PER_SIDE) -> List[GameSpec]:
    """One block of games per unordered pairing (self-play included).

    In each pairing instance A plays first ``games_per_side`` times and instance B
    plays first ``games_per_side`` times.
    """
    specs: List[GameSpec] = []
    for a, b in combinations_with_replacement(names, 2):
        for _ in range(games_per_side):
            specs.append(GameSpec(len(specs) + 1, a, b, "A"))
        for _ in range(games_per_side):
            specs.append(GameSpec(len(specs) + 1, b, a, "B"))
    return specs


# --------------------------------------------------------------------- playing
def play_game(spec: GameSpec, config: TournamentConfig) -> GameRecord:
    """Play one game with two fresh, independent agent instances."""
    first = Agent(spec.first_heuristic, config.depth)
    second = Agent(spec.second_heuristic, config.depth)
    result = Game(first, second, config.rows, config.cols, config.connect).play()
    if result.winner == 1:
        winner, winner_heuristic = "first", spec.first_heuristic
    elif result.winner == 2:
        winner, winner_heuristic = "second", spec.second_heuristic
    else:
        winner, winner_heuristic = "draw", "draw"
    return GameRecord(
        game_number=spec.game_number,
        first_heuristic=spec.first_heuristic,
        second_heuristic=spec.second_heuristic,
        first_instance=spec.first_instance,
        winner=winner,
        winner_heuristic=winner_heuristic,
        moves=len(result.moves),
        duration=result.duration,
        first_avg_decision_time=first.stats.average_decision_time,
        second_avg_decision_time=second.stats.average_decision_time,
        first_nodes=first.stats.nodes_evaluated,
        second_nodes=second.stats.nodes_evaluated,
    )


def _play_task(args):
    return play_game(*args)


def run_tournament(config: TournamentConfig = TournamentConfig(),
                   progress: Optional[Callable[[GameRecord, int], None]] = None) -> TournamentResult:
    """Play every scheduled game. ``progress(record, total_games)`` is called per finished game."""
    names = list(config.heuristics)
    schedule = build_schedule(names, config.games_per_side)
    start = time.perf_counter()
    records: List[GameRecord] = []
    tasks = [(spec, config) for spec in schedule]
    if config.workers > 1:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            iterator = pool.map(_play_task, tasks)
            for record in iterator:
                records.append(record)
                if progress:
                    progress(record, len(schedule))
    else:
        for task in tasks:
            record = _play_task(task)
            records.append(record)
            if progress:
                progress(record, len(schedule))
    return TournamentResult(
        config=config,
        records=records,
        matrices=compute_matrices(records, names),
        duration=time.perf_counter() - start,
        stats=heuristic_statistics(records, names),
    )


# ------------------------------------------------------------------- analysis
def compute_matrices(records: Sequence[GameRecord], names: Sequence[str]) -> Matrices:
    """Aggregate win counts, draw counts and win rates.

    ``win_rate[i][j] = wins of i against j / games i played against j``.
    Distinct heuristics: the games between them (both first-player arrangements).
    Self-play (i == j): each game is counted once from each instance's perspective,
    so a decisive game adds 1 win (and 1 loss) and a draw adds 2 draws; the
    denominator is 2 x the number of self-play games.
    """
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    wins = [[0] * n for _ in range(n)]
    draws = [[0] * n for _ in range(n)]
    games = [[0] * n for _ in range(n)]
    for rec in records:
        i, j = index[rec.first_heuristic], index[rec.second_heuristic]
        if i == j:
            games[i][i] += 2
            if rec.winner == "draw":
                draws[i][i] += 2
            else:
                wins[i][i] += 1
            continue
        games[i][j] += 1
        games[j][i] += 1
        if rec.winner == "draw":
            draws[i][j] += 1
            draws[j][i] += 1
        elif rec.winner == "first":
            wins[i][j] += 1
        else:
            wins[j][i] += 1
    win_rates = [[wins[i][j] / games[i][j] if games[i][j] else 0.0 for j in range(n)]
                 for i in range(n)]
    return Matrices(list(names), wins, draws, games, win_rates)


def heuristic_statistics(records: Sequence[GameRecord], names: Sequence[str]) -> Dict[str, dict]:
    """Per-heuristic wins/losses/draws and performance totals.

    Every game gives each participating agent one result (self-play therefore
    contributes both a win and a loss, or two draws, to that heuristic).
    ``*_vs_others`` fields exclude self-play games.
    """
    stats = {
        name: {"games": 0, "wins": 0, "losses": 0, "draws": 0,
               "wins_vs_others": 0, "losses_vs_others": 0, "draws_vs_others": 0,
               "decisions": 0, "total_decision_time": 0.0, "nodes_evaluated": 0}
        for name in names
    }
    for rec in records:
        first_decisions = (rec.moves + 1) // 2
        second_decisions = rec.moves // 2
        sides = (
            (rec.first_heuristic, "first", first_decisions, rec.first_avg_decision_time, rec.first_nodes),
            (rec.second_heuristic, "second", second_decisions, rec.second_avg_decision_time, rec.second_nodes),
        )
        for name, side, decisions, avg_time, nodes in sides:
            entry = stats[name]
            outcome = "draws" if rec.winner == "draw" else ("wins" if rec.winner == side else "losses")
            entry["games"] += 1
            entry[outcome] += 1
            if rec.first_heuristic != rec.second_heuristic:
                entry[outcome + "_vs_others"] += 1
            entry["decisions"] += decisions
            entry["total_decision_time"] += avg_time * decisions
            entry["nodes_evaluated"] += nodes
    for entry in stats.values():
        games, decisions = entry["games"], entry["decisions"]
        entry["win_rate"] = entry["wins"] / games if games else 0.0
        entry["average_decision_time"] = entry["total_decision_time"] / decisions if decisions else 0.0
        entry["average_nodes_per_decision"] = entry["nodes_evaluated"] / decisions if decisions else 0.0
    return stats
