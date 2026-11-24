"""Compare each heuristic's search depths against its own fixed-depth baseline."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence

from agents.agent import Agent
from agents.heuristics import HEURISTICS
from config import CONNECT_N, EXPERIMENT_SETTINGS, GAMES_PER_SIDE, SEARCH_DEPTHS
from game.game import Game


@dataclass(frozen=True)
class DepthExperimentConfig:
    depths: Sequence[int] = SEARCH_DEPTHS
    baseline_depth: int = 7
    rows: int = EXPERIMENT_SETTINGS[2].rows
    cols: int = EXPERIMENT_SETTINGS[2].cols
    connect: int = CONNECT_N
    games_per_side: int = GAMES_PER_SIDE
    heuristics: Sequence[str] = tuple(HEURISTICS)
    workers: int = 1

    def __post_init__(self):
        if not self.depths or any(depth < 1 for depth in self.depths):
            raise ValueError("Search depths must be positive.")
        if len(set(self.depths)) != len(self.depths):
            raise ValueError("Search depths must be unique.")
        if self.baseline_depth not in self.depths:
            raise ValueError("The fixed baseline depth must be among the tested depths.")
        if self.games_per_side < 1:
            raise ValueError("games_per_side must be at least one.")
        if self.rows < 1 or self.cols < 1 or self.connect < 1:
            raise ValueError("Board dimensions and connect length must be positive.")


@dataclass(frozen=True)
class DepthGameSpec:
    game_number: int
    heuristic: str
    evaluated_depth: int
    evaluated_side: str


@dataclass(frozen=True)
class DepthGameRecord:
    game_number: int
    heuristic: str
    evaluated_depth: int
    baseline_depth: int
    evaluated_side: str
    winner: str
    moves: int
    duration: float
    evaluated_avg_decision_time: float
    baseline_avg_decision_time: float
    evaluated_nodes: int
    baseline_nodes: int


@dataclass
class DepthExperimentResult:
    config: DepthExperimentConfig
    records: List[DepthGameRecord]
    stats: Dict[str, dict]
    duration: float = 0.0
    completed: bool = True


def build_depth_schedule(config: DepthExperimentConfig) -> List[DepthGameSpec]:
    """Schedule each tested heuristic/depth four times with balanced first moves."""
    specs = []
    for heuristic, depth in product(config.heuristics, config.depths):
        for side in ("first", "second"):
            for _ in range(config.games_per_side):
                specs.append(DepthGameSpec(len(specs) + 1, heuristic, depth, side))
    return specs


def play_depth_game(spec: DepthGameSpec, config: DepthExperimentConfig) -> DepthGameRecord:
    """Run one evaluated-agent game against the same heuristic's depth baseline."""
    evaluated = Agent(spec.heuristic, spec.evaluated_depth)
    baseline = Agent(spec.heuristic, config.baseline_depth)
    if spec.evaluated_side == "first":
        first, second = evaluated, baseline
    else:
        first, second = baseline, evaluated
    game_result = Game(first, second, config.rows, config.cols, config.connect).play()
    if game_result.winner == 0:
        winner = "draw"
    elif (game_result.winner == 1) == (spec.evaluated_side == "first"):
        winner = "evaluated"
    else:
        winner = "baseline"
    return DepthGameRecord(
        game_number=spec.game_number,
        heuristic=spec.heuristic,
        evaluated_depth=spec.evaluated_depth,
        baseline_depth=config.baseline_depth,
        evaluated_side=spec.evaluated_side,
        winner=winner,
        moves=len(game_result.moves),
        duration=game_result.duration,
        evaluated_avg_decision_time=evaluated.stats.average_decision_time,
        baseline_avg_decision_time=baseline.stats.average_decision_time,
        evaluated_nodes=evaluated.stats.nodes_evaluated,
        baseline_nodes=baseline.stats.nodes_evaluated,
    )


def _play_depth_task(args):
    return play_depth_game(*args)


def _depth_statistics(records: Sequence[DepthGameRecord],
                      config: DepthExperimentConfig) -> Dict[str, dict]:
    statistics = {}
    for heuristic, depth in product(config.heuristics, config.depths):
        key = f"{heuristic}-D{depth}"
        games = [record for record in records
                 if record.heuristic == heuristic and record.evaluated_depth == depth]
        wins = sum(record.winner == "evaluated" for record in games)
        losses = sum(record.winner == "baseline" for record in games)
        draws = sum(record.winner == "draw" for record in games)
        decisions = sum((record.moves + (1 if record.evaluated_side == "first" else 0)) // 2
                        for record in games)
        total_time = sum(
            record.evaluated_avg_decision_time
            * ((record.moves + (1 if record.evaluated_side == "first" else 0)) // 2)
            for record in games
        )
        total_nodes = sum(record.evaluated_nodes for record in games)
        count = len(games)
        statistics[key] = {
            "heuristic": heuristic,
            "depth": depth,
            "baseline_depth": config.baseline_depth,
            "games": count,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / count if count else 0.0,
            "draw_rate": draws / count if count else 0.0,
            "decisions": decisions,
            "average_decision_time": total_time / decisions if decisions else 0.0,
            "average_nodes_per_decision": total_nodes / decisions if decisions else 0.0,
            "total_nodes_evaluated": total_nodes,
            "average_game_duration": (
                sum(record.duration for record in games) / count if count else 0.0
            ),
            "average_moves_per_game": (
                sum(record.moves for record in games) / count if count else 0.0
            ),
        }
    return statistics


def run_depth_experiment(
    config: DepthExperimentConfig = DepthExperimentConfig(),
    progress: Optional[Callable[[DepthGameRecord, int], None]] = None,
) -> DepthExperimentResult:
    """Run all fixed-baseline depth matchups and aggregate only completed games."""
    schedule = build_depth_schedule(config)
    tasks = [(spec, config) for spec in schedule]
    records = []
    start = time.perf_counter()
    if config.workers > 1:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            iterator = pool.map(_play_depth_task, tasks)
            for record in iterator:
                records.append(record)
                if progress:
                    progress(record, len(schedule))
    else:
        for task in tasks:
            record = _play_depth_task(task)
            records.append(record)
            if progress:
                progress(record, len(schedule))
    return DepthExperimentResult(
        config=config,
        records=records,
        stats=_depth_statistics(records, config),
        duration=time.perf_counter() - start,
        completed=len(records) == len(schedule),
    )
