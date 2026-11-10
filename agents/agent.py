"""Agent: a Minimax searcher parameterised by a heuristic and a search depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agents.heuristics import get_heuristic
from agents.minimax import EvaluationFunction, Minimax, SearchResult
from config import DEFAULT_SEARCH_DEPTH
from game.board import Board


@dataclass
class AgentStats:
    decisions: int = 0
    total_time: float = 0.0
    nodes_evaluated: int = 0

    @property
    def average_decision_time(self) -> float:
        return self.total_time / self.decisions if self.decisions else 0.0


class Agent:
    """Selects moves with the shared Minimax search using its own heuristic.

    Create several instances with the same ``heuristic_id`` and different
    ``depth`` values to compare search depths.
    """

    def __init__(self, heuristic_id: str, depth: int = DEFAULT_SEARCH_DEPTH,
                 evaluation_function: Optional[EvaluationFunction] = None):
        self.heuristic_id = heuristic_id
        self.depth = depth
        self.evaluation_function = evaluation_function or get_heuristic(heuristic_id)
        self._search = Minimax(self.evaluation_function, depth)
        self.stats = AgentStats()
        self.last_result: Optional[SearchResult] = None

    def choose_move(self, board: Board) -> int:
        result = self._search.search(board)
        self.last_result = result
        self.stats.decisions += 1
        self.stats.total_time += result.elapsed
        self.stats.nodes_evaluated += result.nodes_evaluated
        return result.move

    def reset_stats(self) -> None:
        self.stats = AgentStats()

    def __repr__(self) -> str:
        return f"Agent({self.heuristic_id!r}, depth={self.depth})"
