"""Minimax search with alpha-beta pruning, center-first move ordering and make/undo."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

from config import DEFAULT_SEARCH_DEPTH, WIN_SCORE
from game.board import Board

EvaluationFunction = Callable[[Board, int], float]

_INFINITY = WIN_SCORE * 10


@dataclass(frozen=True)
class SearchResult:
    move: int
    score: float
    nodes_evaluated: int  # terminal states plus heuristic evaluations
    elapsed: float  # seconds


class Minimax:
    """Depth-limited Minimax with alpha-beta pruning.

    Scores are always from the perspective of the player to move at the root
    (the agent choosing a move), which is the maximizing player at every level
    where it is that player's turn.
    """

    def __init__(self, evaluate: EvaluationFunction, depth: int = DEFAULT_SEARCH_DEPTH):
        if depth < 1:
            raise ValueError("Search depth must be at least 1.")
        self.evaluate = evaluate
        self.depth = depth
        self._me = 0
        self._nodes = 0

    def search(self, board: Board) -> SearchResult:
        """Pick the best column for ``board.current_player``.

        Center-first ordering controls search traversal. If several moves have the
        same best score, one is selected randomly. Root moves use full alpha-beta
        windows so tie detection compares exact scores rather than bounds.
        """
        if board.is_terminal():
            raise ValueError("Cannot search a finished game.")
        start = time.perf_counter()
        self._me = board.current_player
        self._nodes = 0
        best_moves = []
        best_score = -_INFINITY
        for col in board.legal_moves():
            board.make_move(col)
            score = self._value(
                board, self.depth - 1, -_INFINITY, _INFINITY, False, 1
            )
            board.undo_move()
            if score > best_score:
                best_moves = [col]
                best_score = score
            elif score == best_score:
                best_moves.append(col)
        move = best_moves[0] if len(best_moves) == 1 else random.choice(best_moves)
        return SearchResult(move, best_score, self._nodes, time.perf_counter() - start)

    def _value(self, board: Board, depth: int, alpha: float, beta: float,
               maximizing: bool, ply: int) -> float:
        winner = board.winner
        if winner:
            self._nodes += 1
            return WIN_SCORE - ply if winner == self._me else ply - WIN_SCORE
        if board.move_count == board.cells:
            self._nodes += 1
            return 0
        if depth == 0:
            self._nodes += 1
            return self.evaluate(board, self._me)

        heights, rows = board.heights, board.rows
        if maximizing:
            best = -_INFINITY
            for col in board.column_order:
                if heights[col] >= rows:
                    continue
                board.make_move(col)
                score = self._value(board, depth - 1, alpha, beta, False, ply + 1)
                board.undo_move()
                if score > best:
                    best = score
                    if best > alpha:
                        alpha = best
                        if alpha >= beta:
                            break
            return best
        best = _INFINITY
        for col in board.column_order:
            if heights[col] >= rows:
                continue
            board.make_move(col)
            score = self._value(board, depth - 1, alpha, beta, True, ply + 1)
            board.undo_move()
            if score < best:
                best = score
                if best < beta:
                    beta = best
                    if alpha >= beta:
                        break
        return best
