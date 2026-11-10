"""Runs a full game between two move-selecting agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from config import COLS, CONNECT_N, EMPTY, PLAYER_ONE, PLAYER_TWO, ROWS
from game.board import Board

MoveCallback = Callable[[Board, int, int], None]


@dataclass
class GameResult:
    winner: int  # 0 = draw, 1 = first player, 2 = second player
    moves: List[int] = field(default_factory=list)
    duration: float = 0.0


class Game:
    """Plays ``first_agent`` (player 1) against ``second_agent`` (player 2).

    Agents only need a ``choose_move(board) -> int`` method.
    """

    def __init__(self, first_agent, second_agent, rows: int = ROWS, cols: int = COLS,
                 connect: int = CONNECT_N):
        self.agents = {PLAYER_ONE: first_agent, PLAYER_TWO: second_agent}
        self.board = Board(rows, cols, connect)

    def play(self, on_move: Optional[MoveCallback] = None) -> GameResult:
        """Play to completion. ``on_move(board, column, player)`` is called after each move."""
        board = self.board
        moves: List[int] = []
        start = time.perf_counter()
        while not board.is_terminal():
            player = board.current_player
            col = self.agents[player].choose_move(board)
            board.make_move(col)  # raises InvalidMoveError on an illegal choice
            moves.append(col)
            if on_move is not None:
                on_move(board, col, player)
        winner = board.winner if board.winner != EMPTY else 0
        return GameResult(winner=winner, moves=moves, duration=time.perf_counter() - start)
