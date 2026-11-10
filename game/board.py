"""Connect Four board with fast make/undo moves and incremental win detection."""

from __future__ import annotations

from typing import List, Sequence

from config import COLS, CONNECT_N, EMPTY, PLAYER_ONE, PLAYER_TWO, ROWS

# Four line orientations: horizontal, vertical, and the two diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

_SYMBOLS = {EMPTY: ".", PLAYER_ONE: "X", PLAYER_TWO: "O"}


class InvalidMoveError(ValueError):
    """Raised when a move is illegal (bad column, full column, or game over)."""


class Board:
    """A Connect Four board.

    ``grid[row][col]`` holds 0 (empty), 1 or 2. Row 0 is the TOP row, so pieces
    settle at the highest row index available in a column. ``heights[col]`` is
    the number of pieces in that column.
    """

    __slots__ = (
        "rows", "cols", "connect", "cells", "grid", "heights",
        "current_player", "move_count", "winner", "column_order", "_history",
    )

    def __init__(self, rows: int = ROWS, cols: int = COLS, connect: int = CONNECT_N):
        self.rows = rows
        self.cols = cols
        self.connect = connect
        self.cells = rows * cols
        self.grid: List[List[int]] = [[EMPTY] * cols for _ in range(rows)]
        self.heights: List[int] = [0] * cols
        self.current_player = PLAYER_ONE
        self.move_count = 0
        self.winner = EMPTY
        # Center-outward column order (ties favour the lower index) for move ordering.
        centre = (cols - 1) / 2
        self.column_order = tuple(sorted(range(cols), key=lambda c: (abs(c - centre), c)))
        self._history: List[int] = []

    @classmethod
    def from_grid(cls, grid: Sequence[Sequence[int]], connect: int = CONNECT_N) -> "Board":
        """Build a board from a gravity-consistent grid (row 0 = top).

        The side to move is inferred from the piece counts. Boards created this way
        have no move history, so ``undo_move`` is only valid for later moves.
        """
        board = cls(len(grid), len(grid[0]), connect)
        ones = twos = 0
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                board.grid[r][c] = value
                if value == PLAYER_ONE:
                    ones += 1
                elif value == PLAYER_TWO:
                    twos += 1
        for c in range(board.cols):
            board.heights[c] = sum(1 for r in range(board.rows) if board.grid[r][c] != EMPTY)
        board.move_count = ones + twos
        board.current_player = PLAYER_ONE if ones == twos else PLAYER_TWO
        for r in range(board.rows):
            for c in range(board.cols):
                value = board.grid[r][c]
                if value != EMPTY and board._wins_through(r, c, value):
                    board.winner = value
        return board

    # ------------------------------------------------------------------ moves
    def is_valid_move(self, col: int) -> bool:
        return 0 <= col < self.cols and self.heights[col] < self.rows and not self.winner

    def legal_moves(self) -> List[int]:
        """Playable columns, ordered from the center outward."""
        if self.winner:
            return []
        heights, rows = self.heights, self.rows
        return [c for c in self.column_order if heights[c] < rows]

    def next_row(self, col: int) -> int:
        """Row where a piece dropped in ``col`` would land (assumes the column is not full)."""
        return self.rows - 1 - self.heights[col]

    def make_move(self, col: int) -> int:
        """Drop the current player's piece in ``col`` and return the row it lands on."""
        if self.winner:
            raise InvalidMoveError("The game is already over.")
        if not 0 <= col < self.cols:
            raise InvalidMoveError(f"Column {col} is out of range.")
        if self.heights[col] >= self.rows:
            raise InvalidMoveError(f"Column {col} is full.")
        row = self.rows - 1 - self.heights[col]
        player = self.current_player
        self.grid[row][col] = player
        self.heights[col] += 1
        self.move_count += 1
        self._history.append(col)
        if self._wins_through(row, col, player):
            self.winner = player
        self.current_player = 3 - player
        return row

    def undo_move(self) -> int:
        """Revert the most recent move and return the column it was played in."""
        if not self._history:
            raise InvalidMoveError("No moves to undo.")
        col = self._history.pop()
        self.heights[col] -= 1
        self.grid[self.rows - 1 - self.heights[col]][col] = EMPTY
        self.move_count -= 1
        self.current_player = 3 - self.current_player
        self.winner = EMPTY  # a game with a winner never accepts further moves
        return col

    # ---------------------------------------------------------- terminal state
    def is_full(self) -> bool:
        return self.move_count == self.cells

    def is_draw(self) -> bool:
        return self.winner == EMPTY and self.is_full()

    def is_terminal(self) -> bool:
        return self.winner != EMPTY or self.is_full()

    def is_winning_move(self, col: int, player: int) -> bool:
        """True if ``player`` dropping a piece in ``col`` would make a line (board is not changed)."""
        if not 0 <= col < self.cols or self.heights[col] >= self.rows:
            return False
        return self._wins_through(self.rows - 1 - self.heights[col], col, player)

    def _wins_through(self, row: int, col: int, player: int) -> bool:
        """Does a line of ``connect`` pieces of ``player`` pass through (row, col)?

        The cell itself is not read, so this also answers the hypothetical
        "would a piece placed here win?" question.
        """
        grid, rows, cols, need = self.grid, self.rows, self.cols, self.connect
        for dr, dc in DIRECTIONS:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < cols and grid[r][c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < cols and grid[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= need:
                return True
        return False

    # ---------------------------------------------------------------- display
    def __str__(self) -> str:
        lines = [" ".join(_SYMBOLS[v] for v in row) for row in self.grid]
        lines.append(" ".join(str(c % 10) for c in range(self.cols)))
        return "\n".join(lines)
