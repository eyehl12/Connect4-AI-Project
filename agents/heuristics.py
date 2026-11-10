"""Heuristic evaluation functions.

All heuristics share one signature, ``heuristic(board, player) -> number``, and
return a score from ``player``'s perspective (positive = good for ``player``).
They are only called on non-terminal positions; Minimax scores wins, losses and
draws itself.

How connections are counted (shared by all three heuristics)
------------------------------------------------------------
A *run* is a maximal sequence of consecutive pieces of one player along one of
the four line orientations (horizontal, vertical, diagonal "\\", diagonal "/").
Every run is counted exactly once, in the orientation it lies in, so a line of
three is worth 5 points, not 5 + 2 for its sub-pairs.

* Runs of length 1 score nothing. Runs of length 2 and 3 score the weights in
  ``config.CONNECTION_WEIGHTS`` (2 and 5). Runs of length >= 4 are wins and are
  handled as terminal states, so they are ignored here.
* A run is *viable* only if it can still grow to four. From each end of the run
  we look outward through cells that are empty or hold the player's own piece,
  stopping at an opponent piece or the board edge (both count as "blocked"). The
  run is viable if ``length + room_before + room_after >= 4``. A run blocked at
  both ends with too little room (e.g. an opponent piece on one side and the
  board edge two cells away on the other) scores nothing.
* Room is judged by geometry only: an empty cell counts even if it is not yet
  playable under gravity. Gravity is handled by the winning-move logic in
  heuristic 3.
* Only contiguous runs score; split patterns such as ``X X _ X`` score the
  2-run and the 1-run separately (heuristic 3 picks such threats up through its
  winning-move detection).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from config import CONNECTION_WEIGHTS, EMPTY, MULTI_THREAT_SCORE
from game.board import DIRECTIONS, Board

Heuristic = Callable[[Board, int], float]


# --------------------------------------------------------- shared connection logic
def connection_scores(board: Board) -> Tuple[int, int, int]:
    """Return ``(0, player1_score, player2_score)`` of viable weighted connections.

    The leading 0 lets the result be indexed directly by player number. One scan
    scores both players, so every heuristic uses identical connection counting.
    """
    grid, rows, cols, need = board.grid, board.rows, board.cols, board.connect
    weights = CONNECTION_WEIGHTS
    scores = [0, 0, 0]
    for dr, dc in DIRECTIONS:
        for r in range(rows):
            row = grid[r]
            for c in range(cols):
                player = row[c]
                if player == EMPTY:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < rows and 0 <= pc < cols and grid[pr][pc] == player:
                    continue  # not the start of a run
                length = 1
                nr, nc = r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == player:
                    length += 1
                    nr += dr
                    nc += dc
                weight = weights.get(length)
                if weight is None:
                    continue
                missing = need - length
                room = length
                # Room after the run (starts at the first cell past it: nr, nc).
                k = 0
                while (k < missing and 0 <= nr < rows and 0 <= nc < cols
                       and grid[nr][nc] != 3 - player):
                    k += 1
                    nr += dr
                    nc += dc
                room += k
                # Room before the run.
                br, bc = r - dr, c - dc
                k = 0
                while (k < missing and 0 <= br < rows and 0 <= bc < cols
                       and grid[br][bc] != 3 - player):
                    k += 1
                    br -= dr
                    bc -= dc
                room += k
                if room >= need:
                    scores[player] += weight
    return scores[0], scores[1], scores[2]


# ----------------------------------------------------- multiple winning moves logic
def winning_moves(board: Board, player: int) -> List[int]:
    """Columns where ``player`` could drop a piece right now and complete a line.

    Only the lowest empty cell of each column is playable, so a winning cell that
    is not yet supported from below is not reported.
    """
    return [c for c in range(board.cols) if board.is_winning_move(c, player)]


def multiple_threat_adjustment(board: Board, player: int) -> int:
    """Bonus/penalty for forced-win threat situations, from ``player``'s view.

    Takes whose turn it is into account (isolated here so it can be changed
    independently of the connection scoring):

    * Side to move has at least one legal winning move: it wins immediately.
    * Side to move has none, but the other side has two or more distinct legal
      winning moves: the mover can block at most one, so the other side wins.
    * A single threat for the side NOT on move is ignored, because the mover can
      simply block it.
    """
    opponent = 3 - player
    mover = board.current_player
    mine = len(winning_moves(board, player))
    theirs = len(winning_moves(board, opponent))
    if mover == player:
        if mine >= 1:
            return MULTI_THREAT_SCORE
        if theirs >= 2:
            return -MULTI_THREAT_SCORE
    else:
        if theirs >= 1:
            return -MULTI_THREAT_SCORE
        if mine >= 2:
            return MULTI_THREAT_SCORE
    return 0


# ------------------------------------------------------------------ the heuristics
def weighted_connections(board: Board, player: int) -> float:
    """Heuristic 1: the player's own viable weighted connections only."""
    return connection_scores(board)[player]


def opponent_awareness(board: Board, player: int) -> float:
    """Heuristic 2: own connection score minus the opponent's."""
    scores = connection_scores(board)
    return scores[player] - scores[3 - player]


def multiple_winning_moves(board: Board, player: int) -> float:
    """Heuristic 3: heuristic 2 plus forced-win (multiple winning move) detection."""
    return opponent_awareness(board, player) + multiple_threat_adjustment(board, player)


HEURISTICS: Dict[str, Heuristic] = {
    "weighted_connections": weighted_connections,
    "opponent_awareness": opponent_awareness,
    "multiple_winning_moves": multiple_winning_moves,
}

# Documented abbreviations used in the README.
HEURISTIC_ABBREVIATIONS = {
    "weighted_connections": "H1",
    "opponent_awareness": "H2",
    "multiple_winning_moves": "H3",
}


def get_heuristic(name: str) -> Heuristic:
    try:
        return HEURISTICS[name]
    except KeyError:
        raise ValueError(f"Unknown heuristic '{name}'. Choose from: {', '.join(HEURISTICS)}") from None
