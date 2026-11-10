"""Centralised configuration constants for the Connect Four AI project."""

from pathlib import Path

# Board geometry (changeable in a future stage, e.g. 10 x 11).
ROWS = 6
COLS = 7
CONNECT_N = 4

# Cell contents.
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = 2

# Search.
DEFAULT_SEARCH_DEPTH = 7  # plies

# Terminal scores. A win at ply p scores WIN_SCORE - p (faster wins are better),
# a loss scores -(WIN_SCORE - p) (slower losses are better).
WIN_SCORE = 1_000_000

# Points for a viable connected run of the given length (see agents/heuristics.py).
CONNECTION_WEIGHTS = {2: 2, 3: 5}

# Bonus/penalty for a (likely) forced win detected by the multiple-winning-moves
# heuristic. Must stay well below WIN_SCORE so real terminal states always dominate.
MULTI_THREAT_SCORE = 50_000

# Tournament.
GAMES_PER_SIDE = 2  # games each side of a pairing plays as first player
RESULTS_DIR = Path(__file__).resolve().parent / "results"
