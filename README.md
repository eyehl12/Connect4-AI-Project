# Connect Four AI – Heuristic Comparison (Stage 1)

A Connect Four engine, three Minimax (alpha-beta) agents that differ only in their
heuristic evaluation function, and an automated round-robin tournament.

## Project structure

```
main.py                    CLI entry point (tournament / single debug game)
config.py                  Central constants (board size, depth, scores, weights, paths)
game/board.py              Board: make/undo moves, incremental win detection, text display
game/game.py               Game loop for two agents
agents/heuristics.py       Shared connection logic + the three heuristics
agents/minimax.py          Minimax with alpha-beta pruning, center-first ordering, metrics
agents/agent.py           Agent = heuristic + depth + Minimax, with per-agent statistics
experiments/tournament.py  Schedule, game play, win-rate matrix computation
experiments/results.py     CSV/JSON output and terminal formatting
tests/                     Unit tests (standard library `unittest`)
results/                   Generated outputs (overwritten on each run)
```

## Install

Python 3.9+ is required. There are no third-party dependencies.

## Run

```
python main.py                    # full tournament (24 games, depth 7), saves results/
python main.py --workers 1        # run games sequentially instead of in parallel
python main.py play H1 H3         # one debugging game, board printed every turn
python main.py --depth 4 play H2 H2
```

Heuristics can be named in full or by abbreviation (H1, H2, H3). Global options such
as `--depth`, `--workers` and `--results-dir` go before the sub-command. Games run in
separate processes by default (up to 8); results are identical regardless of worker
count because every agent is deterministic. A full depth-7 tournament takes roughly
a minute on an 8-core machine.

## Tests

```
python -m unittest discover -s tests -t .
```

The tests use shallow searches and finish in a couple of seconds.

## The three heuristics

All three share one connection counter (`connection_scores`) so comparisons are fair.
Wins, losses and draws are scored by Minimax before any heuristic is consulted
(win = `1_000_000 - ply`, loss = `-(1_000_000 - ply)`, draw = 0).

**How connections are counted.** A *run* is a maximal sequence of consecutive pieces of
one player in one of four orientations (horizontal, vertical, both diagonals). Each
run is counted once: 2 pieces = 2 points, 3 pieces = 5 points, 1 piece = 0, and 4+
is a win (terminal). A run only scores if it is *viable*: looking outward from both
ends through cells that are empty or the player's own, stopping at an opponent piece
or the board edge, `run length + room before + room after >= 4`. Runs blocked at both
ends without enough room score nothing. Room is geometric (an empty cell counts even
if it is not yet playable). Only contiguous runs score.

| ID | Name | Evaluation |
|----|------|------------|
| H1 | `weighted_connections` | own connection score |
| H2 | `opponent_awareness` | own score − opponent score |
| H3 | `multiple_winning_moves` | H2 + forced-win threat adjustment |

**H3 threat adjustment** (`multiple_threat_adjustment`, isolated for later changes). A
*winning move* is a cell that completes four and is playable now (the lowest empty
cell of its column). With `MULTI_THREAT_SCORE = 50_000`:

- side to move has ≥ 1 winning move → that side is rewarded (it wins next move);
- otherwise, the other side has ≥ 2 distinct winning moves → that side is rewarded
  (the mover can block only one);
- a single threat for the side not on move is ignored (it can be blocked).

## Tournament and win-rate matrix

The three heuristics form six unordered pairings (H1-H1, H1-H2, H1-H3, H2-H2, H2-H3,
H3-H3). Each pairing plays 4 games (each side first twice), giving 24 games at depth
7. Self-play uses two independent agent instances (`A` and `B`), each going first twice.
Each unordered pairing is played once; H2 vs H1 is read from the same games as H1 vs H2.

`win_rate[row][col] = wins of row heuristic against col heuristic / games they played`.
Draws are counted separately and are not wins.

- Off-diagonal: the 4 games between two distinct heuristics.
- Diagonal (self-play): each game is counted once from each instance's perspective, so
  the denominator is 8 (2 × 4 games), a decisive game adds one win, and a draw adds two
  draws. A self-play matchup with no draws therefore always shows 50%.

The agents are deterministic, so repeated games with the same configuration play
identically. Within a pairing, the two games with the same first player are therefore
the same game; this is expected.

## Results

Written to `results/` (created if missing, overwritten each run, never appended):

- `win_rate_matrix.csv` – win rates in [0, 1], rows = evaluated agent, columns = opponent
- `win_count_matrix.csv`, `draw_count_matrix.csv` – raw counts (same convention)
- `game_results.csv` – one row per game: game number, first/second heuristic, instance,
  winner, moves, duration, average decision time and evaluated states per agent
- `tournament_summary.json` – configuration (board, depth, games), matrices, and per
  heuristic wins/losses/draws, win rate, decision time and nodes evaluated

"Nodes evaluated" counts terminal states plus heuristic evaluations during search.

## Adding later experiments

- **Search depths (4, 7, 10):** `Agent(heuristic_id, depth)` already takes a depth, and
  `TournamentConfig(depth=...)` sets it for a tournament. A new module in
  `experiments/` can create agents with different depths per side.
- **Board size (e.g. 10 × 11):** `Board`, `Game`, the heuristics, and `TournamentConfig`
  take `rows`/`cols` (defaults come from `config.py`); no code assumes 6 × 7.
- **Timing comparisons:** per-decision time and node counts are already recorded in
  `Agent.stats` and the result files.
