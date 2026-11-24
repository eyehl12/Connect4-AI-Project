# Connect Four AI Experiments

This project compares three Connect Four agents. Every agent uses
the same depth-limited Minimax implementation with alpha-beta pruning; only its
evaluation heuristic and, in Experiment 2, its search depth vary.

## Project structure

```
main.py                         CLI for running experiments or a debug game
config.py                       Central board, search, and experiment defaults
game/board.py                   Configurable board, move/undo, and win detection
game/game.py                    Shared game loop for all board dimensions
agents/heuristics.py            Shared connection scoring and three heuristics
agents/minimax.py               Minimax, alpha-beta pruning, and search metrics
agents/agent.py                 Heuristic/depth agent and per-agent statistics
experiments/tournament.py       Tournament scheduling, game play, aggregation
experiments/depth_experiment.py Fixed-baseline search-depth experiment
experiments/board_size_experiment.py Board-size experiment and baseline reuse
experiments/results.py          Experiment 1 CSV, JSON, and heatmap output
experiments/advanced_results.py Experiment 2/3 outputs, comparisons, and plots
tests/                          Unit and reduced-depth integration tests
results/                        Independent results folders for each experiment
```

Install the plotting dependency with `pip install -r requirements.txt`. The game
engine and tests otherwise use the Python standard library.

## Run

```
python main.py --experiment 1
python main.py --experiment 2
python main.py --experiment 3
python main.py --experiment all
python main.py --experiment 2 --depths 4 7 10
python main.py --experiment 1 --games 8
python main.py --experiment 3 --games 6
python main.py play H1 H3
python -m unittest discover -s tests -t .
```

Experiments default to one worker. `--workers N` runs games in parallel. The
`--results-dir PATH` option changes the results root. Experiment 2's `--depths`
option changes its tested depths; depth 7 must remain included because it is the
fixed baseline. The three experiments can be run separately and write only to
their own result directory. An experiment only writes results after all of its
scheduled games finish; the combined summary reports which experiments have
completed.

`--games N` sets the total games per pairing (or per tested depth configuration
in Experiment 2) and defaults to 4. It must be an even number so each side gets
the same number of first-player games. Thus Experiment 1 schedules
`6 pairings × N` games, Experiment 2 schedules
`3 heuristics × 3 depths × N` games, and Experiment 3 schedules
`6 pairings × N` games on each of its two boards. The per-matchup game count is
recorded in each results summary.

## The heuristics

All heuristics use the shared `connection_scores` scan. A viable contiguous run
of two pieces scores 2, a run of three scores 5, and a run of four is handled as
a terminal win by Minimax. A run is viable if enough unblocked geometric space
remains to extend it to four.

| ID | Name | Evaluation |
|---|---|---|
| H1 | `weighted_connections` | Own viable connection score |
| H2 | `opponent_awareness` | Own score minus the opponent's score |
| H3 | `multiple_winning_moves` | H2 plus immediate-win and double-threat detection |

Terminal outcomes take precedence over heuristics. A win at ply `p` scores
`1,000,000 - p`, a loss scores `-(1,000,000 - p)`, and a draw scores zero.
Depth counts plies. Center-first move ordering uses the center column(s) for
any configured board width. If multiple moves have the same best Minimax score,
the agent chooses randomly among those tied moves. A uniquely best move is
selected directly. This lets repeated games explore different equally-scored
choices while preserving the evaluation and search rules.

## Experiment designs

### Experiment 1: heuristic performance

The existing baseline compares all six unordered pairings of H1, H2, and H3,
including self-play. By default, each pairing has four games with the two
instances each playing first twice: 24 games total on a 6 × 7 board at depth 7.
The count can be changed with `--games` (even values only). A matchup is
scheduled once; reversed pairings are not duplicated.

### Experiment 2: search depth

Each heuristic at each tested depth plays four games against its own fixed
depth-7 version. By default the evaluated agent plays first twice and second twice.
Therefore H2-D10 is compared with H2-D7, not a different heuristic. Even the
depth-7 comparison uses two separate agent instances. The defaults are depths
4, 7, and 10 on a 6 × 7 board, for 36 games. `--games N` changes the number of
games for each heuristic/depth configuration and divides them equally by who
moves first.

Per-configuration outcomes and performance metrics are for the evaluated agent,
not the baseline. Win rate is `wins / games`; draws are reported separately and
are not wins. Decision-time averages are measured with `time.perf_counter()`.
Node counts are terminal positions plus heuristic evaluations at the search
horizon, reported per decision and as a total.

### Experiment 3: board size

The same six pairings and four balanced games per pairing run by default on 6 × 7
and 10 × 11 boards, at depth 7 with a four-in-a-row winning condition: 48 games
overall. `--games N` changes the count on each board. Both use the same `Board`,
game loop, heuristics, and Minimax code.
The standard-board results are reused only when the saved Experiment 1
configuration and raw 24-game results match the expected setup; otherwise the
standard tournament is regenerated. The summary records the source.

Experiments 1 and 3 write a single `win_draw_rate_matrix.csv`: each win is 1,
each draw is 0.5, and each loss is 0, divided by games played. The matching
heatmap uses the same combined score. Raw win and draw counts and separate
rates remain available in the JSON summaries. Self-play matrix cells count each
side's perspective, so diagonal denominators are twice the number of self-play
games. Cross-heuristic comparison tables exclude self-play so it cannot obscure
relative performance. For distinct heuristics, each row's denominator is the
number of games between those heuristics; the two combined scores sum to 1.

## Results

```
results/
├── experiment_1/                 # Combined win-draw matrix, raw games, summary
├── experiment_2/                 # Four CSV matrices, raw games, summary, plots/
├── experiment_3/
│   ├── standard/                 # 6 × 7 combined matrix, raw games, summary
│   ├── expanded/                 # 10 × 11 combined matrix, raw games, summary
│   ├── comparison.csv            # Cross-heuristic and overall comparison metrics
│   └── plots/                    # Paired heatmaps and board-size performance charts
└── combined_summary.json         # Completion state and configuration per experiment
```

Experiment 2 plots show win rate, average decision time, and evaluated nodes
against search depth, with one series per heuristic. Experiment 3 includes
side-by-side combined win-draw-rate heatmaps using a matching 0–1 color scale,
and bar charts comparing average decision time and evaluated nodes by board size.
Tournament summaries include raw counts, rate matrices, configuration, node
count definition, and timing statistics. CSV files are designed for Excel or
pandas. Re-running an experiment overwrites only its own folder.

The original top-level result files are retained as historical artifacts.
Their saved metadata identifies the 6 × 7, depth-7, 24-game Experiment 1 setup;
they are copied into `results/experiment_1/` without changing the source files.
Those historical outputs predate random tie-breaking and retain their original
separate rate matrices; they are not reused as a current Experiment 3 baseline.
After rerunning Experiment 1, its result folder is updated with the combined
matrix and the new tie-breaking metadata, allowing Experiment 3 to reuse it.
When no expanded-board or depth experiment has been run, its completion state
remains false in the combined summary rather than implying partial results are
complete.

## Search and measurements

Minimax updates the board in place and undoes each move rather than copying the
board at every node. It checks wins incrementally after each move and searches
center-first with alpha-beta pruning. No transposition table is used, avoiding
cross-heuristic cache contamination and preserving the depth-sensitive terminal
score semantics. Agent decision time is accumulated around each search; nodes
count terminal states and horizon evaluations, not all internal visited nodes.

The tests cover randomized selection among tied best moves, unique-best move
selection, both board sizes, arbitrary-width
move ordering, the tournament schedules, fixed-baseline grouping, result
formats, and plot generation. Integration tests use shallow depths; full depth-10
or expanded-board runs are not part of the test suite.
