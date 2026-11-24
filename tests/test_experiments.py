import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from agents.heuristics import HEURISTICS, multiple_winning_moves, opponent_awareness
from agents.minimax import Minimax
from experiments.advanced_results import save_board_size_results, save_depth_results
from experiments.board_size_experiment import (
    BoardSizeExperimentConfig,
    run_board_size_experiment,
)
from experiments.depth_experiment import (
    DepthExperimentConfig,
    build_depth_schedule,
    run_depth_experiment,
)
from game.board import Board
from tests.helpers import play


NAMES = tuple(HEURISTICS)


class DepthExperimentTests(unittest.TestCase):
    def test_default_schedule_has_36_games_and_balanced_sides(self):
        config = DepthExperimentConfig()
        schedule = build_depth_schedule(config)
        self.assertEqual(len(schedule), 36)
        self.assertEqual(config.baseline_depth, 7)
        self.assertEqual(set(config.depths), {4, 7, 10})
        counts = Counter((item.heuristic, item.evaluated_depth) for item in schedule)
        self.assertEqual(set(counts.values()), {4})
        self.assertTrue(all(item.heuristic in NAMES for item in schedule))
        self.assertTrue(all(item.evaluated_depth in config.depths for item in schedule))
        for heuristic in NAMES:
            for depth in config.depths:
                sides = Counter(
                    item.evaluated_side for item in schedule
                    if item.heuristic == heuristic and item.evaluated_depth == depth
                )
                self.assertEqual(sides, {"first": 2, "second": 2})

    def test_run_records_configuration_statistics_and_isolates_outputs(self):
        config = DepthExperimentConfig(
            heuristics=(NAMES[0],), depths=(1, 2), baseline_depth=2, workers=1,
        )
        result = run_depth_experiment(config)
        self.assertEqual(len(result.records), 8)
        self.assertTrue(result.completed)
        self.assertTrue(all(record.baseline_depth == 2 for record in result.records))
        self.assertTrue(all(record.heuristic == NAMES[0] for record in result.records))
        self.assertEqual(result.stats[f"{NAMES[0]}-D1"]["games"], 4)
        stats = result.stats[f"{NAMES[0]}-D1"]
        self.assertEqual(stats["win_rate"], stats["wins"] / 4)
        self.assertEqual(stats["draw_rate"], stats["draws"] / 4)
        self.assertGreater(stats["average_decision_time"], 0)
        self.assertGreater(stats["average_nodes_per_decision"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp1 = root / "experiment_1"
            exp3 = root / "experiment_3"
            exp1.mkdir()
            exp3.mkdir()
            (exp1 / "keep.txt").write_text("one")
            (exp3 / "keep.txt").write_text("three")
            out = save_depth_results(result, root)
            self.assertEqual((exp1 / "keep.txt").read_text(), "one")
            self.assertEqual((exp3 / "keep.txt").read_text(), "three")
            with (out / "game_results.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.reader(handle))), 9)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["games_completed"], 8)
            self.assertIn("fixed depth-2 baseline", summary["baseline_note"])
            for name in (
                "win_rate_vs_depth.png", "decision_time_vs_depth.png", "nodes_vs_depth.png",
            ):
                self.assertGreater((out / "plots" / name).stat().st_size, 0)


class ExpandedBoardTests(unittest.TestCase):
    def test_board_dimensions_moves_wins_and_draw_detection(self):
        for rows, cols in ((6, 7), (10, 11)):
            with self.subTest(board=(rows, cols)):
                board = Board(rows, cols)
                self.assertEqual((board.rows, board.cols), (rows, cols))
                self.assertEqual(board.make_move(cols - 1), rows - 1)
                self.assertEqual(board.make_move(cols - 1), rows - 2)
                self.assertEqual(board.column_order[0], (cols - 1) // 2)
                self.assertEqual(len(board.legal_moves()), cols)

                self.assertEqual(play([0, cols - 1, 1, cols - 1, 2, cols - 2, 3],
                                     rows, cols).winner, 1)
                self.assertEqual(play([0, 1, 0, 1, 0, 1, 0], rows, cols).winner, 1)
                self.assertEqual(
                    play([0, 1, 1, 2, 3, 2, 2, 3, cols - 1, 3, 3],
                         rows, cols).winner,
                    1,
                )
                self.assertEqual(
                    play([cols - 1, cols - 2, cols - 2, cols - 3, cols - 4,
                          cols - 3, cols - 3, cols - 4, 0, cols - 4, cols - 4],
                         rows, cols).winner,
                    1,
                )

                draw_grid = [
                    [((col // 2) + row) % 2 + 1 for col in range(cols)]
                    for row in range(rows)
                ]
                drawn = Board.from_grid(draw_grid)
                self.assertTrue(drawn.is_full())
                self.assertTrue(drawn.is_draw())

    def test_heuristics_and_minimax_support_board_edges_and_eleven_columns(self):
        board = play([8, 10, 9], rows=10, cols=11)
        self.assertIsInstance(opponent_awareness(board, board.current_player), (int, float))
        self.assertIsInstance(multiple_winning_moves(board, board.current_player), (int, float))
        result = Minimax(opponent_awareness, depth=1).search(Board(10, 11))
        self.assertIn(result.move, range(11))

    def test_board_size_experiment_runs_and_saves_both_boards(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = BoardSizeExperimentConfig(depth=1, workers=1)
            result = run_board_size_experiment(Path(tmp), config)
            self.assertEqual([len(item.records) for item in result.results], [24, 24])
            self.assertEqual(result.standard_source, "regenerated")
            out = save_board_size_results(result, Path(tmp))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["games_completed"], 48)
            with (out / "comparison.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual(sum(row["heuristic"] == "overall" for row in rows), 2)
            self.assertTrue(all(row["games"] == "8" for row in rows if row["heuristic"] != "overall"))
            for folder in ("standard", "expanded"):
                self.assertTrue((out / folder / "game_results.csv").exists())
                self.assertTrue((out / folder / "win_draw_rate_matrix.csv").exists())
                self.assertFalse((out / folder / "win_rate_matrix.csv").exists())
                self.assertFalse((out / folder / "draw_rate_matrix.csv").exists())
            for filename in (
                "win_draw_rate_heatmaps.png",
                "decision_time_by_board_size.png", "nodes_by_board_size.png",
            ):
                self.assertGreater((out / "plots" / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
