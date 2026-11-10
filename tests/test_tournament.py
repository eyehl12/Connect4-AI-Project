import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.results import OUTPUT_FILES, format_matrix, save_results
from experiments.tournament import (GameRecord, TournamentConfig, build_schedule,
                                    compute_matrices, play_game, run_tournament)

NAMES = ["weighted_connections", "opponent_awareness", "multiple_winning_moves"]
FAST = TournamentConfig(depth=1, workers=1)


def record(number, first, second, winner, instance="A"):
    heuristic = {"first": first, "second": second}.get(winner, "draw")
    return GameRecord(number, first, second, instance, winner, heuristic, 10, 0.5, 0.01, 0.02, 100, 200)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.schedule = build_schedule(NAMES, 2)

    def test_24_games_numbered_sequentially(self):
        self.assertEqual(len(self.schedule), 24)
        self.assertEqual([s.game_number for s in self.schedule], list(range(1, 25)))

    def test_six_unique_pairings_four_games_each(self):
        pairs = Counter(frozenset((s.first_heuristic, s.second_heuristic)) for s in self.schedule)
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(count == 4 for count in pairs.values()))

    def test_each_side_goes_first_twice(self):
        for i, a in enumerate(NAMES):
            for b in NAMES[i + 1:]:
                firsts = Counter(s.first_heuristic for s in self.schedule
                                 if {s.first_heuristic, s.second_heuristic} == {a, b})
                self.assertEqual(firsts, {a: 2, b: 2})

    def test_self_play_instances_each_go_first_twice(self):
        for name in NAMES:
            games = [s for s in self.schedule if s.first_heuristic == s.second_heuristic == name]
            self.assertEqual(len(games), 4)
            self.assertEqual(Counter(s.first_instance for s in games), {"A": 2, "B": 2})

    def test_no_reversed_duplicate_pairing(self):
        ordered = {(s.first_heuristic, s.second_heuristic) for s in self.schedule}
        pairs = {frozenset(p) for p in ordered}
        self.assertEqual(len(pairs), 6)


class MatrixTests(unittest.TestCase):
    def test_distinct_pairing_counts(self):
        a, b = NAMES[0], NAMES[1]
        records = [
            record(1, a, b, "first"),
            record(2, a, b, "draw"),
            record(3, b, a, "first"),
            record(4, b, a, "second"),
        ]
        m = compute_matrices(records, NAMES)
        self.assertEqual(m.wins[0][1], 2)  # game 1, game 4
        self.assertEqual(m.wins[1][0], 1)  # game 3
        self.assertEqual((m.draws[0][1], m.draws[1][0]), (1, 1))
        self.assertEqual(m.games[0][1], 4)
        self.assertAlmostEqual(m.win_rates[0][1], 0.5)
        self.assertAlmostEqual(m.win_rates[1][0], 0.25)

    def test_self_play_combined_perspective(self):
        a = NAMES[0]
        records = [record(1, a, a, "first"), record(2, a, a, "second", "B"),
                   record(3, a, a, "draw"), record(4, a, a, "first", "B")]
        m = compute_matrices(records, NAMES)
        self.assertEqual(m.wins[0][0], 3)
        self.assertEqual(m.draws[0][0], 2)
        self.assertEqual(m.games[0][0], 8)
        self.assertAlmostEqual(m.win_rates[0][0], 3 / 8)

    def test_unplayed_cells_are_zero(self):
        m = compute_matrices([record(1, NAMES[0], NAMES[0], "first")], NAMES)
        self.assertEqual(m.win_rates[1][2], 0.0)


class RealGameTests(unittest.TestCase):
    def test_self_play_game_records_one_win_or_draw(self):
        spec = build_schedule(NAMES[:1], 1)[0]
        rec = play_game(spec, FAST)
        self.assertIn(rec.winner, ("first", "second", "draw"))
        self.assertGreater(rec.moves, 0)
        self.assertGreater(rec.first_nodes, 0)
        m = compute_matrices([rec], NAMES[:1])
        self.assertEqual(m.games[0][0], 2)
        self.assertEqual(m.wins[0][0] + m.draws[0][0], 1 if rec.winner != "draw" else 2)

    def test_full_tournament_has_24_games_and_symmetric_counts(self):
        result = run_tournament(FAST)
        self.assertEqual(len(result.records), 24)
        m = result.matrices
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertEqual(m.games[i][j], 4)
                    self.assertEqual(m.wins[i][j] + m.wins[j][i] + m.draws[i][j], 4)
        for stats in result.stats.values():
            self.assertEqual(stats["wins"] + stats["losses"] + stats["draws"], stats["games"])


class ResultsStorageTests(unittest.TestCase):
    def test_files_created_and_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "results"
            first = run_tournament(FAST)
            save_results(first, out)
            for name in OUTPUT_FILES:
                self.assertTrue((out / name).exists(), name)

            with (out / "game_results.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.reader(handle))), 25)

            small = run_tournament(TournamentConfig(heuristics=NAMES[:2], depth=1, games_per_side=1))
            save_results(small, out)
            with (out / "game_results.csv").open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 1 + len(small.records))  # replaced, not appended
            with (out / "win_rate_matrix.csv").open(newline="") as handle:
                matrix_rows = list(csv.reader(handle))
            self.assertEqual(matrix_rows[0], ["agent", *NAMES[:2]])
            summary = json.loads((out / "tournament_summary.json").read_text())
            self.assertEqual(summary["configuration"]["total_games"], len(small.records))
            self.assertEqual(summary["configuration"]["search_depth"], 1)
            self.assertEqual(summary["configuration"]["board_rows"], 6)
            self.assertIn(NAMES[0], summary["heuristic_statistics"])

    def test_format_matrix(self):
        m = compute_matrices([record(1, NAMES[0], NAMES[1], "first")], NAMES)
        text = format_matrix(m)
        self.assertIn("H1", text)
        self.assertIn("100.0%", text)


if __name__ == "__main__":
    unittest.main()
