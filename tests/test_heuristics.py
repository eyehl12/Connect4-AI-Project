import unittest

from agents.heuristics import (HEURISTICS, connection_scores, multiple_winning_moves,
                               opponent_awareness, weighted_connections, winning_moves)
from game.board import Board
from tests.helpers import play


def grid_from(rows):
    """Build a board from strings ('.', 'X' = 1, 'O' = 2), top row first."""
    mapping = {".": 0, "X": 1, "O": 2}
    return Board.from_grid([[mapping[ch] for ch in row] for row in rows])


class ConnectionTests(unittest.TestCase):
    def test_open_pair_scores_two(self):
        board = play([2, 6, 3])  # X on bottom row cols 2,3
        self.assertEqual(connection_scores(board), (0, 2, 0))

    def test_open_triple_scores_five_once(self):
        board = play([1, 6, 2, 6, 3])
        self.assertEqual(connection_scores(board)[1], 5)

    def test_vertical_run(self):
        board = play([0, 1, 0, 1])
        self.assertEqual(connection_scores(board)[1], 2)  # vertical pair, room above
        self.assertEqual(connection_scores(board)[2], 2)

    def test_single_pieces_score_nothing(self):
        self.assertEqual(connection_scores(play([0, 6])), (0, 0, 0))

    def test_pair_blocked_both_ends_not_counted(self):
        board = grid_from([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "OXXO...",
        ])
        self.assertEqual(connection_scores(board)[1], 0)

    def test_pair_boxed_by_edge_and_opponent_not_counted(self):
        board = grid_from([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "XXO....",
        ])
        self.assertEqual(connection_scores(board)[1], 0)

    def test_pair_with_one_blocked_end_still_viable(self):
        board = grid_from([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "OXX....",
        ])
        self.assertEqual(connection_scores(board)[1], 2)

    def test_diagonal_run_counts(self):
        board = play([0, 1, 1, 2, 3, 2, 2])  # X at (5,0),(4,1),(3,2)
        self.assertEqual(connection_scores(board)[1], 5)

    def test_diagonal_blocked_by_edge(self):
        # X pair on the diagonal ending at the top row, opponent below: no room for four.
        board = grid_from([
            "..X....",
            ".X.....",
            "O......",
            ".......",
            ".......",
            ".......",
        ])
        self.assertEqual(connection_scores(board)[1], 0)


class HeuristicTests(unittest.TestCase):
    def test_h1_only_counts_own(self):
        board = play([2, 5, 3, 5])  # X pair, O vertical pair
        self.assertEqual(weighted_connections(board, 1), 2)
        self.assertEqual(weighted_connections(board, 2), 2)

    def test_h1_ignores_opponent_threats(self):
        board = play([2, 5, 3, 5, 6])  # only O has a 2 on col 5; X has a pair
        self.assertEqual(weighted_connections(board, 1), 2)

    def test_h2_subtracts_opponent(self):
        board = play([2, 5, 3, 5, 6, 5])  # O has vertical triple (5), X has a pair
        self.assertEqual(opponent_awareness(board, 1), 2 - 5)
        self.assertEqual(opponent_awareness(board, 2), 5 - 2)

    def test_h2_is_zero_sum(self):
        board = play([3, 3, 2, 4, 1])
        self.assertEqual(opponent_awareness(board, 1), -opponent_awareness(board, 2))

    def test_heuristics_registered(self):
        self.assertEqual(set(HEURISTICS),
                         {"weighted_connections", "opponent_awareness", "multiple_winning_moves"})


class MultipleWinningMovesTests(unittest.TestCase):
    def test_only_legal_cells_counted(self):
        supported = grid_from([
            ".......",
            ".......",
            ".......",
            ".......",
            "XXX....",
            "OXOO...",
        ])
        # Row 4, col 3 sits on an occupied cell, so it is playable.
        self.assertEqual(winning_moves(supported, 1), [3])
        floating = grid_from([
            ".......",
            ".......",
            ".......",
            ".......",
            "XXX....",
            "OOO....",
        ])
        # Row 4, col 3 would win, but row 5, col 3 is empty so it is not playable yet.
        self.assertEqual(winning_moves(floating, 1), [])

    def test_double_threat_for_non_mover_scores_high(self):
        board = play([2, 6, 3, 6, 4])  # O to move, X threatens cols 1 and 5
        self.assertEqual(sorted(winning_moves(board, 1)), [1, 5])
        self.assertGreater(multiple_winning_moves(board, 1), 10_000)
        self.assertLess(multiple_winning_moves(board, 2), -10_000)

    def test_single_threat_for_non_mover_is_not_decisive(self):
        board = play([0, 6, 1, 6, 2])  # O to move; X threatens only col 3
        self.assertEqual(winning_moves(board, 1), [3])
        self.assertEqual(multiple_winning_moves(board, 1), opponent_awareness(board, 1))

    def test_mover_with_winning_move_is_rewarded(self):
        board = play([0, 6, 1, 6, 2, 5])  # X to move and can win at col 3
        self.assertGreater(multiple_winning_moves(board, 1), 10_000)

    def test_opponent_immediate_win_beats_my_double_threat(self):
        # X threatens cols 0 and 4, but O is on move and wins immediately in col 6.
        board = grid_from([
            ".......",
            ".......",
            ".......",
            "......O",
            "......O",
            ".XXX.XO",
        ])
        self.assertEqual(board.current_player, 2)
        self.assertEqual(sorted(winning_moves(board, 1)), [0, 4])
        self.assertEqual(winning_moves(board, 2), [6])
        self.assertLess(multiple_winning_moves(board, 1), 0)
        self.assertGreater(multiple_winning_moves(board, 2), 10_000)


if __name__ == "__main__":
    unittest.main()
