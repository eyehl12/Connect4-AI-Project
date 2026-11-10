import unittest

from game.board import Board, InvalidMoveError
from tests.helpers import play

DRAW_GRID = [[((c // 2) + r) % 2 + 1 for c in range(7)] for r in range(6)]


class BoardTests(unittest.TestCase):
    def test_piece_falls_to_lowest_row(self):
        board = Board()
        self.assertEqual(board.make_move(3), 5)
        self.assertEqual(board.make_move(3), 4)
        self.assertEqual(board.grid[5][3], 1)
        self.assertEqual(board.grid[4][3], 2)

    def test_full_column_rejects_moves(self):
        board = play([0, 0, 0, 0, 0, 0])
        self.assertFalse(board.is_valid_move(0))
        self.assertNotIn(0, board.legal_moves())
        with self.assertRaises(InvalidMoveError):
            board.make_move(0)

    def test_out_of_range_rejected(self):
        for col in (-1, 7):
            with self.assertRaises(InvalidMoveError):
                Board().make_move(col)

    def test_no_moves_after_win(self):
        board = play([0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(board.winner, 1)
        self.assertEqual(board.legal_moves(), [])
        with self.assertRaises(InvalidMoveError):
            board.make_move(2)

    def test_horizontal_win(self):
        self.assertEqual(play([0, 0, 1, 1, 2, 2, 3]).winner, 1)

    def test_vertical_win(self):
        self.assertEqual(play([0, 1, 0, 1, 0, 1, 0]).winner, 1)

    def test_diagonal_wins(self):
        self.assertEqual(play([0, 1, 1, 2, 3, 2, 2, 3, 6, 3, 3]).winner, 1)
        self.assertEqual(play([6, 5, 5, 4, 3, 4, 4, 3, 0, 3, 3]).winner, 1)

    def test_second_player_win(self):
        self.assertEqual(play([0, 1, 0, 1, 2, 1, 3, 1]).winner, 2)

    def test_draw_on_full_board(self):
        board = Board.from_grid(DRAW_GRID)
        self.assertEqual(board.winner, 0)
        self.assertTrue(board.is_full())
        self.assertTrue(board.is_draw())
        self.assertTrue(board.is_terminal())
        self.assertEqual(board.legal_moves(), [])

    def test_undo_restores_board(self):
        board = play([3, 3, 2])
        snapshot = ([row[:] for row in board.grid], board.heights[:], board.current_player,
                    board.move_count)
        board.make_move(4)
        board.undo_move()
        self.assertEqual(snapshot, ([row[:] for row in board.grid], board.heights, board.current_player,
                                    board.move_count))

    def test_undo_clears_winner(self):
        board = play([0, 1, 0, 1, 0, 1, 0])
        board.undo_move()
        self.assertEqual(board.winner, 0)
        self.assertEqual(board.current_player, 1)
        self.assertTrue(board.is_valid_move(0))

    def test_center_first_ordering(self):
        self.assertEqual(Board().legal_moves(), [3, 2, 4, 1, 5, 0, 6])

    def test_is_winning_move_does_not_mutate(self):
        board = play([0, 6, 1, 6, 2, 6])
        before = [row[:] for row in board.grid]
        self.assertTrue(board.is_winning_move(3, 1))
        self.assertFalse(board.is_winning_move(4, 1))
        self.assertEqual(before, board.grid)

    def test_str_shows_pieces(self):
        text = str(play([3, 4]))
        self.assertIn("X", text)
        self.assertIn("O", text)


if __name__ == "__main__":
    unittest.main()
