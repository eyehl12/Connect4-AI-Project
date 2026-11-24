import unittest
from unittest.mock import patch

from agents.agent import Agent
from agents.heuristics import opponent_awareness, weighted_connections
from agents.minimax import Minimax
from config import WIN_SCORE
from game.board import Board
from tests.helpers import play


def reference_minimax(board, depth, me, ply=0):
    """Plain Minimax without pruning, used to validate alpha-beta."""
    if board.winner:
        return WIN_SCORE - ply if board.winner == me else ply - WIN_SCORE
    if board.is_full():
        return 0
    if depth == 0:
        return weighted_connections(board, me) - weighted_connections(board, 3 - me)
    scores = []
    for col in board.legal_moves():
        board.make_move(col)
        scores.append(reference_minimax(board, depth - 1, me, ply + 1))
        board.undo_move()
    return max(scores) if board.current_player == me else min(scores)


def eval_diff(board, me):
    return weighted_connections(board, me) - weighted_connections(board, 3 - me)


class MinimaxTests(unittest.TestCase):
    def test_takes_immediate_win(self):
        board = play([0, 6, 1, 6, 2, 6])  # X to move, wins at column 3
        result = Minimax(opponent_awareness, 3).search(board)
        self.assertEqual(result.move, 3)
        self.assertEqual(result.score, WIN_SCORE - 1)

    def test_blocks_immediate_loss(self):
        board = play([0, 0, 1, 1, 2])  # O to move, X threatens column 3
        self.assertEqual(Minimax(opponent_awareness, 3).search(board).move, 3)

    def test_second_player_perspective(self):
        board = play([0, 1, 0, 1, 2, 1, 3])  # O to move, wins vertically at column 1
        result = Minimax(opponent_awareness, 2).search(board)
        self.assertEqual(result.move, 1)
        self.assertEqual(result.score, WIN_SCORE - 1)

    def test_unavoidable_loss_scored_negative_with_ply(self):
        board = play([2, 6, 3, 6, 4])  # O to move; X has a double threat
        result = Minimax(opponent_awareness, 3).search(board)
        self.assertEqual(result.score, -(WIN_SCORE - 2))

    def test_alpha_beta_matches_plain_minimax(self):
        positions = [[], [3], [3, 3, 2, 4], [0, 1, 2, 3, 3, 2, 1], [3, 4, 3, 4, 3]]
        for moves in positions:
            for depth in (1, 2, 3, 4):
                board = play(moves)
                me = board.current_player
                scores = []
                for col in board.legal_moves():
                    board.make_move(col)
                    score = reference_minimax(board, depth - 1, me, 1)
                    board.undo_move()
                    scores.append((col, score))
                result = Minimax(eval_diff, depth).search(board)
                best_score = max(score for _, score in scores)
                self.assertEqual(result.score, best_score, (moves, depth))
                self.assertIn(result.move, [col for col, score in scores if score == best_score])

    def test_search_restores_board_and_counts_nodes(self):
        board = play([3, 3, 4])
        before = ([row[:] for row in board.grid], board.current_player, board.move_count)
        result = Minimax(opponent_awareness, 3).search(board)
        self.assertEqual(before, ([row[:] for row in board.grid], board.current_player, board.move_count))
        self.assertGreater(result.nodes_evaluated, 0)
        self.assertGreaterEqual(result.elapsed, 0)

    def test_tied_best_moves_are_selected_randomly(self):
        board = Board()
        with patch("agents.minimax.random.choice", side_effect=lambda moves: moves[-1]) as choose:
            result = Minimax(weighted_connections, 1).search(board)
        self.assertGreater(len(choose.call_args.args[0]), 1)
        self.assertEqual(result.move, choose.call_args.args[0][-1])

    def test_unique_best_move_does_not_use_random_tie_breaking(self):
        board = play([0, 6, 1, 6, 2, 6])
        with patch("agents.minimax.random.choice") as choose:
            result = Minimax(opponent_awareness, 3).search(board)
        self.assertEqual(result.move, 3)
        choose.assert_not_called()

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            Minimax(weighted_connections, 0)


class AgentTests(unittest.TestCase):
    def test_agent_tracks_stats(self):
        agent = Agent("opponent_awareness", depth=2)
        board = play([3])
        move = agent.choose_move(board)
        self.assertTrue(board.is_valid_move(move))
        self.assertEqual(agent.stats.decisions, 1)
        self.assertGreater(agent.stats.nodes_evaluated, 0)
        agent.reset_stats()
        self.assertEqual(agent.stats.decisions, 0)

    def test_default_depth_and_unknown_heuristic(self):
        self.assertEqual(Agent("weighted_connections").depth, 7)
        with self.assertRaises(ValueError):
            Agent("nope")


if __name__ == "__main__":
    unittest.main()
