import unittest

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
                best = None
                for col in board.legal_moves():
                    board.make_move(col)
                    score = reference_minimax(board, depth - 1, me, 1)
                    board.undo_move()
                    if best is None or score > best[1]:
                        best = (col, score)
                result = Minimax(eval_diff, depth).search(board)
                self.assertEqual((result.move, result.score), best, (moves, depth))

    def test_search_restores_board_and_counts_nodes(self):
        board = play([3, 3, 4])
        before = ([row[:] for row in board.grid], board.current_player, board.move_count)
        result = Minimax(opponent_awareness, 3).search(board)
        self.assertEqual(before, ([row[:] for row in board.grid], board.current_player, board.move_count))
        self.assertGreater(result.nodes_evaluated, 0)
        self.assertGreaterEqual(result.elapsed, 0)

    def test_deterministic_tie_break_opens_in_center(self):
        first = Minimax(weighted_connections, 2).search(Board())
        second = Minimax(weighted_connections, 2).search(Board())
        self.assertEqual((first.move, first.score, first.nodes_evaluated),
                         (second.move, second.score, second.nodes_evaluated))
        self.assertEqual(first.move, 3)

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
