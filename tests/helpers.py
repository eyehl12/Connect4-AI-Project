"""Shared helpers for tests."""

from game.board import Board


def play(moves, rows=6, cols=7):
    board = Board(rows, cols)
    for col in moves:
        board.make_move(col)
    return board
