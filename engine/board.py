import chess

class ChessBoard:
    def __init__(self):
        self.board = chess.Board()

    def legal_moves(self):
        return list(self.board.legal_moves)

    def push(self, move):
        self.board.push(move)

    def pop(self):
        self.board.pop()

    def is_game_over(self):
        return self.board.is_game_over()

    def turn(self):
        return self.board.turn

    def fen(self):
        return self.board.fen()

    def result(self):
        return self.board.result()