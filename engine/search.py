import chess
from engine.eval import evaluate

class ChessSearch:
    INF = 999999

    def __init__(self, model=None):
        self.model = model

    def _get_evaluation(self, board):
        if self.model:
            return self.model.evaluate(board)

        return evaluate(board)

    def minimax(self, board, depth, alpha, beta, maximizing):

        if depth == 0 or board.is_game_over():
            return evaluate(board), None

        best_move = None

        moves = list(board.legal_moves)

        if maximizing:
            best_score = -self.INF

            for move in moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            return best_score, best_move

        else:
            best_score = self.INF

            for move in moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, score)
                if beta <= alpha:
                    break

            return best_score, best_move


    def find_best_move(self, board, depth=3):
        maximizing = board.turn == chess.WHITE
        _, move = self.minimax(board, depth, -self.INF, self.INF, maximizing)
        return move