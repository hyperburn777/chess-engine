import chess
from engine.eval import evaluate

class ChessSearch:
    INF = 999999

    def __init__(self, model=None):
        self.model = model

    def minimax(self, board, depth, alpha, beta, maximizing, acc=None):

        if depth == 0 or board.is_game_over():

            if self.model is None:
                return evaluate(board), None

            stm = 1.0 if board.turn == chess.WHITE else 0.0
            return self.model.evaluate_acc(acc, stm), None

        best_move = None
        moves = list(board.legal_moves)

        if maximizing:
            best_score = -self.INF

            for move in moves:
                if self.model is None:
                    board.push(move)
                    score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                else:
                    new_acc = self.model.update_accumulator(acc, board, move)
                    board.push(move)
                    score, _ = self.minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                        new_acc
                    )

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

                if self.model is None:
                    board.push(move)
                    score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                else:
                    new_acc = self.model.update_accumulator(acc, board, move)
                    board.push(move)
                    score, _ = self.minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        True,
                        new_acc
                    )

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

        if self.model is None:
            _, move = self.minimax(board, depth, -self.INF, self.INF, maximizing)
            return move

        acc = self.model.init_accumulator(board)

        _, move = self.minimax(
            board,
            depth,
            -self.INF,
            self.INF,
            maximizing,
            acc
        )

        return move