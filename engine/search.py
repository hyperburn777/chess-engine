import chess
import chess.polyglot

from engine.eval import evaluate

class ChessSearch:
    INF = 999999

    def __init__(self, model=None):
        self.model = model
        self.move_cache = set()

    def _get_evaluation(self, board):
        if chess.polyglot.zobrist_hash(board) in self.move_cache:
            return 0

        if self.model:
            return self.model.evaluate(board)

        return evaluate(board)

    def minimax(self, board, depth, alpha, beta, maximizing):

        if depth == 0 or board.is_game_over():
            return self._get_evaluation(board), None

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

    def register_board(self, board):
        z_key = chess.polyglot.zobrist_hash(board)
        self.move_cache.add(z_key)

    def clear_cache(self):
        self.move_cache.clear()

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