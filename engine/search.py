import torch

import chess
import chess.polyglot

from engine.eval import evaluate, model_evaluate_board

class ChessSearch:
    INF = 999999
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model=None):
        self.model = model
        self.move_cache = set()
        self.lookup = dict()

    def _get_evaluation(self, board):
        if chess.polyglot.zobrist_hash(board) in self.move_cache:
            return 0
        
        if self.model:
            # return self.model.evaluate(board)
            return model_evaluate_board(self.model, board, self.device)

        return evaluate(board)

    def minimax(self, board, depth, alpha, beta, maximizing):

        if depth == 0 or board.is_game_over():
            return self._get_evaluation(board), None

        best_move = None

        moves = list(board.legal_moves)

        if maximizing:
            best_score = -self.INF

            for move in moves:
                board.push(move)
                hash = chess.polyglot.zobrist_hash(board)
                if board.is_checkmate():
                    score = 2
                elif hash in self.move_cache:
                    score = 0
                elif hash in self.lookup:
                    score = self.lookup[hash]
                else: 
                    score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                self.lookup[hash] = score
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
                hash = chess.polyglot.zobrist_hash(board)
                if board.is_checkmate():
                    score = -2
                elif hash in self.move_cache:
                    score = 0
                elif hash in self.lookup:
                    score = self.lookup[hash]
                else: 
                    score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                self.lookup[hash] = score
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
        self.lookup.clear()

    def find_best_move(self, board, depth=3):
        maximizing = board.turn == chess.WHITE
        _, move = self.minimax(board, depth, -self.INF, self.INF, maximizing)
        return move