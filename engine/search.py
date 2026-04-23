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

    def negmax(self, board, depth, alpha, beta):
        if board.is_game_over() or depth == 0:
            # Since this is already POV, we return it directly
            return self._get_evaluation(board), None

        best_move = None
        best_score = -self.INF
        
        for move in list(board.legal_moves):
            board.push(move)

            board_hash = chess.polyglot.zobrist_hash(board)
            if board.is_checkmate():
                score = -2
            elif board_hash in self.move_cache:
                score = 0
            elif board_hash in self.lookup:
                score = self.lookup[board_hash]
            else:
                # Recursive call: negate the result and swap alpha/beta
                score, _ = self.negmax(board, depth - 1, -beta, -alpha)
            
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break
        
        board_hash = chess.polyglot.zobrist_hash(board)
        self.lookup[board_hash] = best_score
        return best_score, best_move

    def register_board(self, board):
        z_key = chess.polyglot.zobrist_hash(board)
        self.move_cache.add(z_key)

    def clear_cache(self):
        self.move_cache.clear()
        self.lookup.clear()

    def find_best_move(self, board, depth=3):
        _, move = self.negmax(board, depth, -self.INF, self.INF)
        return move