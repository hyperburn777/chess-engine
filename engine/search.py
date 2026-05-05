import time
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
        lookup_hash = (chess.polyglot.zobrist_hash(board), depth)
        if lookup_hash in self.lookup:
            return self.lookup[lookup_hash]
        
        if board.is_game_over() or depth == 0:
            # Since this is already POV, we return it directly
            best_score, best_move = self._get_evaluation(board), None
        else:
            moves = list(board.legal_moves)

            best_move = moves[0]
            best_score = -self.INF
            
            for move in moves:
                board.push(move)

                board_hash = chess.polyglot.zobrist_hash(board)
                if board_hash in self.move_cache:
                    score = 0
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
        
        self.lookup[lookup_hash] = (best_score, best_move)
        return best_score, best_move

    def register_board(self, board):
        z_key = chess.polyglot.zobrist_hash(board)
        self.move_cache.add(z_key)

    def clear_cache(self):
        self.move_cache.clear()

    def find_best_move(self, board, depth=3):
        _, move = self.negmax(board, depth, -self.INF, self.INF)
        return move
    
    def find_best_move_tl(self, board, limit):
        depth = 1
        best_move = None
        start = time.time()
        while time.time() - start < limit:
            best_move = self.find_best_move(board, depth)
            depth += 1
        
        return best_move