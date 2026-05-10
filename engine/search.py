import time
import torch

import chess
import chess.polyglot

from engine.eval import evaluate
from engine.accumulator import NNUEAccumulator

_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

class ChessSearch:
    INF = 999999
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model=None):
        self.model = model
        self.move_cache = set()
        self.lookup = dict()
        self.best_moves = dict()  # zobrist_hash -> best move found at this position (any depth)
        self.accumulator = NNUEAccumulator(model) if model is not None else None

    def _get_evaluation(self, board):
        if chess.polyglot.zobrist_hash(board) in self.move_cache:
            return 0

        if self.model:
            if board.is_checkmate():
                return -999999
            dev = self.accumulator.model_device
            if board.turn == chess.WHITE:
                stm_acc = self.accumulator.white.to(dev)
                nstm_acc = self.accumulator.black.to(dev)
            else:
                stm_acc = self.accumulator.black.to(dev)
                nstm_acc = self.accumulator.white.to(dev)
            return self.model.evaluate_acc(stm_acc, nstm_acc)

        return evaluate(board)

    def _order_moves(self, board, moves):
        z_hash = chess.polyglot.zobrist_hash(board)
        tt_move = self.best_moves.get(z_hash)

        def score(move):
            if move == tt_move:
                return 30000  # always try the previously-found best move first
            s = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # high-value victim captured by low-value attacker = good
                    s = 10 * _PIECE_VALUES[victim.piece_type] - _PIECE_VALUES[attacker.piece_type]
                s += 10000  # all captures rank above quiet moves
            if move.promotion:
                s += _PIECE_VALUES.get(move.promotion, 0)
            return s

        return sorted(moves, key=score, reverse=True)

    def negmax(self, board, depth, alpha, beta):
        lookup_hash = (chess.polyglot.zobrist_hash(board), depth)
        if lookup_hash in self.lookup:
            return self.lookup[lookup_hash]

        if board.is_game_over() or depth == 0:
            # Since this is already POV, we return it directly
            best_score, best_move = self._get_evaluation(board), None
        else:
            moves = self._order_moves(board, list(board.legal_moves))

            best_move = moves[0]
            best_score = -self.INF

            z_hash_before = chess.polyglot.zobrist_hash(board)
            for move in moves:
                if self.accumulator is not None:
                    self.accumulator.push(board, move)
                else:
                    board.push(move)

                board_hash = chess.polyglot.zobrist_hash(board)
                if board_hash in self.move_cache:
                    score = 0
                else:
                    # Recursive call: negate the result and swap alpha/beta
                    score, _ = self.negmax(board, depth - 1, -beta, -alpha)

                score = -score

                if self.accumulator is not None:
                    self.accumulator.pop(board)
                else:
                    board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move
                    self.best_moves[z_hash_before] = move

                alpha = max(alpha, score)
                if alpha >= beta:
                    break

        self.lookup[lookup_hash] = (best_score, best_move)
        return best_score, best_move

    def register_board(self, board):
        z_key = chess.polyglot.zobrist_hash(board)
        self.move_cache.add(z_key)
        if self.accumulator is not None:
            self.accumulator.init_from_board(board)

    def clear_cache(self):
        self.move_cache.clear()

    def find_best_move(self, board, depth=3):
        if self.accumulator is not None:
            self.accumulator.init_from_board(board)
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