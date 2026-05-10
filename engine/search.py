import copy
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

# Transposition-table value flags. EXACT means the stored value is the true
# minimax result for that position at that depth. LOWER means a beta-cutoff
# happened, so the true value is >= stored value. UPPER means we never beat
# alpha, so the true value is <= stored value. Without these flags, cached
# fail-high/fail-low scores get reused as if exact and poison future searches.
_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2

class ChessSearch:
    INF = 999999
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model=None):
        self.model = model
        self.move_cache = set()
        self.lookup = dict()
        self.best_moves = dict()  # zobrist_hash -> best move found at this position (any depth)
        self.accumulator = NNUEAccumulator(model) if model is not None else None

        # Output layer is tiny; running it on CPU avoids MPS/CUDA kernel-launch
        # and .item() sync overhead per leaf, which dominates with quiescence.
        if model is not None:
            self._output_cpu = copy.deepcopy(model.output_layer).to("cpu").eval()
        else:
            self._output_cpu = None

    @torch.no_grad()
    def _get_evaluation(self, board):
        if self.model:
            # Accumulators already live on CPU; output layer was copied to CPU in __init__.
            if board.turn == chess.WHITE:
                stm_acc, nstm_acc = self.accumulator.white, self.accumulator.black
            else:
                stm_acc, nstm_acc = self.accumulator.black, self.accumulator.white
            stm = torch.clamp(stm_acc, 0.0, 1.0).unsqueeze(0)
            nstm = torch.clamp(nstm_acc, 0.0, 1.0).unsqueeze(0)
            combined = torch.cat([stm, nstm], dim=1)
            return self._output_cpu(combined).item()

        return evaluate(board)

    def _capture_score(self, board, move):
        s = 0
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            s = 10 * _PIECE_VALUES[victim.piece_type] - _PIECE_VALUES[attacker.piece_type]
        s += 10000
        if move.promotion:
            s += _PIECE_VALUES.get(move.promotion, 0)
        return s

    def _order_moves(self, board, moves, z_hash):
        tt_entry = self.lookup.get(z_hash)
        tt_move = tt_entry[3] if tt_entry is not None else self.best_moves.get(z_hash)

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

    QSEARCH_MAX_PLY = 6  # safety cap so quiescence can't explode on tactical positions

    def quiescence(self, board, alpha, beta, ply=0):
        # Stand-pat alpha-beta over captures only. Resolves the "horizon effect"
        # so the static eval at search leaves is taken on quiet positions.
        # Captures are irreversible, so we don't need a move_cache (repetition) check here.
        in_check = board.is_check()

        if in_check:
            # Can't trust stand-pat while in check; must consider all evasions.
            moves = list(board.legal_moves)
            if not moves:
                return -self.INF + 1  # checkmated
            # Fail-soft floor; if every evasion is bad we still return the *best
            # bad option*, not the unchanged input alpha (which would over-estimate).
            best_score = -self.INF
        else:
            stand_pat = self._get_evaluation(board)
            if stand_pat >= beta or ply >= self.QSEARCH_MAX_PLY:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            moves = list(board.generate_legal_captures())
            best_score = stand_pat

        moves.sort(key=lambda m: self._capture_score(board, m), reverse=True)

        for move in moves:
            if self.accumulator is not None:
                self.accumulator.push(board, move)
            else:
                board.push(move)

            score = -self.quiescence(board, -beta, -alpha, ply + 1)

            if self.accumulator is not None:
                self.accumulator.pop(board)
            else:
                board.pop()

            if score >= beta:
                return score
            if score > best_score:
                best_score = score
                if score > alpha:
                    alpha = score

        return best_score

    def negmax(self, board, depth, alpha, beta, z_hash=None):
        # Caller may pass a pre-computed hash (from the parent's post-push hash),
        # so we don't re-walk the board to hash it again.
        if z_hash is None:
            z_hash = chess.polyglot.zobrist_hash(board)

        # 50-move rule short-circuit (cheap O(1) check). Repetition is handled
        # in the move loop below — checking it here would falsely flag the root
        # whenever main.py registered the current position before searching.
        if board.halfmove_clock >= 100:
            return 0, None

        alpha_orig = alpha  # remembered for choosing the TT flag at the end

        # Bounded TT lookup: a stored value is only directly usable if its bound
        # type combined with the current (alpha, beta) lets us return without
        # losing information.
        tt_entry = self.lookup.get(z_hash)
        if tt_entry is not None:
            tt_depth, tt_value, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == _TT_EXACT:
                    return tt_value, tt_move
                if tt_flag == _TT_LOWER and tt_value >= beta:
                    return tt_value, tt_move
                if tt_flag == _TT_UPPER and tt_value <= alpha:
                    return tt_value, tt_move

        if depth == 0:
            # Quiescence value depends on the (alpha, beta) window, so don't TT-cache it.
            return self.quiescence(board, alpha, beta), None

        # Generate legal moves once; replaces the expensive board.is_game_over() call.
        moves = list(board.legal_moves)
        if not moves:
            score = -self.INF + 1 if board.is_check() else 0
            # Terminal score is exact and depth-independent — store at huge depth so any future query reuses it.
            self.lookup[z_hash] = (10_000, score, _TT_EXACT, None)
            return score, None

        moves = self._order_moves(board, moves, z_hash)

        best_move = moves[0]
        best_score = -self.INF

        for move in moves:
            if self.accumulator is not None:
                self.accumulator.push(board, move)
            else:
                board.push(move)

            child_hash = chess.polyglot.zobrist_hash(board)
            if child_hash in self.move_cache:
                # Repeating a previously-played position = draw.
                score = 0
            else:
                score, _ = self.negmax(board, depth - 1, -beta, -alpha, z_hash=child_hash)
            score = -score

            if self.accumulator is not None:
                self.accumulator.pop(board)
            else:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                self.best_moves[z_hash] = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        if best_score <= alpha_orig:
            flag = _TT_UPPER  # never improved on input alpha — value is an upper bound
        elif best_score >= beta:
            flag = _TT_LOWER  # beta cutoff — value is a lower bound
        else:
            flag = _TT_EXACT

        self.lookup[z_hash] = (depth, best_score, flag, best_move)
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