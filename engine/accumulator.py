import chess
import torch

from ml.dataset import extract_halfkp, halfkp_index


def _piece_changes(board_before, move):
    # Returns list of (sign, piece, square): sign=+1 piece appears, -1 piece disappears.
    moving_piece = board_before.piece_at(move.from_square)
    changes = [(-1, moving_piece, move.from_square)]

    if move.promotion is not None:
        arrived = chess.Piece(move.promotion, moving_piece.color)
    else:
        arrived = moving_piece
    changes.append((+1, arrived, move.to_square))

    if board_before.is_capture(move):
        if board_before.is_en_passant(move):
            ep_sq = move.to_square + (-8 if moving_piece.color == chess.WHITE else 8)
            captured = board_before.piece_at(ep_sq)
            if captured is not None:
                changes.append((-1, captured, ep_sq))
        else:
            captured = board_before.piece_at(move.to_square)
            if captured is not None:
                changes.append((-1, captured, move.to_square))

    if board_before.is_castling(move):
        color = moving_piece.color
        if board_before.is_kingside_castling(move):
            r_from, r_to = (chess.H1, chess.F1) if color == chess.WHITE else (chess.H8, chess.F8)
        else:
            r_from, r_to = (chess.A1, chess.D1) if color == chess.WHITE else (chess.A8, chess.D8)
        rook = chess.Piece(chess.ROOK, color)
        changes.append((-1, rook, r_from))
        changes.append((+1, rook, r_to))

    return changes


def _changes_to_indices(changes, perspective, king_sq_for_perspective):
    pos, neg = [], []
    for sign, piece, sq in changes:
        if piece.piece_type == chess.KING:
            continue
        target_sq = chess.square_mirror(sq) if perspective == chess.BLACK else sq
        idx = halfkp_index(piece, target_sq, king_sq_for_perspective, perspective)
        (pos if sign > 0 else neg).append(idx)
    return pos, neg


class NNUEAccumulator:
    def __init__(self, model):
        self.model = model
        self.model_device = next(model.parameters()).device
        self.hidden_dim = model.feature_transformer.embedding_dim
        # Weight lookups happen on every push (2-6 indices at a time).
        # GPU kernel launch overhead dominates for such tiny ops, so we keep a
        # CPU copy of the embedding weight and do all accumulator arithmetic there.
        # Only the final output-layer forward moves tensors to model_device.
        self._weight_cpu = model.feature_transformer.weight.data.cpu()
        self.white = torch.zeros(self.hidden_dim)   # CPU
        self.black = torch.zeros(self.hidden_dim)   # CPU
        self.stack = []

    @torch.no_grad()
    def _sum_indices(self, indices):
        if not indices:
            return torch.zeros(self.hidden_dim)
        idx_t = torch.tensor(indices, dtype=torch.long)
        return self._weight_cpu[idx_t].sum(dim=0)

    @torch.no_grad()
    def init_from_board(self, board):
        self.white = self._sum_indices(extract_halfkp(board, chess.WHITE))   # CPU
        self.black = self._sum_indices(extract_halfkp(board, chess.BLACK))   # CPU
        self.stack.clear()

    @torch.no_grad()
    def push(self, board, move):
        # Caller passes pre-move board. This method pushes the move on the board
        # AND updates the accumulator to reflect the post-move state.
        prev_white, prev_black = self.white, self.black
        self.stack.append((prev_white, prev_black))

        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:
            board.push(move)
            return

        is_king_move = moving_piece.piece_type == chess.KING
        changes = _piece_changes(board, move)
        white_king_sq = board.king(chess.WHITE)
        black_king_mirrored = chess.square_mirror(board.king(chess.BLACK))

        board.push(move)

        if is_king_move and moving_piece.color == chess.WHITE:
            # White king moved → all white-perspective indices change → full refresh.
            self.white = self._sum_indices(extract_halfkp(board, chess.WHITE))
            pos, neg = _changes_to_indices(changes, chess.BLACK, black_king_mirrored)
            self.black = prev_black + self._sum_indices(pos) - self._sum_indices(neg)
        elif is_king_move and moving_piece.color == chess.BLACK:
            self.black = self._sum_indices(extract_halfkp(board, chess.BLACK))
            pos, neg = _changes_to_indices(changes, chess.WHITE, white_king_sq)
            self.white = prev_white + self._sum_indices(pos) - self._sum_indices(neg)
        else:
            pos_w, neg_w = _changes_to_indices(changes, chess.WHITE, white_king_sq)
            self.white = prev_white + self._sum_indices(pos_w) - self._sum_indices(neg_w)
            pos_b, neg_b = _changes_to_indices(changes, chess.BLACK, black_king_mirrored)
            self.black = prev_black + self._sum_indices(pos_b) - self._sum_indices(neg_b)

    @torch.no_grad()
    def pop(self, board):
        board.pop()
        self.white, self.black = self.stack.pop()
