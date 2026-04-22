import chess
import torch

from ml.dataset import extract_halfkp

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def evaluate(board):
    score = 0

    for piece_type in PIECE_VALUES:
        score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]

    return score

class ChessModelEvaluator:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, board):
        # 1. Extract feature indices
        white_feats = extract_halfkp(board, chess.WHITE)
        black_feats = extract_halfkp(board, chess.BLACK)
        stm = 1.0 if board.turn == chess.WHITE else 0.0

        # 2. Build dense feature tensors (batch size = 1)
        white_tensor = torch.zeros(1, 40960, dtype=torch.float32, device=self.device)
        black_tensor = torch.zeros(1, 40960, dtype=torch.float32, device=self.device)

        white_tensor[0, white_feats] = 1.0
        black_tensor[0, black_feats] = 1.0

        # 3. Side-to-move tensor (shape: [1, 1])
        stm_tensor = torch.tensor([[stm]], dtype=torch.float32, device=self.device)

        # 4. Inference
        with torch.no_grad():
            output = self.model(white_tensor, black_tensor, stm_tensor)
            score = output.item() * 1000.0

        return score