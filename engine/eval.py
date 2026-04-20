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
        self.model.eval()  # Set to evaluation mode (disables dropout/batchnorm)

    def evaluate(self, board):
        """
        Takes a python-chess board, extracts features, and runs a forward pass.
        Returns a centipawn-adjacent score.
        """
        # 1. Extract features (using your existing utility function)
        white_feats = extract_halfkp(board, chess.WHITE)
        black_feats = extract_halfkp(board, chess.BLACK)
        stm = 1.0 if board.turn == chess.WHITE else 0.0

        # 2. Convert to tensors and add batch dimension (dim 0)
        white_t = torch.tensor(white_feats, dtype=torch.long).unsqueeze(0).to(self.device)
        black_t = torch.tensor(black_feats, dtype=torch.long).unsqueeze(0).to(self.device)
        stm_t = torch.tensor([stm], dtype=torch.float32).unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            output = self.model(white_t, black_t, stm_t)
            
            # The model output is likely normalized (e.g., -1.0 to 1.0) 
            # based on your dataset code. We scale it back to centipawns.
            score = output.item() * 1000.0

        return int(score)