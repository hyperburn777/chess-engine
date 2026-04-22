import chess
import torch

from ml.dataset import extract_halfkp

@torch.no_grad()
def model_evaluate_board(model, board, device):
    model.eval()
    
    # 1. Extract indices using your existing logic
    w_indices = extract_halfkp(board, chess.WHITE)
    b_indices = extract_halfkp(board, chess.BLACK)

    if board.turn == chess.BLACK:
        w_indices, b_indices = w_indices, b_indices
    
    # 2. Convert to tensors
    # Since it's a single board, the offset is just 0
    stm_idx = torch.tensor(w_indices, dtype=torch.long).to(device)
    stm_off = torch.tensor([0], dtype=torch.long).to(device)
    
    nstm_idx = torch.tensor(b_indices, dtype=torch.long).to(device)
    nstm_off = torch.tensor([0], dtype=torch.long).to(device)
    
    # 3. Forward pass
    prediction = model(stm_idx, stm_off, nstm_idx, nstm_off)
    
    # 4. Perspective Adjustment
    # Remember: your model was trained on "Side To Move" (STM) scores
    # If it's Black's turn, the model output is relative to Black.
    # To keep it standard (White = positive), we flip if it's Black's turn.
    score = prediction.item()
    
    # Optional: Convert Tanh [-1, 1] back to roughly Centipawns
    # cp_score = np.arctanh(np.clip(score, -0.99, 0.99)) * 400
    
    return score

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
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (disables dropout/batchnorm)
        self.device = device

    def evaluate(self, board):

        # 1. Extract features
        white_feats = extract_halfkp(board, chess.WHITE)
        black_feats = extract_halfkp(board, chess.BLACK)
        stm = 1.0 if board.turn == chess.WHITE else 0.0

        # 2. Convert to tensors (FLAT, no batch dimension yet)
        white_idx = torch.tensor(white_feats, dtype=torch.long, device=self.device)
        black_idx = torch.tensor(black_feats, dtype=torch.long, device=self.device)

        # 3. Build "batch index" (everything belongs to sample 0)
        white_batch = torch.zeros_like(white_idx, dtype=torch.long, device=self.device)
        black_batch = torch.zeros_like(black_idx, dtype=torch.long, device=self.device)

        # 4. Side to move tensor (batch size = 1)
        stm_t = torch.tensor([stm], dtype=torch.float32, device=self.device)

        # 5. Add fake batch dimension logic inside model call
        white_idx = white_idx
        black_idx = black_idx

        # 6. Inference
        with torch.no_grad():
            output = self.model(
                white_idx,
                black_idx,
                white_batch,
                black_batch,
                stm_t
            )

            score = output.item()

        return score