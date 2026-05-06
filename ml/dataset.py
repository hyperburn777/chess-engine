import chess
import torch
import numpy as np

def nnue_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None

    targets = torch.stack([b['target'] for b in batch])
    
    def prepare_indices_and_offsets(key):
        all_indices = []
        offsets = []
        current_offset = 0
        
        for sample in batch:
            indices = sample[key]
            all_indices.extend(indices)
            offsets.append(current_offset)
            current_offset += len(indices)
            
        return (
            torch.tensor(all_indices, dtype=torch.long), 
            torch.tensor(offsets, dtype=torch.long)
        )

    stm_indices, stm_offsets = prepare_indices_and_offsets("stm_indices")
    nstm_indices, nstm_offsets = prepare_indices_and_offsets("nstm_indices")

    return {
        "stm_idx": stm_indices, "stm_off": stm_offsets,
        "nstm_idx": nstm_indices, "nstm_off": nstm_offsets,
        "target": targets
    }

#see https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#halfkp
PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

# using the same formula provided by the stockfish site
def halfkp_index(piece, piece_square, king_square, perspective):
    piece_type = PIECE_MAP[piece.piece_type]
    piece_color = 0 if piece.color == perspective else 1
    p_idx = piece_type * 2 + piece_color
    return piece_square + (p_idx + king_square * 10) * 64

def extract_halfkp(board, perspective):
    king_sq = board.king(perspective)
    feats = []
    
    # If we are looking from Black's view, we "flip" the board 
    # so Black's King on g8 looks like a King on g1.
    if perspective == chess.BLACK:
        king_sq = chess.square_mirror(king_sq)

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
            
        target_sq = sq
        # If perspective is Black, we mirror every piece's square too
        if perspective == chess.BLACK:
            target_sq = chess.square_mirror(sq)
            
        feats.append(halfkp_index(piece, target_sq, king_sq, perspective))

    return feats

def transform_row(row):
    # 1. Score logic stays the same...
    if row.get('cp') is None:
        sign = 1 if row['mate'] > 0 else -1
        cp = sign * 1000 + (100 - abs(row['mate']))
    else:
        cp = max(min(row['cp'], 1000), -1000)
    score = np.tanh(cp / 400.0)

    board = chess.Board(row['fen'])
    
    # 2. Get active indices (HalfKP)
    # This now returns a LIST of integers (e.g., [1420, 5602, ...])
    white_indices = extract_halfkp(board, chess.WHITE)
    black_indices = extract_halfkp(board, chess.BLACK)
    
    if board.turn == chess.BLACK:
        score *= -1
        white_indices, black_indices = black_indices, white_indices

    # 3. Return the INDICES, not dense tensors
    # We don't convert to torch.tensor yet because lists of different 
    # lengths can't be stacked into a single tensor without padding.
    return {
        "stm_indices": white_indices,
        "nstm_indices": black_indices,
        "target": torch.tensor([score], dtype=torch.float32)
    }

def transform_batch(batch):
    # 1. Batch process scores using NumPy for speed
    cp_values = np.array(batch.get('cp', []), dtype=np.float32)
    # Handle Nones/NaNs if they exist in the batch
    scores = np.tanh(np.nan_to_num(cp_values, nan=0.0) / 400.0)

    stm_indices_batch = []
    nstm_indices_batch = []
    target_batch = []

    # 2. Iterate through the rows for chess-specific logic
    # (The chess library is not natively vectorized, so we loop here)
    for i, fen in enumerate(batch['fen']):
        board = chess.Board(fen)
        
        white_indices = extract_halfkp(board, chess.WHITE)
        black_indices = extract_halfkp(board, chess.BLACK)
        
        current_score = scores[i]

        # Handle Side-to-Move (STM) perspective
        if board.turn == chess.BLACK:
            current_score *= -1
            # Swap perspectives
            stm_indices = black_indices
            nstm_indices = white_indices
        else:
            stm_indices = white_indices
            nstm_indices = black_indices

        stm_indices_batch.append(stm_indices)
        nstm_indices_batch.append(nstm_indices)
        target_batch.append([current_score])

    # 3. Return a dictionary of lists
    return {
        "stm_indices": stm_indices_batch,
        "nstm_indices": nstm_indices_batch,
        "target": target_batch
    }