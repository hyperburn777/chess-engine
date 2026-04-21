import chess
import torch

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset

def sample_subset(dataset, sample_size=1000, seed=42):
    """
    Shuffles the dataset and selects a small subset.
    """
    if sample_size > len(dataset):
        return dataset
        
    # Shuffle with a seed for reproducibility
    shuffled_dataset = dataset.shuffle(seed=seed)
    
    # Select the first N items
    subset = shuffled_dataset.select(range(sample_size))
    
    return subset

def load_data(chunks):
    subsets = []
    slice = 100 // chunks
    for i in range(chunks):
        print(f"Processing chunk {i+1}/{chunks}...")
        dataset = load_dataset(
            "mateuszgrzyb/lichess-stockfish-normalized",
            split=f"train[{i*slice}%:{(i+1)*slice}%]",
            cache_dir="./training_data"
        )

        subset = sample_subset(dataset)
        subsets.append(subset)
    
    final_ds = concatenate_datasets(subsets)
    final_ds.shuffle()
    return final_ds

#see https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#halfkp
PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

# using the same formula provided by the stockfish site
def halfkp_index(piece, piece_square, king_square):
    piece_type = PIECE_MAP[piece.piece_type]
    piece_color = 0 if piece.color == chess.WHITE else 1
    p_idx = piece_type * 2 + piece_color
    return piece_square + (p_idx + king_square * 10) * 64

def extract_halfkp(board, perspective):
    king_sq = board.king(perspective)
    feats = []

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        feats.append(halfkp_index(piece, sq, king_sq))

    return feats

def collate_fn(batch):
    white = [b["white"] for b in batch]
    black = [b["black"] for b in batch]

    stm = torch.stack([b["stm"] for b in batch])
    target = torch.stack([b["target"] for b in batch])

    return white, black, stm, target

class ChessDataset(Dataset):
    def __init__(self, data, max_samples=None):
        self.ds = data if max_samples is None else data.select(range(max_samples))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        board = chess.Board(item["fen"])

        white_feats = extract_halfkp(board, chess.WHITE)
        black_feats = extract_halfkp(board, chess.BLACK)

        # side to move
        stm = 1.0 if board.turn == chess.WHITE else 0.0

        # clip and normalize bc centipawn can go into the infinities as outliers (for mate)
        cp = item["cp"] if item["cp"] is not None else 0
        cp = max(min(cp, 1000), -1000) / 1000.0 

        return {
            "white": torch.tensor(white_feats, dtype=torch.long),
            "black": torch.tensor(black_feats, dtype=torch.long),
            "stm": torch.tensor(stm, dtype=torch.float32),
            "target": torch.tensor([cp], dtype=torch.float32)
        }