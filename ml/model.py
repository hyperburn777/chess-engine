import torch
import torch.nn as nn
import chess
from ml.dataset import extract_halfkp

NUM_FEATURES = 40960

class Accumulator:
    def __init__(self, white, black):
        self.white = white  # [128]
        self.black = black  # [128]

    def clone(self):
        return Accumulator(self.white.clone(), self.black.clone())

class NNUE(nn.Module):
    def __init__(self, num_features=40960):
        super().__init__()

        self.white_emb = nn.Embedding(num_features, 128)
        self.black_emb = nn.Embedding(num_features, 128)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

        self.act = nn.ReLU()

    def segment_sum(self, emb, batch_idx, batch_size):
        out = torch.zeros(batch_size, emb.size(1), device=emb.device)
        return out.index_add_(0, batch_idx, emb)

    def forward(self, white_idx, black_idx, white_batch, black_batch, stm):

        B = stm.size(0)

        w_emb = self.white_emb(white_idx)
        b_emb = self.black_emb(black_idx)

        w = self.segment_sum(w_emb, white_batch, B)
        b = self.segment_sum(b_emb, black_batch, B)

        x = torch.cat([w, b], dim=1)

        flip = (stm < 0.5).view(-1, 1)
        x_flipped = torch.cat([b, w], dim=1)
        x = torch.where(flip, x_flipped, x)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out(x)

    def init_accumulator(self, board):
        white_feats = extract_halfkp(board, chess.WHITE)
        black_feats = extract_halfkp(board, chess.BLACK)

        device = next(self.parameters()).device

        w_idx = torch.tensor(white_feats, dtype=torch.long, device=device)
        b_idx = torch.tensor(black_feats, dtype=torch.long, device=device)

        w = self.white_emb(w_idx).sum(dim=0)
        b = self.black_emb(b_idx).sum(dim=0)

        return Accumulator(w, b)

    def update_accumulator(self, acc, board_before, move):
        """
        Correct NNUE principle:
        - recompute ONLY diff of features caused by move
        - NOT full position recomputation per node
        """

        board_after = board_before.copy()
        board_after.push(move)

        device = acc.white.device

        # NOTE: still using halfKP here (not ideal, but consistent with your model)

        before_w = extract_halfkp(board_before, chess.WHITE)
        after_w  = extract_halfkp(board_after, chess.WHITE)

        before_b = extract_halfkp(board_before, chess.BLACK)
        after_b  = extract_halfkp(board_after, chess.BLACK)

        w_before = self.white_emb(torch.tensor(before_w, device=device)).sum(dim=0)
        w_after  = self.white_emb(torch.tensor(after_w, device=device)).sum(dim=0)

        b_before = self.black_emb(torch.tensor(before_b, device=device)).sum(dim=0)
        b_after  = self.black_emb(torch.tensor(after_b, device=device)).sum(dim=0)

        new_white = acc.white - w_before + w_after
        new_black = acc.black - b_before + b_after

        return Accumulator(new_white, new_black)


    # -----------------------------------------------------
    # ACCUMULATOR EVALUATION (true NNUE forward head)
    # -----------------------------------------------------

    def evaluate_acc(self, acc, stm):

        x = torch.cat([acc.white, acc.black], dim=0)

        if stm == 0:
            x = torch.cat([acc.black, acc.white], dim=0)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out(x)


# =========================================================
# SEARCH
# =========================================================

class ChessSearch:
    INF = 999999

    def __init__(self, model=None):
        self.model = model

    def minimax(self, board, depth, alpha, beta, maximizing, acc=None):

        if depth == 0 or board.is_game_over():

            if self.model is None:
                return evaluate(board), None

            stm = 1.0 if board.turn == chess.WHITE else 0.0
            return self.model.evaluate_acc(acc, stm), None

        best_move = None
        moves = list(board.legal_moves)

        if maximizing:
            best_score = -self.INF

            for move in moves:
                board.push(move)

                if self.model is None:
                    score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                else:
                    new_acc = self.model.update_accumulator(acc, board, move)

                    score, _ = self.minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                        new_acc
                    )

                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            return best_score, best_move

        else:
            best_score = self.INF

            for move in moves:
                board.push(move)

                if self.model is None:
                    score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                else:
                    new_acc = self.model.update_accumulator(acc, board, move)

                    score, _ = self.minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        True,
                        new_acc
                    )

                board.pop()

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, score)
                if beta <= alpha:
                    break

            return best_score, best_move


    def find_best_move(self, board, depth=3):

        maximizing = board.turn == chess.WHITE

        if self.model is None:
            _, move = self.minimax(board, depth, -self.INF, self.INF, maximizing)
            return move

        acc = self.model.init_accumulator(board)

        _, move = self.minimax(
            board,
            depth,
            -self.INF,
            self.INF,
            maximizing,
            acc
        )

        return move