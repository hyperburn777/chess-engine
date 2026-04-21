import torch
import torch.nn as nn

NUM_FEATURES = 40960
EMB_DIM = 2048

class NNUE(nn.Module):
    def __init__(self, num_features=40960):
        super().__init__()

        self.white_emb = nn.Embedding(num_features, 256)
        self.black_emb = nn.Embedding(num_features, 256)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

        self.act = nn.ReLU()

    def segment_sum(self, emb, batch_idx, batch_size):
        out = torch.zeros(batch_size, emb.size(1), device=emb.device)
        return out.index_add_(0, batch_idx, emb)

    def forward(self, white_idx, black_idx, white_batch, black_batch, stm):

        B = stm.size(0)

        w_emb = self.white_emb(white_idx)
        b_emb = self.black_emb(black_idx)

        # 🔥 vectorized accumulation (NO LOOP)
        w = self.segment_sum(w_emb, white_batch, B)
        b = self.segment_sum(b_emb, black_batch, B)

        x = torch.cat([w, b], dim=1)

        flip = (stm == 0).view(-1, 1)
        x_flipped = torch.cat([b, w], dim=1)
        x = torch.where(flip, x_flipped, x)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.out(x)