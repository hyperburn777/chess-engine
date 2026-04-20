import torch
import torch.nn as nn

NUM_FEATURES = 40960
EMB_DIM = 2048

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_FEATURES, EMB_DIM)

        self.mlp = nn.Sequential(
            nn.Linear(2 * EMB_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode(self, feats):
        padded = torch.nn.utils.rnn.pad_sequence(
            feats, batch_first=True, padding_value=0
        )
        emb = self.embedding(padded)
        mask = (padded != 0).unsqueeze(-1)
        return (emb * mask).sum(dim=1)

    
    def forward(self, white, black, stm):
        w = self.encode(white)
        b = self.encode(black)

        x = torch.where(
            stm.unsqueeze(1) == 1,
            torch.cat([w, b], dim=1),
            torch.cat([b, w], dim=1),
        )

        return self.mlp(x)