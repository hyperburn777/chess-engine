import torch
import torch.nn as nn

NUM_FEATURES=40960
M = 256
N = 32
K = 1

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()

        self.ft = nn.Linear(NUM_FEATURES, M)
        self.l1 = nn.Linear(2 * M, N)
        self.l2 = nn.Linear(N, K)

    # The inputs are a whole batch!
    # `stm` indicates whether white is the side to move. 1 = true, 0 = false.
    def forward(self, white_features, black_features, stm):
        w = self.ft(white_features)
        b = self.ft(black_features)

        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        x = torch.clamp(accumulator, 0.0, 1.0)
        x = torch.clamp(self.l1(x), 0.0, 1.0)

        return self.l2(x)