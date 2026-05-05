import torch
import torch.nn as nn

class NNUE(nn.Module):
    def __init__(self, feature_dim=40960, hidden_dim=512):
        super().__init__()
        # We use a simpler "Half-KA" or "Half-KP" feature set for speed
        self.feature_transformer = nn.EmbeddingBag(feature_dim, hidden_dim, mode='sum')
        self.activation = nn.Hardtanh(0, 1)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Hardtanh(0, 1),
            nn.Linear(128, 32),
            nn.Hardtanh(0, 1),
            nn.Linear(32, 1),
            nn.Tanh() # Maps output to [-1, 1] range for minimax
        )

    def forward(self, stm_idx, stm_off, nstm_idx, nstm_off):
        # EmbeddingBag magic: It sums the vectors for each "bag" (board)
        stm = self.activation(self.feature_transformer(stm_idx, stm_off))
        nstm = self.activation(self.feature_transformer(nstm_idx, nstm_off))
        
        combined = torch.cat([stm, nstm], dim=1)
        return self.output_layer(combined)