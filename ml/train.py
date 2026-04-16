import torch
from ml.model import ChessNet

model = ChessNet()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

print("Training script ready.")