import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from model import NNUE
from plot import plot_loss_curves
from dataset import ChessDataset, collate_fn, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNUE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []


dataset = load_data(20)

split1 = dataset.train_test_split(test_size=0.2, seed=42)

train_ds = split1["train"]
temp_ds = split1["test"]

# second split: val vs test
split2 = temp_ds.train_test_split(test_size=0.5, seed=42)

val_ds = split2["train"]
test_ds = split2["test"]

print(f"Train data size: {len(train_ds)}")
print(f"Val data size: {len(val_ds)}")
print(f"Test data size: {len(test_ds)}")

train_data = ChessDataset(train_ds)
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True
)
test_data = ChessDataset(test_ds)
test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True
)
val_data = ChessDataset(val_ds)
val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True
)


EPOCHS = 10

for epoch in range(EPOCHS):

    # --------------------
    # TRAIN
    # --------------------
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")

    for white, black, stm, target in train_bar:
        stm = stm.to(device)
        target = target.to(device)

        white = [w.to(device) for w in white]
        black = [b.to(device) for b in black]

        pred = model(white, black, stm)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()
        # train_bar.set_postfix(loss=loss.item())

    # --------------------
    # VALIDATION
    # --------------------
    model.eval()
    val_loss = 0.0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]")

    with torch.no_grad():
        for white, black, stm, target in val_bar:
            stm = stm.to(device)
            target = target.to(device)

            white = [w.to(device) for w in white]
            black = [b.to(device) for b in black]

            pred = model(white, black, stm)
            loss = loss_fn(pred, target)

            val_loss += loss.item()
            # val_bar.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(
        f"\nEpoch {epoch+1} Summary | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}\n"
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    torch.save(model.state_dict(), 'model_weights.pth')

model.eval()

test_loss = 0.0
mae = 0.0
count = 0

with torch.no_grad():
    for white, black, stm, target in test_loader:
        stm = stm.to(device)
        target = target.to(device)

        white = [w.to(device) for w in white]
        black = [b.to(device) for b in black]

        pred = model(white, black, stm)

        loss = loss_fn(pred, target)
        test_loss += loss.item()

        # optional: MAE (more interpretable than MSE)
        mae += torch.abs(pred - target).sum().item()
        count += target.numel()

test_loss /= len(test_loader)
mae /= count

print(f"\nFINAL TEST RESULTS")
print(f"Test MSE: {test_loss:.6f}")
print(f"Test MAE: {mae:.6f}")

plot_loss_curves(train_losses, val_losses)