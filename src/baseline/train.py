import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from time import time

from src.tools.mels_dataset import MelNpyDataset
from src.tools.CNNs.small_cnn import SmallCNN


def eval_loss_acc(model, loader, loss_fn, device):
    model.eval()
    total_loss, ok, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            ok += (pred == y).sum().item()
            total += y.numel()

    return total_loss / total, ok / total


def train(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=1e-3,
    device="cpu",
    log_dir="./runs/baseline"
):
    model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        nbatches = 0
        t0 = time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            global_step += 1

            epoch_loss += loss.item()
            nbatches += 1

        train_loss = epoch_loss / nbatches
        val_loss, val_acc = eval_loss_acc(model, val_loader, loss_fn, device)

        writer.add_scalar("train/epoch_loss", train_loss, ep)
        writer.add_scalar("val/loss", val_loss, ep)
        writer.add_scalar("val/accuracy", val_acc, ep)

        dt = time() - t0
        print(
            f"epoch {ep:02d} | train loss: {train_loss:.4f} "
            f"| val loss: {val_loss:.4f} | val acc: {val_acc:.3f} "
            f"| {dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"saved best model at epoch {ep}")

    writer.close()


if __name__ == "__main__":
    mels_root = "../../data/mels128"
    metadata_root = "../../data/fma_metadata"

    train_ds = MelNpyDataset(mels_root, metadata_root, split="training", target_T=1292)
    val_ds   = MelNpyDataset(mels_root, metadata_root, split="validation", target_T=1292)
    test_ds  = MelNpyDataset(mels_root, metadata_root, split="test", target_T=1292)

    print("train classes:", train_ds.classes)
    print("val classes  :", val_ds.classes)
    print("test classes :", test_ds.classes)
    print("same train/val?", train_ds.classes == val_ds.classes)
    print("same train/test?", train_ds.classes == test_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=12)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=12)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=12)

    model = SmallCNN(train_ds.n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(model, train_loader, val_loader, epochs=20, lr=1e-3, device=device)

    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc = eval_loss_acc(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    print("FINAL test accuracy:", test_acc)
