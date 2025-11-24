import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.tools.mels_dataset import MelNpyDataset
from src.tools.CNNs.CNNs import SmallCNN

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)


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


if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parents[2]
    mels_root = str(ROOT / "data" / "mels128")
    metadata_root = str(ROOT / "data" / "fma_metadata")

    test_ds  = MelNpyDataset(mels_root, metadata_root, split="test", target_T=1292)

    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=12)

    model = SmallCNN(test_ds.n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = str(Path(__file__).resolve().parents[0] / "baseline_model.pt")
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc = eval_loss_acc(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    print("FINAL test accuracy:", test_acc)