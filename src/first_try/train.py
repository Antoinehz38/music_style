import os, numpy as np, json
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from time import time
from pathlib import Path

from src.tools.mels_dataset import MelNpyDataset
from src.tools.CNNs.CNNs import MODEL_PARAMS, SmallCNN
from src.tools.parse_args import parse_args
from src.tools.config_saver.saver import RunSummary
from src.first_try.eval_model import eval_loss_acc_multicrop

OPTIM_PARAMS = {"AdamW":AdamW}

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

g_train = torch.Generator().manual_seed(1234)
g_val   = torch.Generator().manual_seed(1234)
g_test  = torch.Generator().manual_seed(1234)

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
    run_config:RunSummary,
    device="cpu",
    log_dir="./runs/first_try",
):
    model.to(device)
    opt = OPTIM_PARAMS.get(run_config.optim_type, AdamW)(model.parameters(), lr=run_config.lr,
                                                         weight_decay=run_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=config_run.epoch
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    global_step = 0

    for ep in range(1, config_run.epoch + 1):
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
            model_path = "src/first_try/best_model.pt"
            torch.save(model.state_dict(), model_path)
            print(f"saved best model at epoch {ep}")

        scheduler.step()
    torch.save(model.state_dict(), "src/first_try/" + config_run.name)
    writer.close()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    mels_root = str(ROOT / "data" / "mels128")
    metadata_root = str(ROOT / "data" / "fma_metadata")
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args = parse_args()
    if args.run_from == "new_conf":
        config_run = RunSummary(random_crop=True, model_type="CRNN", dataset_type="MelNpyDataset",optim_type="AdamW",
                                target_T=256, seed=1234, batch_size=32, lr=3e-4, weight_decay=1e-4, epoch=35)
        config_run.name = f"best_model_{time}.pt"
    else:
        config_run = RunSummary()
        config_run.name = f"best_model_{time}.pt"

        with open(args.run_from, "r", encoding="utf-8") as f:
            data = json.load(f)
        config_run.load_data(data)

    train_ds = MelNpyDataset(mels_root, metadata_root, split="training",
                             target_T=config_run.target_T, random_crop=config_run.random_crop)
    val_ds   = MelNpyDataset(mels_root, metadata_root, split="validation",
                             target_T=config_run.target_T, random_crop=config_run.random_crop)
    test_ds  = MelNpyDataset(mels_root, metadata_root, split="test",
                             target_T=1292, random_crop=config_run.random_crop)


    train_loader = DataLoader(train_ds, batch_size=config_run.batch_size, shuffle=True, num_workers=12,
                              worker_init_fn=seed_worker,generator=g_train,persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=config_run.batch_size, shuffle=False, num_workers=12,
                              worker_init_fn=seed_worker, generator=g_val, persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=config_run.batch_size, shuffle=False, num_workers=12,
                              worker_init_fn=seed_worker,generator=g_test,persistent_workers=True)

    model = MODEL_PARAMS.get(config_run.model_type, SmallCNN)(train_ds.n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(model, train_loader, val_loader, run_config=config_run, device=device,
          log_dir=str(Path(__file__).resolve().parents[0] / "./runs/first_try"))

    model.load_state_dict(torch.load("src/first_try/best_model.pt"))
    test_loss, test_acc = eval_loss_acc_multicrop(
        model, test_loader, torch.nn.CrossEntropyLoss(), device,
        target_T=config_run.target_T, K=5
    )
    print("FINAL test accuracy:", test_acc)
    config_run.test_results=test_acc

    file_to_save = f"src/runs_configs/{time}.json"

    if args.run_from == "new_conf":
        with open(file_to_save, "w", encoding="utf-8") as f:
            json.dump(config_run.to_dict(), f, indent=2)
