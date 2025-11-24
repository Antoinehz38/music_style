import os, numpy as np
import torch, json
from torch.utils.data import DataLoader
from pathlib import Path

from src.tools.mels_dataset import MelNpyDataset
from src.tools.CNNs.CNNs import MODEL_PARAMS, SmallCNN
from src.tools.parse_args import parse_args
from src.tools.config_saver.saver import RunSummary

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)

seed = 1234

def seed_worker(worker_id):
    # seed unique et déterministe par worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
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

import torch
import torch.nn.functional as F

@torch.no_grad()
def multicrop_logits(model, x_full, target_T=128, K=5):
    """
    x_full: [B,1,128,T_full]
    retourne: logits_moy [B,n_classes]
    crops déterministes espacés régulièrement.
    """
    B, C, Freq, T_full = x_full.shape
    device = x_full.device

    if T_full <= target_T:
        # pad à droite si besoin (ou laisse tel quel si ton dataset pad déjà)
        pad = target_T - T_full
        if pad > 0:
            x = F.pad(x_full, (0, pad), mode="constant", value=0.0)
        else:
            x = x_full
        return model(x)

    # K positions régulièrement espacées
    starts = torch.linspace(0, T_full - target_T, K, device=device).long()

    logits_sum = 0.0
    for s in starts:
        crop = x_full[..., s:s+target_T]        # [B,1,128,target_T]
        logits_sum = logits_sum + model(crop)  # [B,n_classes]

    return logits_sum / K


@torch.no_grad()
def eval_loss_acc_multicrop(model, loader, loss_fn, device, target_T=128, K=5):
    model.eval()
    total_loss, ok, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = multicrop_logits(model, x, target_T=target_T, K=K)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        ok += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, ok / total



if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parents[2]
    mels_root = str(ROOT / "data" / "mels128")
    metadata_root = str(ROOT / "data" / "fma_metadata")

    args = parse_args()
    if args.run_from == "new_conf":
        config_run = RunSummary(random_crop=True, model_type="CRNN", dataset_type="MelNpyDataset", optim_type="AdamW",
                                target_T=128, seed=1234, batch_size=32, lr=3e-4, weight_decay=1e-4, epoch=23)
    else:
        config_run = RunSummary()

        with open(args.run_from, "r", encoding="utf-8") as f:
            data = json.load(f)
        config_run.load_data(data)

    test_ds  = MelNpyDataset(mels_root, metadata_root, split="test",
                             target_T=1292, random_crop=config_run.random_crop)

    test_loader = DataLoader(test_ds, batch_size=config_run.batch_size, shuffle=False, num_workers=12,
                             worker_init_fn=seed_worker, generator=g_test, persistent_workers=True)

    model = MODEL_PARAMS.get(config_run.model_type, SmallCNN)(test_ds.n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(Path(__file__).resolve().parents[0] / config_run.name)
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc = eval_loss_acc_multicrop(
        model, test_loader, torch.nn.CrossEntropyLoss(), device,
        target_T=config_run.target_T, K=5
    )
    print("FINAL test accuracy:", test_acc)