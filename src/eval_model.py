import numpy as np
import json, torch, os
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F

from src.tools.mels_dataset import MelNpyDataset
from src.tools.CNNs import MODEL_PARAMS, SmallCNN
from src.tools.parse_args import parse_args
from src.tools.saver import RunSummary



def seed_worker(worker_id):
    # seed unique et déterministe par worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

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



@torch.no_grad()
def eval_acc_multicrop_majority(model, loader, device, target_T=128, K=10):

    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        B, C, Freq, T_full = x.shape

        # Cas où le signal est plus court que target_T : on pad et on fait une seule prédiction
        if T_full <= target_T:
            pad = target_T - T_full
            if pad > 0:
                x_pad = F.pad(x, (0, pad), mode="constant", value=0.0)
            else:
                x_pad = x
            logits = model(x_pad)              # [B, n_classes]
            preds = logits.argmax(1)           # [B]
        else:
            # T_full > target_T : K crops régulièrement espacés
            starts = torch.linspace(0, T_full - target_T, K, device=x.device).long()

            all_preds = []
            for s in starts:
                crop = x[..., s:s+target_T]    # [B, 1, 128, target_T]
                logits = model(crop)           # [B, n_classes]
                pred = logits.argmax(1)        # [B]
                all_preds.append(pred)

            # all_preds : liste de K tensors [B] -> [K, B] -> [B, K]
            all_preds = torch.stack(all_preds, dim=0).transpose(0, 1)  # [B, K]

            # majority vote sur la dimension K
            preds = torch.mode(all_preds, dim=1).values                # [B]

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total



if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parents[1]
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
    print(f'config = {config_run.to_dict()}')
    g_test = torch.Generator().manual_seed(config_run.seed)
    test_ds  = MelNpyDataset(mels_root, metadata_root, split="test",
                             target_T=1292, random_crop=config_run.random_crop)

    num_worker = min(os.cpu_count(), 12)
    test_loader = DataLoader(test_ds, batch_size=config_run.batch_size, shuffle=False, num_workers=num_worker,
                             worker_init_fn=seed_worker, generator=g_test, persistent_workers=True)

    model = MODEL_PARAMS.get(config_run.model_type, SmallCNN)(test_ds.n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(Path(__file__).resolve().parents[0] / "weight" / config_run.name)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    if args.vote:
        test_acc = eval_acc_multicrop_majority(
            model, test_loader, device,
            target_T=config_run.target_T, K=10
        )
    else:
        print('No vote classic evaluation')
        test_loss, test_acc = eval_loss_acc(
            model, test_loader, torch.nn.CrossEntropyLoss(), device,
        )
    print("FINAL test accuracy:", test_acc)