import os, numpy as np, json
import torch
import shutil

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.set_num_threads(os.cpu_count())
print('nombre de cpu : ', os.cpu_count())
torch.set_num_interop_threads(1)

from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from torch.utils.tensorboard import SummaryWriter
from time import time
from pathlib import Path

from src.tools.mels_dataset import DATASET_PARAMS, MelNpyDataset
from src.tools.CNNs import MODEL_PARAMS, SmallCNN
from src.tools.parse_args import parse_args
from src.tools.saver import RunSummary
from src.eval_model import eval_acc_multicrop_majority, eval_loss_acc

OPTIM_PARAMS = {"AdamW":AdamW, "Adam":Adam}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

g_train = torch.Generator().manual_seed(seed)
g_val   = torch.Generator().manual_seed(seed)
g_test  = torch.Generator().manual_seed(seed)

def train(
    model,
    train_loader,
    val_loader,
    config_run:RunSummary,
    device="cpu",
    log_dir="./runs",
):
    global args
    model.to(device)
    if config_run.weight_decay:
        opt = OPTIM_PARAMS.get(config_run.optim_type, AdamW)(model.parameters(),
                                                             lr=config_run.lr,
                                                             weight_decay=config_run.weight_decay)
    else:
        opt = OPTIM_PARAMS.get(config_run.optim_type, AdamW)(model.parameters(),
                                                             lr=config_run.lr)

    if not config_run.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=config_run.epoch
        )

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    global_step = 0


    patience = int(config_run.epoch * 1)
    epochs_no_improve = 0

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

        if val_loss < best_val_loss :
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_path = "src/weight/best_model_val_loss.pt"
            torch.save(model.state_dict(), model_path)
            print(f"saved best_val_loss model at epoch {ep}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            model_path = "src/weight/best_model_val_acc.pt"
            torch.save(model.state_dict(), model_path)
            print(f"saved best_val_acc model at epoch {ep}")
        if val_acc < best_val_acc and val_loss > best_val_loss:
            epochs_no_improve += 1
            print(f"no improvement for {epochs_no_improve} epoch(s)")

        # scheduler
        if not config_run.scheduler:
            scheduler.step()

        # early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {ep}")
            break

    torch.save(model.state_dict(), "src/weight/" + config_run.name)
    writer.close()



if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    mels_root = str(ROOT / "data" / "mels128")
    metadata_root = str(ROOT / "data" / "fma_metadata")
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args = parse_args()
    print('baseline = ', args.baseline)
    if args.run_from == "new_conf":
        config_run = RunSummary(random_crop=args.random_crop, model_type=args.model_type, dataset_type=args.dataset_type,
                                optim_type=args.optim_type,target_T=args.target_T, seed=seed, batch_size=args.batch_size,
                                lr=args.lr, weight_decay=args.weight_decay, epoch=args.epoch, val_training=args.val_training,
                                scheduler = args.scheduler, use_augment=args.use_augment)
    else:
        config_run = RunSummary()
        with open(args.run_from, "r", encoding="utf-8") as f:
            data = json.load(f)
        config_run.load_data(data)

    config_run.name = f"best_model_{now}.pt"

    if config_run.val_training:
        split_train = "trainval"
    else:
        split_train = "training"

    Dataset = DATASET_PARAMS.get(config_run.dataset_type, MelNpyDataset)

    train_ds = Dataset(mels_root, metadata_root, split=split_train,
                             target_T=config_run.target_T, random_crop=config_run.random_crop)
    val_ds   = MelNpyDataset(mels_root, metadata_root, split="validation",
                             target_T=config_run.target_T, random_crop=config_run.random_crop)
    test_ds  = MelNpyDataset(mels_root, metadata_root, split="test",
                             target_T=1292, random_crop=config_run.random_crop)

    num_worker = min(os.cpu_count(), 12)
    train_loader = DataLoader(train_ds, batch_size=config_run.batch_size, shuffle=True, num_workers=num_worker,
                              worker_init_fn=seed_worker,generator=g_train,persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=config_run.batch_size, shuffle=False, num_workers=num_worker,
                              worker_init_fn=seed_worker, generator=g_val, persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=config_run.batch_size, shuffle=False, num_workers=num_worker,
                              worker_init_fn=seed_worker,generator=g_test,persistent_workers=True)

    model = MODEL_PARAMS.get(config_run.model_type, SmallCNN)(train_ds.n_classes, augment=config_run.use_augment)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(config_run.to_dict())
    train(model, train_loader, val_loader, config_run=config_run, device=device,
          log_dir=str(Path(__file__).resolve().parents[0] / "./runs"))


    model.load_state_dict(torch.load("src/weight/" + config_run.name))
    _, test_acc = eval_loss_acc(
        model, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), device=device)
    print("FINAL test accuracy last :", test_acc)

    model.load_state_dict(torch.load("src/weight/best_model_val_loss.pt"))
    _, test_acc_best_val_loss = eval_loss_acc(
        model, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), device=device)
    print("FINAL test accuracy best val loss :", test_acc_best_val_loss)

    model.load_state_dict(torch.load("src/weight/best_model_val_acc.pt"))
    _, test_acc_best_val_acc = eval_loss_acc(
        model, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), device=device)
    print("FINAL test accuracy best val acc:", test_acc_best_val_acc)

    if test_acc_best_val_acc > test_acc_best_val_loss:
        if test_acc_best_val_acc > test_acc:
            shutil.copyfile("src/weight/best_model_val_acc.pt","src/weight/" + config_run.name)
            config_run.test_results=test_acc_best_val_acc
        else:
            config_run.test_results = test_acc
    else:
        if test_acc_best_val_loss > test_acc:
            shutil.copyfile("src/weight/best_model_val_loss.pt","src/weight/" + config_run.name)
            config_run.test_results=test_acc_best_val_loss
        else:
            config_run.test_results = test_acc

    file_to_save = f"src/runs_configs/{now}.json"

    if args.run_from == "new_conf":
        with open(file_to_save, "w", encoding="utf-8") as f:
            json.dump(config_run.to_dict(), f, indent=2)
