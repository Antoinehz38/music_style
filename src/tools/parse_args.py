import argparse

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run_from", type=str, default="new_conf")
    p.add_argument("--baseline", type=bool, default=False)
    p.add_argument("--target_T", type=int, default=1292)
    p.add_argument("--random_crop", type=bool, default=False)
    p.add_argument("--model_type", type=str, default="SmallCNN")
    p.add_argument("--dataset_type", type=str, default="MelNpyDataset")
    p.add_argument("--optim_type", type=str, default="Adam")
    p.add_argument("--batch_size", type=int,default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--val_training", type=bool, default=False)
    p.add_argument("--scheduler", type=bool,default=False)
    p.add_argument("--use_augment", type=bool, default=False)


    p.add_argument("--vote", dest="vote", action="store_true")
    p.add_argument("--no-vote", dest="vote", action="store_false")
    p.set_defaults(vote=True)


    return p.parse_args()

