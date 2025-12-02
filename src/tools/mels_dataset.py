import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MelNpyDataset(Dataset):
    def __init__(self, mels_root, metadata_root, split="training", target_T=1292, random_crop:bool = False):
        tracks = pd.read_csv(
            os.path.join(metadata_root, "tracks.csv"),
            header=[0, 1],
            index_col=0
        )

        subset_mask = tracks[("set", "subset")] == "small"

        # gérer plusieurs splits
        if split == "trainval":
            split_values = ["training", "validation"]
        elif isinstance(split, (list, tuple, set)):
            split_values = list(split)
        else:
            split_values = [split]

        split_mask = tracks[("set", "split")].isin(split_values)
        df = tracks[subset_mask & split_mask]

        self.track_ids = df.index.tolist()
        self.files = [os.path.join(mels_root, f"{tid}.npy") for tid in self.track_ids]

        kept = [(tid, f) for tid, f in zip(self.track_ids, self.files) if os.path.isfile(f)]
        self.track_ids, self.files = zip(*kept)
        self.track_ids = list(self.track_ids)
        self.files = list(self.files)

        self.genre_ids = df.loc[self.track_ids, ("track", "genre_top")].astype("category")
        self.labels = self.genre_ids.cat.codes.to_numpy()
        self.classes = list(self.genre_ids.cat.categories)
        self.n_classes = len(self.classes)

        self.target_T = target_T
        self.random_crop = random_crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        mel = np.load(self.files[i]).astype(np.float32)  # (128, T)

        T = mel.shape[1]
        if T > self.target_T:
            if self.random_crop:
                start = np.random.randint(0, T - self.target_T)
                mel = mel[:, start:start + self.target_T]
            else:
                mel = mel[:, :self.target_T]
        elif T < self.target_T:
            pad = self.target_T - T
            mel = np.pad(mel, ((0,0),(0,pad)), mode="constant")

        mel = torch.from_numpy(mel).unsqueeze(0)  # (1,128,target_T)
        y = int(self.labels[i])
        return mel, torch.tensor(y, dtype=torch.long)


class NewMelNpyDataset(Dataset):
    def __init__(self, mels_root, metadata_root, split="training", target_T=1292, random_crop: bool = False):
        tracks = pd.read_csv(os.path.join(metadata_root, "tracks.csv"), header=[0, 1], index_col=0)

        subset = tracks[("set", "subset")] == "small"
        split_mask = tracks[("set", "split")] == split
        df = tracks[subset & split_mask]

        self.track_ids = df.index.tolist()
        self.files = [os.path.join(mels_root, f"{tid}.npy") for tid in self.track_ids]

        kept = [(tid, f) for tid, f in zip(self.track_ids, self.files) if os.path.isfile(f)]
        if len(kept) == 0:
            raise RuntimeError("No .npy files found for given root/metadata/split")
        self.track_ids, self.files = zip(*kept)
        self.track_ids = list(self.track_ids)
        self.files = list(self.files)

        self.genre_ids = df.loc[self.track_ids, ("track", "genre_top")].astype("category")
        self.labels = self.genre_ids.cat.codes.to_numpy()
        self.classes = list(self.genre_ids.cat.categories)
        self.n_classes = len(self.classes)

        self.target_T = target_T
        self.random_crop = random_crop
        self.split = split

        # Pour le split "training", on pré-génère les segments
        self.seg_index = None
        if self.split == "training":
            self.seg_index = []  # liste de (track_idx, start)
            for track_idx, f in enumerate(self.files):
                mel = np.load(f, mmap_mode="r")  # (128, T)
                T = mel.shape[1]
                n_segs = max(1, T // self.target_T)  # int(T/target_T), mais au moins 1
                for k in range(n_segs):
                    start = k * self.target_T
                    self.seg_index.append((track_idx, start))

    def __len__(self):
        if self.split == "training":
            return len(self.seg_index)
        else:
            return len(self.files)

    def __getitem__(self, i):
        if self.split == "training":
            track_idx, start = self.seg_index[i]
            mel = np.load(self.files[track_idx]).astype(np.float32)  # (128, T)
            T = mel.shape[1]

            end = start + self.target_T
            seg = mel[:, start:min(end, T)]
            if seg.shape[1] < self.target_T:
                pad = self.target_T - seg.shape[1]
                seg = np.pad(seg, ((0, 0), (0, pad)), mode="constant")

            x = torch.from_numpy(seg).unsqueeze(0)  # (1, 128, target_T)
            y = int(self.labels[track_idx])
            return x, torch.tensor(y, dtype=torch.long)

        # comportement identique à MelNpyDataset pour val/test
        mel = np.load(self.files[i]).astype(np.float32)  # (128, T)
        T = mel.shape[1]

        if T > self.target_T:
            if self.random_crop:
                start = np.random.randint(0, T - self.target_T)
                mel = mel[:, start:start + self.target_T]
            else:
                mel = mel[:, :self.target_T]
        elif T < self.target_T:
            pad = self.target_T - T
            mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")

        x = torch.from_numpy(mel).unsqueeze(0)  # (1, 128, target_T)
        y = int(self.labels[i])
        return x, torch.tensor(y, dtype=torch.long)


DATASET_PARAMS = {"NewMelNpyDataset": NewMelNpyDataset, "MelNpyDataset": MelNpyDataset}