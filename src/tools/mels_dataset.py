import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MelNpyDataset(Dataset):
    def __init__(self, mels_root, metadata_root, split="training", target_T=1292):
        tracks = pd.read_csv(os.path.join(metadata_root, "tracks.csv"), header=[0,1], index_col=0)

        subset = tracks[("set", "subset")] == "small"
        split_mask = tracks[("set", "split")] == split
        df = tracks[subset & split_mask]

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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        mel = np.load(self.files[i]).astype(np.float32)  # (128, T)

        T = mel.shape[1]
        if T > self.target_T:
            mel = mel[:, :self.target_T]
        elif T < self.target_T:
            pad = self.target_T - T
            mel = np.pad(mel, ((0,0),(0,pad)), mode="constant")

        mel = torch.from_numpy(mel).unsqueeze(0)  # (1,128,target_T)
        y = int(self.labels[i])
        return mel, torch.tensor(y, dtype=torch.long)

