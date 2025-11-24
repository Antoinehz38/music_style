import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MelNpyDataset(Dataset):
    def __init__(self, mels_root, metadata_root, split="training", target_T=1292, random_crop:bool = False):
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
    def __init__(
        self,
        mels_root,
        metadata_root,
        split="training",
        target_T=1292,
        random_crop=False,
        augment=False,          # <-- new
        p_specaug=0.8,          # prob SpecAugment
        p_noise=0.3,            # prob noise
        p_gain=0.3,             # prob gain
        p_shift=0.3,            # prob time shift
        max_time_mask=0.12,     # % of T
        max_freq_mask=0.12,     # % of F (128)
        noise_std=0.01,         # relative noise
        max_shift_frac=0.1      # % of T
    ):
        tracks = pd.read_csv(os.path.join(metadata_root, "tracks.csv"),
                             header=[0,1], index_col=0)

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
        self.random_crop = random_crop

        # aug params
        self.augment = augment and (split == "training")
        self.p_specaug = p_specaug
        self.p_noise = p_noise
        self.p_gain = p_gain
        self.p_shift = p_shift
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask
        self.noise_std = noise_std
        self.max_shift_frac = max_shift_frac

    def __len__(self):
        return len(self.files)

    def _spec_augment(self, mel):
        # mel: (F, T)
        F, T = mel.shape

        # time mask
        t_mask = int(self.max_time_mask * T)
        if t_mask > 0:
            w = np.random.randint(0, t_mask + 1)
            if w > 0:
                t0 = np.random.randint(0, T - w + 1)
                mel[:, t0:t0+w] = 0.0

        # freq mask
        f_mask = int(self.max_freq_mask * F)
        if f_mask > 0:
            w = np.random.randint(0, f_mask + 1)
            if w > 0:
                f0 = np.random.randint(0, F - w + 1)
                mel[f0:f0+w, :] = 0.0

        return mel

    def _time_shift(self, mel):
        # circular shift on time axis
        T = mel.shape[1]
        max_s = int(self.max_shift_frac * T)
        if max_s > 0:
            s = np.random.randint(-max_s, max_s + 1)
            mel = np.roll(mel, shift=s, axis=1)
        return mel

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

        # ----- AUGMENTATIONS (train only) -----
        if self.augment:
            if np.random.rand() < self.p_shift:
                mel = self._time_shift(mel)

            if np.random.rand() < self.p_specaug:
                mel = self._spec_augment(mel)

            if np.random.rand() < self.p_gain:
                g = np.random.uniform(0.8, 1.2)
                mel = mel * g

            if np.random.rand() < self.p_noise:
                # noise scaled to mel energy
                scale = self.noise_std * (mel.std() + 1e-6)
                mel = mel + np.random.randn(*mel.shape).astype(np.float32) * scale
        # ------------------------------------

        mel = torch.from_numpy(mel).unsqueeze(0)  # (1,128,target_T)
        y = int(self.labels[i])
        return mel, torch.tensor(y, dtype=torch.long)
