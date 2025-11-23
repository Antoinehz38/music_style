import os
import pandas as pd
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset

class FMASmallDataset(Dataset):
    def __init__(self, audio_root, metadata_root, split="training",
                 sample_rate=22050, duration=30.0):
        self.audio_root = audio_root
        self.sr = sample_rate
        self.num_samples = int(sample_rate * duration)

        tracks = pd.read_csv(os.path.join(metadata_root, "tracks.csv"), header=[0,1], index_col=0)

        subset = tracks[("set", "subset")] == "small"
        split_mask = tracks[("set", "split")] == split
        df = tracks[subset & split_mask]

        self.genre_ids = df[("track", "genre_top")].astype("category")
        self.labels = self.genre_ids.cat.codes
        self.classes = list(self.genre_ids.cat.categories)
        self.n_classes = len(self.classes)

        self.track_ids = df.index.tolist()
        self.files = [self._id_to_path(tid) for tid in self.track_ids]

        kept = [(tid, f, lab)
                for tid, f, lab in zip(self.track_ids, self.files, self.labels)
                if os.path.isfile(f)]

        self.track_ids, self.files, self.labels = zip(*kept)
        self.track_ids = list(self.track_ids)
        self.files = list(self.files)
        self.labels = pd.Series(self.labels, index=self.track_ids)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=1024, hop_length=512, n_mels=128
        )

    def _id_to_path(self, tid):
        tid_str = f"{tid:06d}"
        return os.path.join(self.audio_root, tid_str[:3], tid_str + ".mp3")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        y = int(self.labels.iloc[i])

        # --- FIX 1 : ignorer MP3 cassés ---
        try:
            wav, sr = librosa.load(path, sr=self.sr, mono=True)
        except Exception:
            return self.__getitem__((i + 1) % len(self.files))

        wav = torch.tensor(wav).unsqueeze(0)  # [1, n]

        # --- pad/trim ---
        n = wav.size(1)
        if n < self.num_samples:
            wav = torch.nn.functional.pad(wav, (0, self.num_samples - n))
        else:
            wav = wav[:, :self.num_samples]

        mel = self.mel(wav).squeeze(0)
        mel = torch.log(mel + 1e-6)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = mel.unsqueeze(0)

        return mel, torch.tensor(y, dtype=torch.long)

