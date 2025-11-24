import numpy as np
import librosa
from pathlib import Path

def mel_128_norm_per_track(path, sr=22050, n_fft=2048, hop_length=512,
                           n_mels=128, fmin=20, fmax=None):
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mu = S_db.mean()
    sigma = S_db.std() + 1e-8
    return (S_db - mu) / sigma


if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parents[2]
    in_dir = ROOT / "data" / "fma_small"
    out_dir = ROOT / "data" / "mels128"
    out_dir.mkdir(parents=True, exist_ok=True)

    # récup liste audio
    audio_files = [p for p in in_dir.rglob("*")
                   if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]

    # récup liste npy existants (stems)
    existing = {p.stem for p in out_dir.glob("*.npy")}

    # filtrer fichiers restants
    todo = [p for p in audio_files if p.stem not in existing]

    print(f"Total audio: {len(audio_files)}")
    print(f"Déjà calculés: {len(existing)}")
    print(f"Restants à faire: {len(todo)}")

    for p in todo:
        try:
            mel = mel_128_norm_per_track(str(p))
            np.save(out_dir / f"{p.stem}.npy", mel.astype(np.float32))
        except Exception as e:
            print(f"skip {p}: {e}")
