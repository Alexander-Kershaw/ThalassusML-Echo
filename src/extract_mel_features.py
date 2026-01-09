from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from termcolor import colored

"""
Extracting features from the oceanographic acoustic sample clips using Log-Mel 
spectrograms (Mel spectrograms converted to a logarithmic scale) since it is pretty industry standard with good interpetibility and
compatibility with covolutional neural networks (CNNs) which is what is generally
used for audio machine learning
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]

META_IN = PROJECT_ROOT / "data/metadata.csv"
OUT_DIR = PROJECT_ROOT / "data/features/mel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_OUT = PROJECT_ROOT / "data/metadata_with_features.csv"

# Feature parameters
SR = 16000
N_FFT = 1024
HOP = 320
N_MELS = 64
FMIN = 0
FMAX = 8000

# Leverage librosa to obtain Log-Mel spectrograms
def extract_logmel(wav_path: Path):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max) # convert to logarithmic scale
    return logmel.astype(np.float32)

def main():
    meta = pd.read_csv(META_IN)

    feature_rows = []

    for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting features"):
        wav_path = Path(row["path"])
        if not wav_path.exists():
            continue

        logmel = extract_logmel(wav_path)

        # save feature
        feat_path = OUT_DIR / f"{wav_path.stem}.npy"
        np.save(feat_path, logmel)

        feature_rows.append({
            **row,
            "feature_path": str(feat_path),
            "feat_type": "logmel",
            "n_mels": N_MELS,
            "n_frames": logmel.shape[1],
        })

    out_meta = pd.DataFrame(feature_rows)
    out_meta.to_csv(META_OUT, index=False)

    print(colored("Extracted features (Log-Mel Spectrograms)", "light_green"))
    print("Features written:", len(out_meta))
    print("Feature dir:", OUT_DIR.resolve())
    print("Metadata:", META_OUT.resolve())

if __name__ == "__main__":
    main()
