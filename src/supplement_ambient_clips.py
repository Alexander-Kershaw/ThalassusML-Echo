from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm


"""
To augment the number of ambient acoustics samples I am going to partition a long
hydrophone ambient recording into 5 second long segments to supplement sample sizes and
reduce class imbalances
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_DIR = PROJECT_ROOT / "data/raw/ambient_long"
OUT_DIR = PROJECT_ROOT / "data/clips/ambient"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = PROJECT_ROOT / "data/metadata_ambient_added.csv"

TARGET_SR = 16000
CLIP_S = 5.0
CLIP_LEN = int(TARGET_SR * CLIP_S)

TARGET_NEW = 700  # number new ambient clips to add

def load_resample_mono(path: Path):
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1).astype(np.float32)  # stereo to mono

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # normalization
    scale = np.percentile(np.abs(y), 95)
    if scale > 0:
        y = y / scale
    y = np.clip(y, -1.0, 1.0)
    return y

def main():
    mp3s = sorted(IN_DIR.glob("*.mp3"))
    if not mp3s:
        raise RuntimeError(f"No MP3s found in {IN_DIR}")

    rows = []
    out_count = 0

    for mp3 in mp3s:
        y = load_resample_mono(mp3)
        n_clips = len(y) // CLIP_LEN

        for i in range(n_clips):
            if out_count >= TARGET_NEW:
                break

            clip = y[i * CLIP_LEN:(i + 1) * CLIP_LEN]
            out_path = OUT_DIR / f"ambient_hydro_{out_count:04d}.wav"
            sf.write(out_path, clip, TARGET_SR)

            rows.append({
                "path": str(out_path),
                "label": "ambient",
                "source": str(mp3),
                "sr": TARGET_SR,
                "duration_s": CLIP_S
            })

            out_count += 1

        if out_count >= TARGET_NEW:
            break

    pd.DataFrame(rows).to_csv(META_PATH, index=False)

    print("Done.")
    print("New ambient clips written:", out_count)
    print("Metadata:", META_PATH.resolve())
    print("Output dir:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
