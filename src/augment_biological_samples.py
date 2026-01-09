from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm

"""
Due to the sheer lack of much acean acoustic data for biological classes avaliable open source
I am resorting to deterministic (reproducable rather than stochatic) audio sample augmentation
of the whale calls I have managed to acquire

This is not ideal as I would rather have a diverse reportouire of species encompassed in
the biological sample class since that would make predictive classifications more resilient
to a vaster degree of oceanographic acoustic climates
"""


# Inputs
IN_DIR = Path(__file__).resolve().parents[1] / "data/raw/biological"  # So far just whale sounds

# Outputs
OUT_DIR = Path("data/clips/biological") # back into clips folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = Path("data/metadata_biological.csv")
META_PATH.parent.mkdir(parents=True, exist_ok=True)

# Standardization (5 sec clip standard)
TARGET_SR = 16000 #16KHz sample rate
CLIP_S = 5.0 # 5 second long samples
CLIP_LEN = int(TARGET_SR * CLIP_S)

# Target number of biological acoustics recordings (still will be imbalanced)
TARGET_TOTAL = 300

# Deterministic augmentation:
# For each original clip, generate these variants in order.
AUGMENTS = [
    ("orig", 1.00, 0),    # name, gain, shift_samples
    ("gain_up", 1.25, 0),
    ("gain_dn", 0.80, 0),
    ("shift_fwd", 1.00, 800),   # shift by 800 samples = 0.05s at 16kHz
    ("shift_back", 1.00, -800),
    ("noise", 1.00, 0),
]

def load_resample_mono(path: Path):
    y, sr = sf.read(path, always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)

    # float32 type for librosa
    y = y.astype(np.float32)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # normalize to avoid extreme amplitudes
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0:
        y = y / peak

    return y

def pad_or_trim_to_5s(y: np.ndarray):
    if len(y) >= CLIP_LEN:
        # Center-crop to 5s should isolate the biological noise rather than 
        # Just registering the ambient sounds on either end of the noise
        start = (len(y) - CLIP_LEN) // 2
        return y[start:start + CLIP_LEN]
    else:
        # Center-pad with zeros
        pad_total = CLIP_LEN - len(y)
        left = pad_total // 2
        right = pad_total - left
        return np.pad(y, (left, right), mode="constant")

def apply_shift(y: np.ndarray, shift: int):
    if shift == 0:
        return y
    out = np.zeros_like(y)
    if shift > 0:
        out[shift:] = y[:-shift]
    else:
        s = -shift
        out[:-s] = y[s:]
    return out

def apply_noise(y: np.ndarray, level=0.01):
    # deterministic noise: seeded by fixed seed so it’s repeatable
    rng = np.random.default_rng(12345)
    noise = rng.normal(0, 1, size=len(y)).astype(np.float32)
    return np.clip(y + level * noise, -1.0, 1.0)

def main():
    wavs = sorted(IN_DIR.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files found in {IN_DIR.resolve()}")

    rows = []
    out_count = 0

    for wav in tqdm(wavs, desc="Biological files"):
        base = wav.stem
        y = load_resample_mono(wav)
        y5 = pad_or_trim_to_5s(y)

        for aug_name, gain, shift in AUGMENTS:
            if out_count >= TARGET_TOTAL:
                break

            ya = y5 * gain
            ya = np.clip(ya, -1.0, 1.0)

            ya = apply_shift(ya, shift)

            if aug_name == "noise":
                ya = apply_noise(ya, level=0.01)

            out_path = OUT_DIR / f"{base}__{aug_name}__{out_count:04d}.wav"
            sf.write(out_path, ya, TARGET_SR)

            rows.append({
                "path": str(out_path),
                "label": "biological",
                "source_file": str(wav),
                "augment": aug_name,
                "sr": TARGET_SR,
                "duration_s": CLIP_S
            })
            out_count += 1

        if out_count >= TARGET_TOTAL:
            break

    df = pd.DataFrame(rows)
    df.to_csv(META_PATH, index=False)

    print("Done.")
    print("Biological clips written:", out_count)
    print("Metadata:", META_PATH.resolve())
    print("Output dir:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
