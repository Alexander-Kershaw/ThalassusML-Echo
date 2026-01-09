from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
from obspy import read
from scipy.signal import resample_poly
from tqdm import tqdm
from termcolor import colored


IN_FILE = Path(__file__).resolve().parents[1] / "data/raw/seismic/seismic_30min_IU_ANMO_BHZ.mseed"
# 30 min seismic waveform file + fix relative path issues

OUT_DIR = Path("data/clips/seismic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = Path("data/metadata_seismic.csv")
META_PATH.parent.mkdir(parents=True, exist_ok=True)

# Target sample rate and individual clip duration
TARGET_SR = 16000
CLIP_S = 5.0
CLIP_LEN = int(TARGET_SR * CLIP_S)

TARGET_CLIPS = 300  # target number of inidivdual seismic clips

def main():
    st = read(str(IN_FILE))
    st.merge(fill_value="interpolate")
    tr = st[0]

    x = tr.data.astype(np.float32)
    sr_in = float(tr.stats.sampling_rate)

    # Undoing the mean average 
    x = x - np.mean(x)

    # 2) Normalize (stops exceedingly high ampitude seismic events from dominating)
    scale = np.percentile(np.abs(x), 95)
    if scale > 0:
        x = x / scale
    x = np.clip(x, -1.0, 1.0)

    # Upsample from sr_in to TARGET_SR using polyphase resampling
    # This should be superior to standard interpolation
    up = int(TARGET_SR)
    down = int(sr_in)
    y = resample_poly(x, up, down).astype(np.float32)

    # Normalize again after resampling
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0:
        y = y / peak
    y = np.clip(y, -1.0, 1.0)

    # Partition into 5-second non-overlapping distinct clips
    n_total = len(y) // CLIP_LEN
    n_use = min(n_total, TARGET_CLIPS)

    rows = []
    for i in tqdm(range(n_use), desc="Writing seismic clips"):
        clip = y[i * CLIP_LEN : (i + 1) * CLIP_LEN]
        out_path = OUT_DIR / f"seismic_{i:04d}.wav"
        sf.write(out_path, clip, TARGET_SR)

        rows.append({
            "path": str(out_path),
            "label": "seismic",
            "source": str(IN_FILE),
            "sr": TARGET_SR,
            "duration_s": CLIP_S,
            "clip_index": i,
            "origin_sampling_rate_hz": sr_in,
        })

    pd.DataFrame(rows).to_csv(META_PATH, index=False)

    print(colored("Seismic sample clipping complete", "light_green"))
    print("Seismic clips written:", n_use, "out of possible", n_total)
    print("Metadata:", META_PATH.resolve())
    print("Output dir:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()