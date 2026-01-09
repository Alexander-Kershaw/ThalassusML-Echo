from pathlib import Path
import zipfile

ZIP_PATH = Path("/Users/alexanderjameskershaw/.cache/huggingface/hub/datasets--peng7554--DS3500/snapshots/b654ef17b317722f98fc6b22c41480913ea7c7dc/DS3500.zip")

OUT_ROOT = Path("data/raw/ds3500_limited")
ANTH_DIR = OUT_ROOT / "anthropogenic"
AMB_DIR = OUT_ROOT / "ambient"
ANTH_DIR.mkdir(parents=True, exist_ok=True)
AMB_DIR.mkdir(parents=True, exist_ok=True)

# Target of 2000 audio samples for BOTH ambient and anthropogenic classifications
TARGET = 2000

def class_id_from_filename(filename: str):
    try:
        return int(Path(filename).name.split("_", 1)[0])
    except Exception:
        return None

def main():
    counts = {"anthropogenic": 0, "ambient": 0} # Want 2000 each (can source imbalance from other datasets)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for info in z.infolist():
            name = info.filename
            if not name.lower().endswith(".wav"):
                continue

            cid = class_id_from_filename(name)
            if cid is None:
                continue

            if cid == 4:
                label = "ambient"
            elif cid in (0, 1, 2, 3):
                label = "anthropogenic"
            else:
                continue

            if counts[label] >= TARGET:
                if counts["anthropogenic"] >= TARGET and counts["ambient"] >= TARGET:
                    break
                continue

            # Reads bytes from the zip and write them out unchanged
            data = z.read(name)

            if label == "ambient":
                out = AMB_DIR / f"ambient_{counts[label]:06d}.wav"
            else:
                out = ANTH_DIR / f"anthropogenic_{counts[label]:06d}.wav"

            out.write_bytes(data)
            counts[label] += 1

    print("Done.")
    print("Saved:", counts)
    print("Output:", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
