from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

"""
I have data extracted in both src and in the project root data directories so to 
have a complete dataset of acoustic samples this needs to be unified while preserving
the class categories (anthropogenic, bioligical, seismic, ambient)

A metadata folder to track the consistency of the samples is also to be implemented
"""
CLIP_ROOTS = [
    PROJECT_ROOT / "data/clips",
    PROJECT_ROOT / "src/data/clips",
]

OUT_CLIPS = PROJECT_ROOT / "data/clips"
OUT_CLIPS.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_all_wavs(src_dir: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    n = 0
    for wav in src_dir.glob("*.wav"):
        dst = dst_dir / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst)
        n += 1
    return n

def build_metadata_from_folder(folder: Path, label: str, source: str):
    rows = []
    for wav in sorted(folder.glob("*.wav")):
        rows.append({
            "path": str(wav),
            "label": label,
            "source": source
        })
    return pd.DataFrame(rows)

def main():
    # Consolidate WAVs into data/clips/<class>/ to preserve categories
    classes = ["anthropogenic", "ambient", "biological", "seismic"]
    for cls in classes:
        ensure_dir(OUT_CLIPS / cls)

    copied = {c: 0 for c in classes}

    for root in CLIP_ROOTS:
        if not root.exists():
            continue
        for cls in classes:
            src = root / cls
            if src.exists():
                copied[cls] += copy_all_wavs(src, OUT_CLIPS / cls)

    print("Copied wav counts by class folder:", copied)

    # 2) Build/merge metadata
    meta_parts = []

    # Biological metadata file 
    bio_meta = PROJECT_ROOT / "data/metadata_biological.csv"
    if bio_meta.exists():
        df = pd.read_csv(bio_meta)
        # rewrite paths to canonical data/clips/biological if needed
        df["path"] = df["path"].apply(lambda p: p.replace(str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))
        meta_parts.append(df[["path", "label", "source_file", "augment", "sr", "duration_s"]].copy())
    else:
        # fallback to inference
        meta_parts.append(build_metadata_from_folder(OUT_CLIPS / "biological", "biological", "biological_unknown"))

    # Seismic metadata file 
    seismic_meta_candidates = [
        PROJECT_ROOT / "data/metadata_seismic.csv",
        PROJECT_ROOT / "src/data/metadata_seismic.csv",
    ]
    seismic_meta = next((p for p in seismic_meta_candidates if p.exists()), None)
    if seismic_meta:
        df = pd.read_csv(seismic_meta)
        df["path"] = df["path"].apply(lambda p: p.replace(str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))
        meta_parts.append(df.copy())
    else:
        meta_parts.append(build_metadata_from_folder(OUT_CLIPS / "seismic", "seismic", "seismic_unknown"))

    # Anthropogenic + Ambient: infer from folders (extracted from ds3500)
    meta_parts.append(build_metadata_from_folder(OUT_CLIPS / "anthropogenic", "anthropogenic", "ds3500"))
    meta_parts.append(build_metadata_from_folder(OUT_CLIPS / "ambient", "ambient", "ds3500_and_hydrophones"))

    # Merge everything
    meta = pd.concat(meta_parts, ignore_index=True)

    # Keep only rows whose file exists so everyhing is clean
    meta["exists"] = meta["path"].apply(lambda p: Path(p).exists())
    meta = meta[meta["exists"]].drop(columns=["exists"])

    out_meta = PROJECT_ROOT / "data/metadata.csv"
    meta.to_csv(out_meta, index=False)

    # Final counts summary for final look at class balance before applying any ML
    print("\nFinal class counts from metadata.csv:")
    print(meta["label"].value_counts())

    print("\nWrote:", out_meta.resolve())

if __name__ == "__main__":
    main()
