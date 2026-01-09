from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

"""
For reproducibility I am creating a CSV with a split column and to make sure
classes are calanced across splits

Performing a stratified train test validation split so sets have the same
proportions of classes as the original main dataset preserving the established
class balancing

OUTPUT:

Counts by label (total):
label
anthropogenic    1999
ambient           924
seismic           300
biological        294
Name: count, dtype: int64

Counts by label (train):
label
anthropogenic    1599
ambient           739
seismic           240
biological        235
Name: count, dtype: int64

Counts by label (val):
label
anthropogenic    400
ambient          185
seismic           60
biological        59
Name: count, dtype: int64


"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_IN = PROJECT_ROOT / "data/metadata_with_features.csv"
META_OUT = PROJECT_ROOT / "data/metadata_with_split.csv"

VAL_FRACTION = 0.2 # 20 percent of data reserved for validation set
SEED = 42 # for reproducibility

def main():
    df = pd.read_csv(META_IN)

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=df["label"],
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"

    # Print counts so we can verify stratification
    print("Counts by label (total):")
    print(df["label"].value_counts())

    print("\nCounts by label (train):")
    print(df[df["split"] == "train"]["label"].value_counts())

    print("\nCounts by label (val):")
    print(df[df["split"] == "val"]["label"].value_counts())

    df.to_csv(META_OUT, index=False)
    print("\nWrote:", META_OUT.resolve())

if __name__ == "__main__":
    main()
