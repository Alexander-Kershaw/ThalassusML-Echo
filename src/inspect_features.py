from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

"""
Load metadate with features, picking one random example per class, printing feature
shape and plots the log-mel spectrogram just to view the form the features are taking
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META = PROJECT_ROOT / "data/metadata_with_features.csv"

def show_one_per_class():
    df = pd.read_csv(META)

    # Selecting one example per class as a demonstration
    classes = df["label"].unique().tolist()

    for cls in classes:
        row = df[df["label"] == cls].iloc[0]
        feat_path = Path(row["feature_path"])
        x = np.load(feat_path)

        print(colored(f"\nClass: {cls}", "light_blue"))
        print("  feature:", feat_path.name)
        print("  shape:", x.shape)
        print("  min/max:", float(x.min()), float(x.max()))

        plt.figure()
        plt.imshow(x, aspect="auto", origin="lower")
        plt.title(f"log-mel spectrogram — {cls}")
        plt.xlabel("time frames")
        plt.ylabel("mel bins")
        plt.colorbar()
        plt.show()

def main():
    show_one_per_class()

if __name__ == "__main__":
    main()


"""
Class: biological
  feature: 62018001__orig__0000.npy
  shape: (64, 251)
  min/max: -80.0 0.0
2026-01-09 16:16:57.269 python[63934:3704359] +[IMKClient subclass]: chose IMKClient_Modern
2026-01-09 16:16:57.269 python[63934:3704359] +[IMKInputSession subclass]: chose IMKInputSession_Modern

Class: seismic
  feature: seismic_0000.npy
  shape: (64, 251)
  min/max: -80.0 0.0

Class: anthropogenic
  feature: anthropogenic_000000.npy
  shape: (64, 251)
  min/max: -53.39239501953125 0.0

Class: ambient
  feature: ambient_000000.npy
  shape: (64, 251)
  min/max: -51.2577018737793 0.0


  
  All classes have the identical shape these are ready for train validation test splits
"""