from pathlib import Path
import zipfile
from collections import Counter

ZIP_PATH = Path("/Users/alexanderjameskershaw/.cache/huggingface/hub/datasets--peng7554--DS3500/snapshots/b654ef17b317722f98fc6b22c41480913ea7c7dc/DS3500.zip")

def class_id_from_filename(filename: str):
    try:
        return int(Path(filename).name.split("_", 1)[0])
    except Exception:
        return None

def main():
    counts = Counter()

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for info in z.infolist():
            if not info.filename.lower().endswith(".wav"):
                continue
            cid = class_id_from_filename(info.filename)
            if cid is not None:
                counts[cid] += 1

    print("WAV count by class id:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

if __name__ == "__main__":
    main()


"""
Output:

WAV count by class id:
  0: 369
  1: 301
  2: 843
  3: 486
  4: 224


Need to source another 1776 ambient clips (since class lable 4 only has 224) and I
want balanced classes (classes 0-3 are fine as they are anthropogenic so
I am going to keep them grouped under the single label anthropogenic without any 
deeper sub-classification)

To source more ambient sounds I would have to look elsewhere, perhaps 
longer hydrophone recordings to have the additioinal 1776 samples as this dataset
by default is rather unbalanced
"""