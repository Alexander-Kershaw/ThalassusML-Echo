
***
# ThalassaML-Echo — Oceanographic Acoustics Classification
***

ThalassalML-Echo is a ML project to build a practical underwater acoustic classifier using public data and a reproducible pipeline.

We classify 5-second audio clips into 4 coarse classes:
1. **Anthropogenic** (man-made: ships/boats, synthetic/machinery-like noise)
2. **Ambient** (background ocean/hydrophone noise)
3. **Biological** (marine mammal calls; whale-call clips augmented deterministically)
4. **Seismic** (earthquake-related waveform segments converted into audio-adjacent clips)

This project focuses on:
- building a dataset with controlled download sizes from multiple distinct and divergent sources,
- standardizing audio consistently,
- establishing a strong baseline model,
- improving errors via targeted augmentation.

---

## Results (validation set)

After adding light SpecAugment during training:

- **Accuracy:** ~0.97  
- **Macro F1:** ~0.98  
- Very strong separation across all classes.

Confusion matrix (rows=true, cols=pred):
- Ambient: 173 correct, 11 -> anthropogenic, 1 -> biological
- Anthropogenic: 391 correct, 9 -> ambient
- Biological: 59 correct
- Seismic: 60 correct

Important takeaway:
- The biggest early failure mode was **anthropogenic being misclassified as ambient** (distant/low-SNR ship noise looks like background).
- **SpecAugment** fixed this by forcing the CNN to learn more robust time-frequency patterns.

---

## Data sources and their use

### Anthropogenic + small Ambient (benchmark ship dataset)
- **DS3500 / ShipsEar family on Hugging Face**
- Extracted approximately 2000 anthropogenic clips and the available ambient subset (approx 224).
- Benchmark datasets are often class-imbalanced; DS3500 had limited ambient samples.

### Ambient data supplementation (hydrophone field recordings)
- Downloaded a few long hydrophone recordings (MP3) and clipped them into 5-second segments.
- Resampled to 16 kHz, mono, robust-normalized.
- Used to bring ambient to a practical target (approx 900).

### Biological (whale calls + deterministic augmentation)
- A small whale-call WAV collection (~49 files, ~1–4 sec each at 40 kHz).
- Standardized each to 5 seconds at 16 kHz mono via center padding.
- Created deterministic augment variants (gain up/down, slight shift, tiny noise) to reach ~294 biological clips.

### Seismic (IRIS FDSN via ObsPy)
- Downloaded a 30-minute waveform segment (IU.ANMO.00.BHZ) around the 2011 Tohoku event time window.
- Demean + robust normalize, resample 20 Hz → 16 kHz (polyphase), then cut into 5-second clips.
- Capped at ~300 seismic clips.

---

## Project structure

Typical layout:

```text
ThalassaML/
data/
raw/ # raw downloads (not committed)
clips/
anthropogenic/
ambient/
biological/
seismic/
features/
mel/ # .npy log-mel spectrograms
metadata.csv
metadata_with_features.csv
metadata_with_split.csv
models/
cnn_baseline.pt
src/
...scripts...
requirements.txt
README.md
.gitignore
```


---

## Pipeline

### 1 Standardize audio
All training examples are standardized to:
- **mono**
- **16 kHz**
- **5.0 seconds**

### 2 Extract features (log-mel)
For each clip:
- Compute mel spectrogram (`64` mel bins)
- Convert to log dB
- Saved as `.npy` with consistent shape `(64, 251)`

### 3 Train/test split
- Stratified train/val split (80/20) for reproducibility.

### 4 CNN baseline
- Simple 2D CNN on log-mel "images"
- CrossEntropy with class weights
- Added light SpecAugment in training for robustness

---

## How to run

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```
***

### Feature extraction

```bash
python src/extract_mel_features.py
python src/make_split.py
```

***

### Train and evaluate

```bash
python src/train_cnn_baseline.py
python src/eval_cnn_baseline.py
```


---

## Notes on reproducibility

- We intentionally avoided fragile audio decoding dependencies (torchcodec/ffmpeg) where possible.

- Hydrophone MP3 decoding depends on your local audio stack; if MP3 decode fails elsewhere:

- convert MP3 → WAV externally, or install ffmpeg system-wide and re-run.


---

### Future extensions


- **CRNN / temporal modeling** : Add GRU/LSTM after CNN to better capture long-term patterns (especially weak ship tonals).

- **Open-set detection** : Treat unknown anthropogenic sounds as "unknown" and detect out-of-distribution examples.

- **Better ambient realism** : Add more diverse background conditions (wind, rain, distant shipping lanes) to reduce dataset bias.

- **Event-based evaluation** : Move from clip classification to stream evaluation: false alarms per hour, missed detections.

- **Domain adaptation** : Train on one region’s data, test on another to measure generalization.

***

