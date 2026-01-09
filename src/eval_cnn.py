from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META = PROJECT_ROOT / "data/metadata_with_split.csv"
CKPT = PROJECT_ROOT / "models/cnn_baseline.pt"

BATCH_SIZE = 64

class MelDataset(Dataset):
    def __init__(self, df, classes):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.class_to_id = {c:i for i,c in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["feature_path"])
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = self.class_to_id[row["label"]]
        return x, y

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main():
    ckpt = torch.load(CKPT, map_location="cpu")
    classes = ckpt["label_classes"]

    df = pd.read_csv(META)
    val_df = df[df["split"] == "val"].copy()

    ds = MelDataset(val_df, classes)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(n_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())

    print("Classes:", classes)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=3
    ))

if __name__ == "__main__":
    main()


"""
OUTPIT: 

Classes: ['ambient', 'anthropogenic', 'biological', 'seismic']

Confusion matrix (rows=true, cols=pred):
[[184   1   0   0]
 [118 276   6   0]
 [  0   0  59   0]
 [  0   0   0  60]]

Classification report:
               precision    recall  f1-score   support

      ambient      0.609     0.995     0.756       185
anthropogenic      0.996     0.690     0.815       400
   biological      0.908     1.000     0.952        59
      seismic      1.000     1.000     1.000        60

     accuracy                          0.822       704
    macro avg      0.878     0.921     0.881       704
 weighted avg      0.888     0.822     0.827       704


 The model is excellent at classifying seismic, biological and ambient data correctly
 
 However, it stugggles with the correct classification of anthropogenic samples. 
 Therefore, ship noises tend to be mistaken as ambient background noise

 This is common for oceanographic acoustic classification since the recordings of ships 
 can vary significantly and provide stonger and weaker signals depending on:

 - how distant the ship is from a hydrophone (attenuated signal)
 - tonal noises get interfered by propagation

When the model assumes a ship, it is almost always correct, but misclassifies some
ship noises as being ambient instead








AFTER ADDING SpecAugment to train_cnn.py:

Confusion matrix (rows=true, cols=pred):
[[173  11   1   0]
 [  9 391   0   0]
 [  0   0  59   0]
 [  0   0   0  60]]

Classification report:
               precision    recall  f1-score   support

      ambient      0.951     0.935     0.943       185
anthropogenic      0.973     0.978     0.975       400
   biological      0.983     1.000     0.992        59
      seismic      1.000     1.000     1.000        60

     accuracy                          0.970       704
    macro avg      0.977     0.978     0.977       704
 weighted avg      0.970     0.970     0.970       704



 After this improvement the confusion between ambient and anthropogenic signals are now
 significantly less frequent as the model learnt structure of spectrographic energies

 This is a much more robust classifier model as a result


 EXPLAINATION:


SpecAugment essentially obscures random time and frequency regions in the mel spectrograms
thus preventing the CNN from being disporportionately influenced by high ampitude signals
single frequency tones and such

This foced the CNN to learn disributed patterns which made it more resilient to
the misclassification of anthropogenic signals a ambient because they far away
(distance from hydrophone causing signal attenuation), and tonality (meaning
temporal persistance of spectrographic patterns matters more than the power of a signal)
"""