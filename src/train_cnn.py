from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from termcolor import colored

"""
First model here will be a simple CNN that is relatively fast and implemented with
class weighted loss to remedy the underlying class impalances of the data
"""

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
META = PROJECT_ROOT / "data/metadata_with_split.csv"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Dataset
class MelDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder
        self.training = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["feature_path"])
        x = torch.tensor(x, dtype=torch.float32)

        """
        Improvement:

        time and frequency masking: force model to rely on structure not energy,
        should help distingish weak anthropogenic signals from true ambient noise

        Time masking: masking up to 10 percent of the time
        Frequency masking: masking up to 10 percent of mel spectrogram bins

        Includes repeatable deterministic seed
        
        """

        # SpecAugment 
        if self.training:
            rng = torch.Generator().manual_seed(idx)

            # Time mask
            T = x.shape[1]
            t = int(0.1 * T)
            if t > 0:
                t0 = torch.randint(0, T - t, (1,), generator=rng).item()
            x[:, t0:t0+t] = 0

            # Frequency mask
            F = x.shape[0]
            f = int(0.1 * F)
            if f > 0:
                f0 = torch.randint(0, F - f, (1,), generator=rng).item()
                x[f0:f0+f, :] = 0

        x = x.unsqueeze(0)

        y = self.le.transform([row["label"]])[0]
        return x, y

# A basic CNN model
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

# Train and evaluation
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

# Main
def main():
    df = pd.read_csv(META)

    # Encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    print(colored("Label mapping:", "light_blue"))
    for i, c in enumerate(le.classes_):
        print(f"  {i}: {c}")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    # Class weight
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label_id"]),
        y=train_df["label_id"],
    )
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print("\nClass weights:", class_weights.tolist())

    train_ds = MelDataset(train_df, le)
    train_ds.training = True

    val_ds = MelDataset(val_df, le)
    val_ds.training = False


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = SimpleCNN(n_classes=len(le.classes_))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(colored(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.3f}", "light_yellow"))

    # Save model
    out = PROJECT_ROOT / "models/cnn_baseline.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_classes": le.classes_.tolist(),
        },
        out,
    )
    print("\nSaved model to:", out.resolve())

if __name__ == "__main__":
    main()


"""
OUTPUTS:

Label mapping:
  0: ambient
  1: anthropogenic
  2: biological
  3: seismic

Class weights: [0.951623797416687, 0.43980613350868225, 2.992553234100342, 2.930208444595337]
Epoch 01 | Train loss 0.6958, acc 0.651 | Val loss 0.4957, acc 0.838
Epoch 02 | Train loss 0.3556, acc 0.891 | Val loss 0.2816, acc 0.923
Epoch 03 | Train loss 0.2486, acc 0.915 | Val loss 0.2479, acc 0.920
Epoch 04 | Train loss 0.2107, acc 0.922 | Val loss 0.2095, acc 0.930
Epoch 05 | Train loss 0.1787, acc 0.935 | Val loss 0.2539, acc 0.902
Epoch 06 | Train loss 0.1469, acc 0.952 | Val loss 0.1708, acc 0.960
Epoch 07 | Train loss 0.1145, acc 0.959 | Val loss 0.3028, acc 0.884
Epoch 08 | Train loss 0.1069, acc 0.960 | Val loss 0.1562, acc 0.957
Epoch 09 | Train loss 0.0919, acc 0.964 | Val loss 0.2645, acc 0.891
Epoch 10 | Train loss 0.1177, acc 0.952 | Val loss 0.4820, acc 0.791
Epoch 11 | Train loss 0.0986, acc 0.965 | Val loss 0.1945, acc 0.928
Epoch 12 | Train loss 0.0606, acc 0.976 | Val loss 0.1713, acc 0.928
Epoch 13 | Train loss 0.0550, acc 0.978 | Val loss 0.0582, acc 0.989
Epoch 14 | Train loss 0.0620, acc 0.974 | Val loss 0.2019, acc 0.923



Something to note:

validation accuracy spikes significantly at epoch 13 and dips dramatically at 
epochs 10, 9 and 7 which is indicative that the model is potentially overconfident in
its classification of minority classes. Performance per class evaluation is necessary.
"""