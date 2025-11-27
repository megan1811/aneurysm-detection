import argparse
from pathlib import Path
import pandas as pd
import torch
import monai.transforms as MT
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from utils.CONSTANTS import CAT_COLS, MODALITIES
from utils.datasets import AneurysmPatchDataset
from utils.models import PatchClassifier
from utils.classifier_training_functions import train_model, eval

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
# EPOCHS = 2  # For testing purposes
LEARNING_RATE = 1e-4
ES_PATIENCE = 5
ES_FACTOR = 0.5
WEIGHT_DECAY = 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patches_dir", type=Path, help="Path to patches folder")
    # Optional experient name for output folder
    parser.add_argument(
        "experiment_name",
        nargs="?",
        default=None,
        type=str,
        help="Experiment name to append to timestamp (optional)",
    )
    args = parser.parse_args()

    ### Load patches data
    df_data = pd.read_csv(args.patches_dir / "patches_metadata.csv")

    # Optional debugging
    # df_data = df_data.sample(frac=0.01, random_state=42).reset_index(drop=True)
    # EPOCHS = 2

    ### Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ### Split into train/val/test
    df_train = df_data[df_data["split"] == "train"]
    df_val = df_data[df_data["split"] == "val"]
    df_test = df_data[df_data["split"] == "test"]

    ### Data Augmentation (train only)
    transforms = MT.Compose(
        [
            MT.RandGaussianNoised(keys=["patch"], prob=0.1, mean=0.0, std=0.1),
            MT.RandShiftIntensityd(keys=["patch"], prob=0.1, offsets=0.1),
            MT.RandAdjustContrastd(keys=["patch"], prob=0.1, gamma=(0.9, 1.2)),
            MT.Rand3DElasticd(
                keys=["patch"],
                prob=0.1,
                sigma_range=(2.0, 3.0),
                magnitude_range=(5, 15),
            ),
        ]
    )

    ### Create datasets and dataloaders
    train_dataset = AneurysmPatchDataset(
        df_data=df_train, test=False, transform=transforms
    )
    val_dataset = AneurysmPatchDataset(df_data=df_val, test=False)
    test_dataset = AneurysmPatchDataset(df_data=df_test, test=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    ### Model, loss, optimizer, schedule
    classifier = PatchClassifier(
        modality_num_classes=len(MODALITIES), num_classes=len(CAT_COLS) + 1
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=ES_PATIENCE, factor=ES_FACTOR
    )

    ### Output directory (timestamp + optional experiment name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.experiment_name:
        timestamp += f"_{args.experiment_name}"
    output_dir = Path("output/patch_classifier/") / f"{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Train model with early stoppping
    model_weights, metrics = train_model(
        model=classifier,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=ES_PATIENCE,
        output_dir=output_dir,
    )

    ### Final evaluation on test set
    # Load best model weights
    classifier.load_state_dict(model_weights)
    _, test_acc, balanced_test_acc = eval(classifier, test_loader, criterion, device)

    pd.DataFrame.from_dict(
        {"accuracy": [test_acc], "balanced_accuracy": [balanced_test_acc]}
    ).to_csv(output_dir / "test_metrics.csv", index=False)

    print(f"Test Accuracy:           {test_acc:.4f}")
    print(f"Test Balanced Accuracy:  {balanced_test_acc:.4f}")
