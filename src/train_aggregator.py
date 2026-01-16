import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.models import ScanClassifier
from utils.datasets import ScanDataset
from utils.aggregator_training_functions import train_model, evaluate

BATCH_SIZE = 4
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
ALPHA_LOC = 0.7  # weight for localization loss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FUSION_OUT_DIM = 128  # should match model fusion_out_dim


def list_scan_files(data_paths: list[Path]) -> list[Path]:

    scan_files = []
    for data_path in data_paths:
        scan_files.extend(list(data_path.glob("*.npz")))

    if len(scan_files) == 0:
        raise ValueError(f"No .npz files found in {data_paths}")

    return sorted(scan_files)


def identity_collate(batch):
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agg_data", type=Path, help="Path to patches folder")
    # Optional experient name for output folder
    parser.add_argument(
        "experiment_name",
        nargs="?",
        default=None,
        type=str,
        help="Experiment name to append to timestamp (optional)",
    )

    args = parser.parse_args()

    data_path = args.agg_data
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"

    train_scans = list_scan_files([train_path])
    val_scans = list_scan_files([val_path])
    test_scans = list_scan_files([test_path])

    ### TESTING
    # train_scans = train_scans[: (len(train_scans) // 10)]
    # val_scans = val_scans[: (len(val_scans) // 10)]
    # test_scans = test_scans[: (len(test_scans) // 10)]
    # NUM_EPOCHS = 2
    ###

    train_dataset = ScanDataset(train_scans)
    val_dataset = ScanDataset(val_scans)
    test_dataset = ScanDataset(test_scans)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=identity_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=identity_collate,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=identity_collate,
        num_workers=4,
    )

    model = ScanClassifier(embed_dim=FUSION_OUT_DIM).to(DEVICE)

    loss_pres_fn = nn.BCEWithLogitsLoss()
    loss_loc_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ### Output directory (timestamp + optional experiment name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.experiment_name:
        timestamp += f"_{args.experiment_name}"
    output_dir = Path("output/aggregator/") / f"{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_weights, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_pres_fn=loss_pres_fn,
        loss_loc_fn=loss_loc_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        patience=5,
        device=DEVICE,
        output_dir=output_dir,
        alpha_loc=ALPHA_LOC,
    )

    model.load_state_dict(model_weights)

    (
        loss,
        loss_pres,
        loss_loc,
        weighted_auc,
        presence_auc,
        localization_auc,
    ) = evaluate(model, val_loader, loss_pres_fn, loss_loc_fn, DEVICE, ALPHA_LOC)

    pd.DataFrame.from_dict(
        {
            "loss": [loss],
            "loss_pres": [loss_pres],
            "loss_loc": [loss_loc],
            "weighted_auc": [weighted_auc],
            "presence_auc": [presence_auc],
            "localization_auc": [localization_auc],
        }
    ).to_csv(output_dir / "test_metrics.csv", index=False)

    print(f"Test Loss:              {loss:.4f}")
    print(f"Test Weighted AUROC:    {weighted_auc:.4f}")
    print(f"Test Presence AUC:      {presence_auc:.4f}")
    print(f"Test Localization AUC:  {localization_auc:.4f}")
