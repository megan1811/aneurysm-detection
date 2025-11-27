import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pydicom
from sklearn.model_selection import train_test_split

import ast
from utils.preprocess import dicom_pixel_to_world

DATA_DIR = Path("/Volumes/T7 Shield/aneurysm-detection-data")


def compute_world_coordinates(df_loc: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 3D patient-space (LPS) world coordinates for each ground truth point
    using the corresponding DICOM slice metadata. Missing files or slices without
    required spatial tags are skipped.

    Args:
        df_loc (pd.DataFrame): DataFrame containing aneurism localizations.

    Returns:
        pd.DataFrame: Input DataFrame with added `l`, `p`, and `s` columns
            containing world coordinates in millimeters (NaN when unavailable).
    """
    df_loc[["l", "p", "s"]] = np.nan

    missing_data = 0
    coordinate_issue = 0
    for index, row in tqdm(
        df_loc.iterrows(), desc="Computing world coordinates", total=len(df_loc)
    ):
        series_id = row["SeriesInstanceUID"]
        instance_uid = row["SOPInstanceUID"]
        path = DATA_DIR / "series" / series_id / f"{instance_uid}.dcm"
        if not path.exists():
            # print(f"File not found: {path}")
            missing_data += 1
            continue

        ds = pydicom.dcmread(path)

        coord = ast.literal_eval(row.coordinates)
        try:
            world_coord = dicom_pixel_to_world(ds, coord["x"], coord["y"])
        except ValueError as e:
            # print(e)
            coordinate_issue += 1
            continue

        # update DataFrame by index
        df_loc.at[index, "l"] = world_coord[0]
        df_loc.at[index, "p"] = world_coord[1]
        df_loc.at[index, "s"] = world_coord[2]

    print(f"\nMissing data files: {missing_data}")
    print(f"Coordinate conversion issues: {coordinate_issue}")
    print(
        f"Total coordinates computed: {len(df_loc) - missing_data - coordinate_issue}"
    )
    return df_loc


def train_test_val_split(df_series: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/val/test split labels to ground truth series.

    Args:
        df_series (pd.DataFrame): DataFrame of series entries.

    Returns:
        pd.DataFrame: Input DataFrame with a new `split` column
            containing "train", "val", or "test".
    """
    df_val, df_test = train_test_split(
        df_series.sample(frac=0.4), test_size=0.6, random_state=42, shuffle=True
    )

    df_series["split"] = "train"
    df_series.loc[df_val.index, "split"] = "val"
    df_series.loc[df_test.index, "split"] = "test"

    return df_series


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", type=Path, help="Path to data folder")
    args = parser.parse_args()

    # Read in series and localizer CSVs
    loc_path = args.csv_dir / "train_localizers.csv"
    series_path = args.csv_dir / "train.csv"
    df_series = pd.read_csv(series_path)
    df_loc = pd.read_csv(loc_path)

    # Compute world coordinates for localizer points
    df_loc = compute_world_coordinates(df_loc)

    # Drop Series where we weren't able to compute world coordinates
    mask_drop = df_loc[["l", "p", "s"]].isna().any(axis=1)
    series_uids_drop = df_loc.loc[mask_drop, "SeriesInstanceUID"].unique()
    df_loc = df_loc.loc[~mask_drop].reset_index(drop=True)
    df_series = df_series.loc[
        ~df_series["SeriesInstanceUID"].isin(series_uids_drop)
    ].reset_index(drop=True)

    # Generate train/val/test split
    df_series = train_test_val_split(df_series)

    # Report split distribution
    counts = df_series["split"].value_counts()
    fractions = df_series["split"].value_counts(normalize=True)
    print("\nSplit distribution:")
    for split in counts.index:
        print(f"{split:5s}: {counts[split]:4d} ({fractions[split]*100:.1f}%)")

    # Save updated CSVs
    df_loc.to_csv(args.csv_dir / "train_localizers_processed.csv", index=False)
    df_series.to_csv(args.csv_dir / "train_processed.csv", index=False)
