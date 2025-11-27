import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import concurrent.futures
import uuid

from utils.preprocess import (
    load_and_preprocess,
    jitter_coords,
    extract_patch_sitk,
    patch_sitk_to_numpy,
)

# Patch extraction parameters
PATCH_SIZE = 32
POS_SAMPLES = 5
NEG_SAMPES = 15
MAX_JITTER = 10


def save_patch(patch_np: np.array, patch_filepath: Path):
    """
    Save a single patch to disk as a compressed ``.npz`` file.

    Args:
        patch_np (np.ndarray): Patch data as a NumPy array.
        patch_filepath (str or pathlib.Path): Destination file path.
    """
    np.savez_compressed(patch_filepath, patch=patch_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("series_dir", type=Path, help="Path to series folder")
    parser.add_argument("data_dir", type=Path, help="Path to data folder")
    args = parser.parse_args()

    ### Paths and metadata setup
    patch_dir = args.data_dir / "patches"
    patch_metadata_path = patch_dir / "patches_metadata.csv"

    ### Read in series and localization data
    df_loc = pd.read_csv(args.data_dir / "train_localizers_processed.csv")
    df_series = pd.read_csv(args.data_dir / "train_processed.csv")

    ### Load existing metadata if it already exists, to allow incremental patch generation
    if patch_metadata_path.exists():
        df_metadata = pd.read_csv(patch_metadata_path)
    else:
        df_metadata = pd.DataFrame(
            columns=[
                "patch_id",  # Unique patch identifier (UUID)
                "series_id",  # DICOM SeriesInstanceUID
                "modality",  # Imaging modality (CTA, MRA, etc.)
                "patch_filepath",  # Path to .npz file on disk
                "split",  # Data split ("train", "val", "test")
                "world_coords",  # Patch center in world coordinates (mm)
                "label",  # 1 = aneurysm, 0 = negative
                "location",  # Aneurysm location label (or NaN for negatives)
            ]
        )

    ### Loop over each series
    for i, row in tqdm(
        df_series.iterrows(),
        desc="Generating patches",
        total=(len(df_series)),
    ):
        series_id = row.SeriesInstanceUID
        modality = row.Modality
        split = row.split
        patch_rows = []
        save_tasks = []

        # Use a thread pool to offload disk writes (np.savez_compressed)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            ### Load Volume and basic preprocess / check
            try:
                sitk_img = load_and_preprocess(
                    series_id, modality, base_dir=args.series_dir
                )
            except Exception as e:
                print(f"Error loading series {series_id}: {e}")
                continue

            vol_size = sitk_img.GetSize()

            # Skip series that are too small to fit a full PATCH_SIZE^3 cube
            if any(vol_size[d] < PATCH_SIZE for d in range(3)):
                print(
                    f"Series {series_id}: volume too small for patch size {PATCH_SIZE}. Skipping."
                )
                continue

            df_aneurysms = df_loc[df_loc["SeriesInstanceUID"] == series_id]

            ### Positive (aneurysm-centered) patches
            for _, loc_row in df_aneurysms.iterrows():
                loc_coord = (loc_row["l"], loc_row["p"], loc_row["s"])

                # Draw several jittered patches around the aneurysm center
                for _ in range(POS_SAMPLES):
                    jittered_coord = jitter_coords(loc_coord, max_jitter=MAX_JITTER)
                    try:
                        patch_sitk = extract_patch_sitk(
                            sitk_img, jittered_coord, patch_size=PATCH_SIZE
                        )
                    except ValueError as e:
                        print(
                            f"Error extracting patch for series {series_id} at coord {jittered_coord}: {e}"
                        )
                        continue

                    patch_np = patch_sitk_to_numpy(patch_sitk)
                    patch_id = str(uuid.uuid4())
                    patch_filepath = patch_dir / split / f"{patch_id}.npz"

                    # Schedule async save
                    save_tasks.append(
                        executor.submit(save_patch, patch_np, patch_filepath)
                    )

                    # Build row dictionary with one key per category column
                    patch_dict = {
                        "patch_id": patch_id,
                        "series_id": series_id,
                        "modality": modality,
                        "patch_filepath": str(patch_filepath),
                        "split": split,
                        "world_coords": jittered_coord,
                        "label": 1,
                        "location": loc_row.location,
                    }
                    patch_rows.append(patch_dict)

            ### Generate Negative patches (not near any aneurysm)
            radius_vox = PATCH_SIZE // 2
            padding = 18.0  # mm
            aneurysm_coords = df_aneurysms[["l", "p", "s"]].to_numpy()  # Nx3 array

            # Negative samples
            for j in range(NEG_SAMPES):
                center_voxel = [
                    np.random.randint(radius_vox, vol_size[d] - radius_vox)
                    for d in range(3)
                ]
                # Convert from voxel index to physical (world) coordinates
                center_coord = sitk_img.TransformIndexToPhysicalPoint(center_voxel)
                # Compute distances from this candidate center to all aneurysm centers
                dists = np.linalg.norm(aneurysm_coords - np.array(center_coord), axis=1)
                # Reject candidates that are too close to any aneurysm
                if np.any(dists < padding):
                    continue  # too close to an aneurysm

                try:
                    patch_sitk = extract_patch_sitk(sitk_img, center_coord, PATCH_SIZE)
                except ValueError as e:
                    print(
                        f"Error extracting negative patch for series {series_id} at coord {center_coord}: {e}"
                    )
                    continue
                patch_np = patch_sitk_to_numpy(patch_sitk)

                patch_id = str(uuid.uuid4())
                patch_filepath = patch_dir / split / f"{patch_id}.npz"
                save_tasks.append(executor.submit(save_patch, patch_np, patch_filepath))

                patch_dict = {
                    "patch_id": patch_id,
                    "series_id": series_id,
                    "modality": modality,
                    "patch_filepath": str(patch_filepath),
                    "split": split,
                    "world_coords": center_coord,
                    "label": 0,
                    "location": np.nan,
                }
                patch_rows.append(patch_dict)

        ### Synchronize save tasks and update metadata
        for task in save_tasks:
            task.result()

        if patch_rows:
            df_rows = pd.DataFrame(patch_rows)
            df_metadata = pd.concat([df_metadata, df_rows], ignore_index=True)
            df_metadata.to_csv(patch_metadata_path, index=False)
