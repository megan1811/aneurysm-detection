import pandas as pd
from tqdm import tqdm
from utils.preprocess import (
    load_and_preprocess,
    extract_patch_sitk,
    patch_sitk_to_numpy,
    coords_to_m1p1,
)
import argparse
from pathlib import Path
import numpy as np
import torch
from utils.models import PatchClassifier
from utils.CONSTANTS import CAT_COLS, MODALITIES, LABEL_COLS

# Sliding window inference parameters
PATCH_SIZE = 32
PATCH_SIZE_OUTPUT = 32
STRIDE = PATCH_SIZE // 2  # 50% overlap
BATCH_SIZE = 8


def register_fusion_embedding_hook(model):
    """
    Register a forward hook on the fusion module to capture the
    fused patch embedding (shape: [B, fusion_out_dim]).
    """

    def hook_fn(module, input, output):
        # output is h: (B, fusion_out_dim)
        model._last_embedding = output.detach()

    handle = model.fusion.register_forward_hook(hook_fn)
    return handle


def generate_sliding_centers(vol_size, patch_size, stride):
    """
    Generate voxel-space patch centers for 3D sliding-window inference.
    Ensures full patch fits inside the volume.
    """
    radius = patch_size // 2

    xs = range(radius, vol_size[0] - radius + 1, stride)
    ys = range(radius, vol_size[1] - radius + 1, stride)
    zs = range(radius, vol_size[2] - radius + 1, stride)

    for x in xs:
        for y in ys:
            for z in zs:
                yield (x, y, z)


@torch.no_grad()
def infer_scan_sliding_window(
    sitk_img,
    modality,
    classifier,
    device,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
):
    vol_size = sitk_img.GetSize()
    radius = patch_size // 2

    # Volume-size guard
    if any(vol_size[d] <= 2 * radius for d in range(3)):
        return None

    # --- batch buffers ---
    patch_batch = []
    modality_batch = []
    coord_batch = []
    center_vox_buffer = []
    center_norm_buffer = []

    modality_idx_scalar = MODALITIES.index(modality)

    centers_vox = []
    centers_norm = []
    pres_logits = []
    loc_logits = []
    embeddings = []

    def flush_batch():
        if len(patch_batch) == 0:
            return

        patch_tensor = torch.stack(patch_batch).unsqueeze(1).to(device)
        coord_tensor = torch.stack(coord_batch).to(device)
        modality_tensor = torch.tensor(modality_batch, dtype=torch.long, device=device)

        pres, loc = classifier(
            patch_tensor,
            coord_tensor,
            modality_tensor,
        )

        pres_logits.extend(pres.detach().cpu().numpy())
        loc_logits.extend(loc.detach().cpu().numpy())
        embeddings.extend(classifier._last_embedding.detach().cpu().numpy())
        centers_vox.extend(center_vox_buffer)
        centers_norm.extend(center_norm_buffer)

        center_vox_buffer.clear()
        center_norm_buffer.clear()
        patch_batch.clear()
        coord_batch.clear()
        modality_batch.clear()

    for center in generate_sliding_centers(vol_size, patch_size, stride):
        try:
            patch_sitk = extract_patch_sitk(sitk_img, center, patch_size=patch_size)
        except ValueError:
            continue

        patch_np = patch_sitk_to_numpy(patch_sitk)
        center_norm = coords_to_m1p1(center, vol_size)

        patch_batch.append(torch.from_numpy(patch_np).float())
        coord_batch.append(torch.tensor(center_norm, dtype=torch.float32))
        modality_batch.append(modality_idx_scalar)
        center_vox_buffer.append(center)
        center_norm_buffer.append(center_norm)

        if len(patch_batch) == BATCH_SIZE:
            flush_batch()

    # Flush leftovers
    flush_batch()

    return {
        "centers_voxel": np.array(centers_vox),
        "centers_norm": np.array(centers_norm),
        "presence_logits": np.array(pres_logits),
        "location_logits": np.array(loc_logits),
        "embeddings": (np.array(embeddings) if embeddings else None),
    }


def build_scan_label(row: pd.Series) -> np.ndarray:
    """
    Build a (14,) multi-hot scan-level label vector from df_series row.
    """
    y = []

    # 13 location labels
    for loc in LABEL_COLS:
        y.append(int(row[loc]))

    return np.array(y, dtype=np.int8)


def save_scan_aggregation(
    output_dir: Path,
    series_id: str,
    scan_data: dict,
    row: pd.Series,
):
    """
    Save per-scan aggregation data for MIL / scan-level models.
    """
    # Build scan-level label
    y = build_scan_label(row)

    out_path = output_dir / f"{series_id}.npz"

    # Note: logits are stored (not probabilities) to allow flexible aggregation later
    np.savez_compressed(
        out_path,
        series_id=series_id,
        centers_voxel=scan_data["centers_voxel"].astype(np.int16),
        centers_norm=scan_data["centers_norm"].astype(np.float32),
        presence_logits=scan_data["presence_logits"].astype(np.float32),
        location_logits=scan_data["location_logits"].astype(np.float32),
        embeddings=scan_data["embeddings"].astype(np.float32),
        y=y,
        modality=MODALITIES.index(row.Modality),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("series_dir", type=Path, help="Path to series folder")
    parser.add_argument("data_dir", type=Path, help="Path to data folder")
    parser.add_argument("model_weights", type=Path, help="Path to model weights")
    parser.add_argument("output_dir", type=Path, help="Path to output folder")

    args = parser.parse_args()

    ### Create output directories if they don't exist
    output_dir = args.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.joinpath("train").mkdir(exist_ok=True)
        output_dir.joinpath("val").mkdir(exist_ok=True)
        output_dir.joinpath("test").mkdir(exist_ok=True)

    ### Read in series data
    df_series = pd.read_csv(args.data_dir / "train_processed.csv")

    ### Load classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = PatchClassifier(
        modality_num_classes=len(MODALITIES),
        num_loc_classes=len(CAT_COLS),
    )
    classifier.load_state_dict(torch.load(args.model_weights, map_location=device))
    classifier.to(device)
    classifier.eval()
    # Register hook to capture embeddings
    hook_handle = register_fusion_embedding_hook(classifier)

    ### Loop over each series
    for i, row in tqdm(
        df_series.iterrows(),
        desc="Generating aggregates",
        total=(len(df_series)),
    ):
        series_id = row.SeriesInstanceUID
        split = row.split
        modality = row.Modality

        ### Load Volume and basic preprocess / check
        try:
            sitk_img = load_and_preprocess(
                series_id, modality, base_dir=args.series_dir
            )
        except Exception as e:
            print(f"Error loading series {series_id}: {e}")
            continue

        scan_data = infer_scan_sliding_window(
            sitk_img=sitk_img,
            modality=modality,
            classifier=classifier,
            device=device,
        )
        if scan_data is None or len(scan_data["presence_logits"]) == 0:
            continue
        save_scan_aggregation(
            output_dir=output_dir / split,
            series_id=series_id,
            scan_data=scan_data,
            row=row,
        )

    # Remove hook after inference
    hook_handle.remove()
