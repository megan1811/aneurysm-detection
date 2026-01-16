import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast
from pathlib import Path


from utils.CONSTANTS import MODALITY_TO_INT, CAT_COLS


class AneurysmPatchDataset(Dataset):
    """
    Dataset for loading 3D aneurysm patches, metadata, and categorical labels.
    """

    def __init__(self, df_data: pd.DataFrame, transform=None, test: bool = False):
        """
        Init function for AneurysmPatchDataset.

        Args:
            df_data (pd.DataFrame): DataFrame containing patch filepaths, world coords,
                modality, location, and labels.
            transform (callable, optional): Optional transform applied to each sample. Defaults to None.
            test (bool, optional): If True, the dataset does not return training labels. Defaults to False.
        """
        self.df_data = df_data.reset_index(drop=True)
        self.test = test
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of rows in the underlying DataFrame.
        """
        return len(self.df_data)

    def load_patch(self, patch_filepath) -> torch.Tensor:
        """
        Load a 3D patch from an .npz file and convert it to a float tensor.

        Args:
            patch_filepath (str): Path to the .npz file containing a 'patch' array.

        Returns:
            torch.Tensor: Tensor of shape (1, D, H, W) representing the patch.
        """
        patch_npz = np.load(patch_filepath)
        patch_np = patch_npz["patch"]
        patch_tensor = torch.from_numpy(patch_np).float()
        patch_tensor = patch_tensor.unsqueeze(0)  # add channel dimension
        return patch_tensor

    def __getitem__(self, idx):
        """
        Retrieve a single sample consisting of a 3D patch, spatial coordinates,
        modality metadata, and (when not in test mode) aneurysm targets.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            dict: A dictionary containing:
                - "patch" (torch.Tensor): 3D patch tensor of shape (1, D, H, W), dtype float32.
                - "coords" (torch.Tensor): Patch-center coordinates of shape (3,), dtype float32
                  (expected to be scaled to [-1, 1]).
                - "modality" (torch.Tensor): Modality index of shape (), dtype long (for nn.Embedding).
                - "y_pres" (torch.Tensor, optional): Presence label (0.0 or 1.0), shape (), dtype float32
                  (for BCEWithLogitsLoss). Present only when test=False.
                - "y_loc" (torch.Tensor, optional): Location class index in [0, 12], shape (), dtype long
                  (for CrossEntropyLoss). Only meaningful when y_pres == 1; for y_pres == 0 it is set
                  to a dummy value (0). Present only when test=False.
        """
        row = self.df_data.iloc[idx]

        patch_tensor = self.load_patch(row["patch_filepath"])

        coords = ast.literal_eval(row["center_coords"].replace("np.float64", ""))
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        output = {
            "patch": patch_tensor,
            "coords": coords_tensor,  # assuming row['coords'] is iterable of length 3
            "modality": torch.tensor(
                MODALITY_TO_INT[row["modality"]], dtype=torch.long
            ),  # assuming label/int
        }

        # Apply transforms if provided
        if self.transform:
            output = self.transform(output)

        if not self.test:
            y_pres = int(row["label"])
            output["y_pres"] = torch.tensor(y_pres, dtype=torch.float32)

            if y_pres == 1:
                location = row["location"]
                if location not in CAT_COLS:
                    raise ValueError(f"Positive patch has invalid location: {location}")
                loc_idx = CAT_COLS.index(location)
            else:
                loc_idx = 0  # dummy value, ignored by loss via masking

            output["y_loc"] = torch.tensor(loc_idx, dtype=torch.long)

        return output


class ScanDataset(Dataset):
    """
    Dataset for scan-level aggregation.

    One item corresponds to one scan (.npz file) and contains
    variable-length patch-level outputs + fixed scan-level labels.
    """

    def __init__(self, scan_files: list[Path]):
        """
        Args:
            scan_files (list): List of paths to .npz files.
        """
        self.scan_files = scan_files

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        scan_path = self.scan_files[idx]
        data = np.load(scan_path, allow_pickle=True)

        # Patch-level data (variable length)
        centers_norm = torch.from_numpy(data["centers_norm"]).float()  # (N, 3)
        presence_logits = (
            torch.from_numpy(data["presence_logits"]).float().unsqueeze(-1)
        )  # (N,1)
        location_logits = torch.from_numpy(data["location_logits"]).float()  # (N, 13)
        embeddings = torch.from_numpy(data["embeddings"]).float()  # (N, D)

        # Scan-level labels
        y = torch.from_numpy(data["y"]).float()  # (14,)
        y_presence = y[-1]
        y_location = y[:13]

        # Metadata (keep as Python types)
        series_id = str(data["series_id"])
        modality = int(data["modality"])

        return {
            "presence_logits": presence_logits,
            "location_logits": location_logits,
            "embeddings": embeddings,
            "centers_norm": centers_norm,
            "y_presence": y_presence,
            "y_location": y_location,
            "y": y,
            "series_id": series_id,
            "modality": modality,
        }
