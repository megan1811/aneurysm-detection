import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast

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
        modality metadata, and (when not in test mode) aneurysm labels.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            dict: A dictionary containing:
                - "patch" (torch.Tensor): 3D patch tensor of shape (1, D, H, W).
                - "coords" (torch.Tensor): Tensor of world coordinates (3,).
                - "modality" (torch.Tensor): Encoded modality index.
                - "y" (torch.Tensor, optional): Location class index (train mode only).
                - "label" (int, optional): Binary aneurysm-presence label (train mode only).
                - "location" (str, optional): Anatomical location of the aneurysm;
                    must be present in CAT_COLS (train mode only).
        """
        row = self.df_data.iloc[idx]

        patch_tensor = self.load_patch(row["patch_filepath"])

        coords = ast.literal_eval(row["world_coords"].replace("np.float64", ""))
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
            location = row["location"]
            if location not in CAT_COLS:
                output["y"] = torch.tensor(len(CAT_COLS)).long()  # no aneurysm present
            else:
                output["y"] = torch.tensor(
                    CAT_COLS.index(location)
                ).long()  # location of aneurysm
            # output["label"] = row["label"]
            # output["location"] = row["location"]
        # print(output)

        return output
