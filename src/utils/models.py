import torch
import torch.nn as nn


class CoordMLP(nn.Module):
    """
    Small MLP to embed 3D world coordinates into a fixed-length vector.
    """

    def __init__(self, in_dim=3, hidden_dim=32, out_dim=32):
        """
        Initialize the coordinate MLP Head.

        Args:
            in_dim (int, optional): Input dimension of the coordinates
                (default is 3 for LPS coordinates).
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 32.
            out_dim (int, optional): Output embedding dimension for the
                coordinates. Defaults to 32.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords) -> torch.Tensor:
        """
        Embed raw 3D coordinates into a learned feature space.

        Args:
            coords (torch.Tensor): Tensor of shape (B, 3) containing
                continuous world coordinates (e.g., LPS).

        Returns:
            torch.Tensor: Tensor of shape (B, out_dim) containing the
                coordinate embeddings.
        """
        return self.net(coords)


class PatchClassifier(nn.Module):
    """
    3D patch classifier that combines image features, modality embeddings,
    and embedded world coordinates to predict aneurysm-related classes.
    """

    def __init__(self, modality_num_classes, num_classes, coor_emb_dim=32):
        """
        Initialize the PatchClassifier model.

        Args:
            modality_num_classes (int): Number of distinct imaging modalities
                (used for the modality embedding lookup).
            num_classes (int): Number of output classes for classification
                (e.g., anatomical locations + "no aneurysm" class).
            coor_emb_dim (int, optional): Output embedding dimension of the
                coordinate MLP. Defaults to 32.
        """
        super().__init__()

        self.backbone = torch.hub.load(
            "Warvito/MedicalNet-models", "medicalnet_resnet10_23datasets"
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        backbone_embedding_dim = 512  # output feature dim of the backbone

        self.modality_embedding = nn.Embedding(modality_num_classes, 16)

        self.coord_mlp = CoordMLP(in_dim=3, hidden_dim=32, out_dim=coor_emb_dim)

        # Classifier MLP layers
        decoder_input_dim = backbone_embedding_dim + 16 + coor_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(decoder_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, patch_volume, world_coords, modality_idx) -> torch.Tensor:
        """
        Forward pass of the patch classifier.

        Args:
            patch_volume (torch.Tensor): 3D patch tensor of shape
                (B, 1, D, H, W), e.g. (B, 1, 32, 32, 32).
            # TODO: normalize world coordinates in dataset preparation
            world_coords (torch.Tensor): Tensor of shape (B, 3) containing
                raw world coordinates (e.g., LPS) for the patch center.
            modality_idx (torch.Tensor): Tensor of shape (B,) with integer
                modality indices (dtype torch.long).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes) representing
                the unnormalized class scores.
        """

        x = self.backbone(patch_volume)
        x = self.pool(x).view(x.size(0), -1)  # shape (B, C)

        m = self.modality_embedding(modality_idx)  # (B, 16)
        c = self.coord_mlp(world_coords)

        combined = torch.cat([x, m, c], dim=1)  # (B, decoder_input_dim)
        out = self.classifier(combined)  # (B, num_classes), raw logits
        return out
